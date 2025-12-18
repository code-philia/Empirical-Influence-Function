import torch
import logging
from typing import List, Dict, Callable, Optional
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类型别名
BatchDict = Dict[str, torch.Tensor]
ParamFilterFn = Callable[[str, nn.Parameter], bool]

MAX_LENGTH = 8192 * 5

def list_of_dicts_to_dict_of_lists(data_list):
    return {
        "input": [d["input"] for d in data_list],
        "output": [d["output"] for d in data_list]
    }


def process_func_chatml(examples, tokenizer, max_len=MAX_LENGTH, system_message="You are a helpful assistant."):
    """
    将 input/output 转换为 ChatML 格式：
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {input}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>

    并仅对 assistant 的回复部分计算 Loss。
    """
    inputs = examples["input"]
    outputs = examples["output"]

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    def _build_turn(role, content, is_train=False):
        role_ids = [im_start_id] + tokenizer.encode(role, add_special_tokens=False) + nl_tokens
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = [im_end_id] + nl_tokens
        full_ids = role_ids + content_ids + footer_ids

        if is_train:
            # -100 表示忽略计算 Loss
            labels = [-100] * len(role_ids) + content_ids + footer_ids
        else:
            labels = [-100] * len(full_ids)
        return full_ids, labels

    new_input_ids = []
    new_labels = []
    # 注意：不再需要手动生成 attention_mask，DataCollator 会自动处理

    for inp, outp in zip(inputs, outputs):
        input_ids, labels = [], []

        # System
        sys_ids, sys_labels = _build_turn("system", system_message, is_train=False)
        input_ids += sys_ids
        labels += sys_labels

        # User
        user_ids, user_labels = _build_turn("user", inp, is_train=False)
        input_ids += user_ids
        labels += user_labels

        # Assistant
        asst_ids, asst_labels = _build_turn("assistant", outp, is_train=True)
        input_ids += asst_ids
        labels += asst_labels

        # Truncation (仅截断，不 Padding)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        new_input_ids.append(input_ids)
        new_labels.append(labels)

    return {
        "input_ids": new_input_ids,
        "labels": new_labels
    }


def compute_loss_per_sample(model: nn.Module, batch: BatchDict, device: torch.device) -> torch.Tensor:

    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in ['input_ids', 'attention_mask', 'labels']
    }

    # 前向传播 (不自动计算 Loss)
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits  # [B, Seq_Len, Vocab]

    # Causal LM 的预测是基于前一个 token 预测下一个，所以 logits 要左移，labels 要右移
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()

    # 不进行 reduction (mean/sum)
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    # [B * (Seq-1), Vocab] vs [B * (Seq-1)]
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    token_losses = token_losses.view(shift_labels.size())

    valid_mask = shift_labels.ne(-100).float()

    # Sum(Loss) / Count(Valid_Tokens)
    sum_loss = (token_losses * valid_mask).sum(dim=1)
    num_valid = valid_mask.sum(dim=1)

    per_sample_loss = sum_loss / (num_valid + 1e-9)

    return per_sample_loss


# 为了兼容 compute_gradients，保留原来的标量计算函数
def compute_loss_scalar(model, batch, device):
    loss_tensor = compute_loss_per_sample(model, batch, device)
    return loss_tensor.mean()


def compute_gradients(model, batch, param_filter_fn, device):
    model.eval()
    model.zero_grad(set_to_none=True)
    loss = compute_loss_scalar(model, batch, device)

    params = [p for n, p in model.named_parameters() if
              p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]
    if not params: raise ValueError("No params selected")

    return list(torch.autograd.grad(loss, params))



class EmpiricalIF:
    def __init__(self, dl_train, model, device, param_filter_fn=None):
        self.dl_train = dl_train
        self.model = model
        self.device = device
        self.param_filter_fn = param_filter_fn

        # 缓存训练数据 (Cache)
        self.train_batches = []
        logger.info("Caching training data (Dynamic Padding happens here)...")
        for batch in self.dl_train:
            self.train_batches.append(batch)

        self.n_samples = sum(len(b['input_ids']) for b in self.train_batches)
        logger.info(f"Cached {len(self.train_batches)} batches, total {self.n_samples} samples.")

    def _get_train_losses(self) -> torch.Tensor:
        all_losses = []
        with torch.no_grad():
            for batch in self.train_batches:
                batch_loss = compute_loss_per_sample(self.model, batch, self.device)
                all_losses.append(batch_loss.cpu())  # 存回 CPU
        return torch.cat(all_losses)  # [N_Train]

    @staticmethod
    def get_param_snapshot(model: nn.Module, param_filter_fn: Optional[ParamFilterFn]) -> List[torch.Tensor]:
        snapshot = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    snapshot.append(param.detach().clone())
        return snapshot

    @staticmethod
    def restore_params(model: nn.Module, snapshot: List[torch.Tensor], param_filter_fn: Optional[ParamFilterFn]):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    param.data.copy_(snapshot[idx])
                    idx += 1

    @staticmethod
    def apply_gradient_update(
            model: nn.Module,
            grads: List[torch.Tensor],
            param_filter_fn: Optional[ParamFilterFn],
            lr: float
    ):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    if grads[idx] is not None:
                        # 转换 device 和 dtype
                        grad_device = grads[idx].to(param.device).to(param.dtype)
                        param.data -= lr * grad_device
                    idx += 1

    def query_influence(self, query_batch: BatchDict, lr: float = 1e-4) -> List[float]:
        """
        Implementation of:
        Term 1: (L_test' - L_test) * (L_train' - L_train)  [from Test Descent]
        Term 2: (L_test'' - L_test) * (L_train'' - L_train) [from Test Ascent]
        Score = (Term 1 + Term 2) / 2
        """

        # 1. 计算 Test Sample 的梯度 (Perturbation Source)
        logger.info("Computing gradient for query (test) sample...")
        test_grads = compute_gradients(self.model, query_batch, self.param_filter_fn, self.device)

        # 2. 计算 Base Loss (L_test, L_train) - 更新前
        logger.info("Calculating Base Losses...")
        l_test_base = compute_loss_scalar(self.model, query_batch, self.device).item()
        l_train_base = self._get_train_losses()  # Vector [N]

        # 备份参数
        snapshot = self.get_param_snapshot(self.model, self.param_filter_fn)

        # Part A: Descent on Test (L')
        logger.info("Step A: Descent on Test Gradient...")
        # Theta' = Theta - lr * Test_Grad
        self.apply_gradient_update(self.model, test_grads, self.param_filter_fn, lr=lr)

        # L_test'
        l_test_des = compute_loss_scalar(self.model, query_batch, self.device).item()
        l_train_des = self._get_train_losses()  # Vector [N]

        # 恢复参数
        self.restore_params(self.model, snapshot, self.param_filter_fn)

        # Part B: Ascent on Test (L'')
        logger.info("Step B: Ascent on Test Gradient...")
        # Theta'' = Theta + lr * Test_Grad (即 lr = -lr)
        self.apply_gradient_update(self.model, test_grads, self.param_filter_fn, lr=-lr)

        # L_test''
        l_test_asc = compute_loss_scalar(self.model, query_batch, self.device).item()
        l_train_asc = self._get_train_losses()  # Vector [N]

        # 恢复参数
        self.restore_params(self.model, snapshot, self.param_filter_fn)

        # Final Score Calculation
        logger.info("Computing final scores...")

        # Scalar deltas
        delta_test_des = l_test_des - l_test_base
        delta_test_asc = l_test_asc - l_test_base

        # Vector deltas [N]
        delta_train_des = l_train_des - l_train_base
        delta_train_asc = l_train_asc - l_train_base

        # Term 1 & 2
        term_1 = delta_test_des * delta_train_des
        term_2 = delta_test_asc * delta_train_asc

        scores = (term_1 + term_2) / 2
        return scores.tolist()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 模型加载 (Qwen2.5-Coder-1.5B)
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    logger.info(f"Loading model: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Error loading model via AutoModel: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "device"):
        device = model.device

    target_layer_keyword = "model.layers.27.mlp"
    logger.info(f"Enabling gradients for layers matching: '{target_layer_keyword}'")
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 仅解冻目标层
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if target_layer_keyword in name:
            param.requires_grad = True
            unfrozen_count += 1
    logger.info(f"Total parameters unfrozen: {unfrozen_count}")

    if unfrozen_count == 0:
        logger.error(f"Failed to find any layers matching '{target_layer_keyword}'. Check model structure.")
        return

    # ==========================================
    # 准备代码相关数据
    # ==========================================
    train_texts = [
        {
            "input":  "This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/urfave/cli Below is the code repository: github.com/urfave/cli/ Below is the imported package path \"fmt\"; \"time\" The receiver struct definitions of the function is type stringSliceArgs struct {\n\tv []string\n}\n\n// Methods:\n- func (a *stringSliceArgs) Get(n int) string\n- func (a *stringSliceArgs) First() string\n- func (a *stringSliceArgs) Tail() *ast.ArrayType\n- func (a *stringSliceArgs) Len() int\n- func (a *stringSliceArgs) Present() bool\n- func (a *stringSliceArgs) Slice() *ast.ArrayType\n The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is *ast.ArrayType The code snippets before the function is func (a *stringSliceArgs) Get(n int) string {\n\tif len(a.v) > n {\n\t\treturn a.v[n]\n\t}\n\treturn \"\"\n}\n\nfunc (a *stringSliceArgs) First() string {\n\treturn a.Get(0)\n} And here is the function you are asked to complete func (a *stringSliceArgs) Tail() *ast.ArrayType\nfunc (a *stringSliceArgs) Tail() []string {\n\tif a.Len() >= 2 { <MID> \t}\n\n\treturn []string{}\n} Ensure that only missing codes marked as <MID> are returned. ",
            "output": "\t\ttail := a.v[1:]\n\t\tret := make([]string, len(tail))\n\t\tcopy(ret, tail)\n\t\treturn ret",
        },
        {
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "\t\ttail := a.v[1:]\n\t\tret := make([]string, len(tail))\n\t\tcopy(ret, tail)\n\t\treturn ret",
        },
        {
            "input": "class MyDataset(Dataset): pass",
            "output": "import torch\nx = torch.randn(10)",
        }
    ]

    test_data_dict =  {
            "input":  "This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/urfave/cli Below is the code repository: github.com/urfave/cli/ Below is the imported package path \"fmt\"; \"time\" The receiver struct definitions of the function is type stringSliceArgs struct {\n\tv []string\n}\n\n// Methods:\n- func (a *stringSliceArgs) Get(n int) string\n- func (a *stringSliceArgs) First() string\n- func (a *stringSliceArgs) Tail() *ast.ArrayType\n- func (a *stringSliceArgs) Len() int\n- func (a *stringSliceArgs) Present() bool\n- func (a *stringSliceArgs) Slice() *ast.ArrayType\n The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is *ast.ArrayType The code snippets before the function is func (a *stringSliceArgs) Get(n int) string {\n\tif len(a.v) > n {\n\t\treturn a.v[n]\n\t}\n\treturn \"\"\n}\n\nfunc (a *stringSliceArgs) First() string {\n\treturn a.Get(0)\n} And here is the function you are asked to complete func (a *stringSliceArgs) Tail() *ast.ArrayType\nfunc (a *stringSliceArgs) Tail() []string {\n\tif a.Len() >= 2 { <MID> \t}\n\n\treturn []string{}\n} Ensure that only missing codes marked as <MID> are returned. ",
            "output": "import torch\nx = torch.randn(10)"
    }

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # 启用动态 Padding
        label_pad_token_id=-100
    )

    process_func = partial(process_func_chatml, tokenizer=tokenizer)

    # Train Dataset
    train_ds = Dataset.from_dict(list_of_dicts_to_dict_of_lists(train_texts))
    train_ds = train_ds.map(process_func, batched=True, remove_columns=["input", "output"])

    BATCH_SIZE = 3  # 根据显存调整，如果显存够大，可以设为 4, 8, 16
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator  # 使用 collator
    )

    # Test Dataset (Query)
    test_ds = Dataset.from_dict(list_of_dicts_to_dict_of_lists([test_data_dict]))
    test_ds = test_ds.map(process_func, batched=True, remove_columns=["input", "output"])
    # Query 只有一个样本，BatchSize=1 即可
    query_batch = data_collator([test_ds[0]])
    for k, v in query_batch.items():
        query_batch[k] = v.to(device)

    # 运行优化后的 IF
    def filter_params(n, p):
        return target_layer_keyword in n and p.requires_grad

    eif = EmpiricalIF(train_loader, model, device, filter_params)
    scores = eif.query_influence(query_batch, lr=1e-4)

    logger.info("Results:")
    for i, score in enumerate(scores):
        print(f"Sample {i} | Score: {score:.6e}")


if __name__ == '__main__':
    main()