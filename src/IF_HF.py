import torch
import logging
from typing import List, Dict, Callable, Optional
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    # 辅助函数：构建单个 Turn 的 ID 和 Label
    def _build_turn(role, content, is_train=False):
        # Role Header: <|im_start|>role\n
        role_ids = [im_start_id] + tokenizer.encode(role, add_special_tokens=False) + nl_tokens
        # Content
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        # Footer: <|im_end|>\n
        footer_ids = [im_end_id] + nl_tokens

        # 拼接完整的 input_ids
        full_ids = role_ids + content_ids + footer_ids

        # 构建 Labels
        if is_train:
            # User 部分不计算 Loss (-100)
            labels = [-100] * len(role_ids) + content_ids + footer_ids
        else:
            labels = [-100] * len(full_ids)

        return full_ids, labels

    new_input_ids = []
    new_labels = []
    new_masks = []

    for inp, outp in zip(inputs, outputs):
        input_ids, labels = [], []

        sys_ids, sys_labels = _build_turn("system", system_message, is_train=False)
        input_ids += sys_ids
        labels += sys_labels

        # --- User Turn ---
        user_ids, user_labels = _build_turn("user", inp, is_train=False)
        input_ids += user_ids
        labels += user_labels

        # --- Assistant Turn: output 是我们希望模型学习的内容
        asst_ids, asst_labels = _build_turn("assistant", outp, is_train=True)
        input_ids += asst_ids
        labels += asst_labels

        # Truncation
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        attention_mask = [1] * len(input_ids)

        # Padding
        padding_len = max_len - len(input_ids)
        if padding_len > 0:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else im_end_id
            input_ids = input_ids + [pad_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            labels = labels + [-100] * padding_len

        new_input_ids.append(input_ids)
        new_labels.append(labels)
        new_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(new_input_ids),
        "attention_mask": torch.tensor(new_masks),
        "labels": torch.tensor(new_labels)
    }

def compute_loss_causallm(model: nn.Module, batch: BatchDict, device: torch.device) -> torch.Tensor:
    """
    计算 Causal LM 的 Loss. Causal LM通常是 next-token prediction，这里我让 labels=input_ids, 它内部会自动对labels做shift -1
    """
    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in ['input_ids', 'attention_mask', 'labels']
    }
    outputs = model(**inputs)
    return outputs.loss


def compute_gradients(
        model: nn.Module,
        batch: BatchDict,
        param_filter_fn: Optional[ParamFilterFn],
        device: torch.device
) -> List[torch.Tensor]:
    """
    计算单个 Batch 针对特定参数的梯度。
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    loss = compute_loss_causallm(model, batch, device)

    # 筛选参数
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param_filter_fn is None or param_filter_fn(name, param):
                params.append(param)

    if not params:
        sample_names = [n for n, _ in list(model.named_parameters())[-10:]]
        error_msg = (
            "No parameters were selected for gradient computation.\n"
            "Possible reasons:\n"
            "1. `param_filter_fn` name matching is wrong for this model architecture.\n"
            "2. Parameters are frozen (requires_grad=False).\n"
            f"Last 10 layer names in model: {sample_names}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 计算梯度
    grads = torch.autograd.grad(loss, params, create_graph=False)

    return list(grads)


def calc_loss_scalar(model: nn.Module, batch: BatchDict, device: torch.device) -> float:
    """计算 Loss 并返回 Python float"""
    model.eval()
    with torch.no_grad():
        return compute_loss_causallm(model, batch, device).item()


class BaseInfluenceFunction:
    def __init__(
            self,
            dl_train: DataLoader,
            model: nn.Module,
            device: torch.device,
            param_filter_fn: Optional[ParamFilterFn] = None,
    ):
        self.dl_train = dl_train
        self.model = model
        self.device = device
        self.param_filter_fn = param_filter_fn

        # 【关键修改】不再缓存梯度，只缓存训练数据 (Data Cache)
        # 节省显存，且速度更快
        self.train_data_cache: List[BatchDict] = []
        self._cache_train_data()
        self.n_train = len(self.train_data_cache)

    def _cache_train_data(self):
        logger.info("Caching training data to CPU (for fast forward pass)...")
        # 直接遍历 DataLoader，将 Batch 存入内存 (CPU)
        for batch in self.dl_train:
            # 移除不需要的 key，只保留 Tensor 并移至 CPU
            batch_cpu = {
                k: v.cpu()
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            self.train_data_cache.append(batch_cpu)
        logger.info(f"Cached {len(self.train_data_cache)} training samples.")

    def query_influence(self, query_batch: BatchDict) -> List[float]:
        raise NotImplementedError


class EmpiricalIF(BaseInfluenceFunction):
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
        l_test_base = calc_loss_scalar(self.model, query_batch, self.device)

        l_train_base_list = []
        for train_batch in self.train_data_cache:
            l_train_base_list.append(calc_loss_scalar(self.model, train_batch, self.device))

        # 备份参数
        snapshot = self.get_param_snapshot(self.model, self.param_filter_fn)

        # Part A: Descent on Test (L')
        logger.info("Step A: Descent on Test Gradient...")
        # Theta' = Theta - lr * Test_Grad
        self.apply_gradient_update(self.model, test_grads, self.param_filter_fn, lr=lr)

        # L_test'
        l_test_descent = calc_loss_scalar(self.model, query_batch, self.device)
        delta_test_descent = l_test_descent - l_test_base  # 通常应该是负数

        # L_train' (遍历所有训练样本)
        l_train_descent_list = []
        for train_batch in self.train_data_cache:
            l_train_descent_list.append(calc_loss_scalar(self.model, train_batch, self.device))

        # 恢复参数
        self.restore_params(self.model, snapshot, self.param_filter_fn)

        # Part B: Ascent on Test (L'')
        logger.info("Step B: Ascent on Test Gradient...")
        # Theta'' = Theta + lr * Test_Grad (即 lr = -lr)
        self.apply_gradient_update(self.model, test_grads, self.param_filter_fn, lr=-lr)

        # L_test''
        l_test_ascent = calc_loss_scalar(self.model, query_batch, self.device)
        delta_test_ascent = l_test_ascent - l_test_base  # 通常应该是正数

        # L_train'' (遍历所有训练样本)
        l_train_ascent_list = []
        for train_batch in self.train_data_cache:
            l_train_ascent_list.append(calc_loss_scalar(self.model, train_batch, self.device))

        # 恢复参数
        self.restore_params(self.model, snapshot, self.param_filter_fn)

        # Final Score Calculation
        logger.info("Computing final scores...")
        influences = []

        # 遍历每个训练样本进行计算
        for i in range(self.n_train):
            # Term 1: Descent Component
            # (L_test' - L_test) * (L_train' - L_train)
            delta_train_descent = l_train_descent_list[i] - l_train_base_list[i]
            term_1 = delta_test_descent * delta_train_descent

            # Term 2: Ascent Component
            # (L_test'' - L_test) * (L_train'' - L_train)
            delta_train_ascent = l_train_ascent_list[i] - l_train_base_list[i]
            term_2 = delta_test_ascent * delta_train_ascent

            # Average
            score = (term_1 + term_2) / 2
            influences.append(score)

        return influences

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

    # 处理训练集
    train_dataset = Dataset.from_dict(list_of_dicts_to_dict_of_lists(train_texts))

    # 绑定 tokenizer
    process_func_partial = partial(
        process_func_chatml,
        tokenizer=tokenizer,
        max_len=MAX_LENGTH,
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )

    train_dataset = train_dataset.map(process_func_partial, batched=True, remove_columns=["input", "output"])
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # 处理测试集
    test_dataset = Dataset.from_dict(list_of_dicts_to_dict_of_lists([test_data_dict]))
    test_dataset = test_dataset.map(process_func_partial, batched=True, remove_columns=["input", "output"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 提取 Tensor 字典
    query_batch = {
        "input_ids": test_dataset[0]["input_ids"].unsqueeze(0).to(device),
        "attention_mask": test_dataset[0]["attention_mask"].unsqueeze(0).to(device),
        "labels": test_dataset[0]["labels"].unsqueeze(0).to(device)
    }

    # 运行 Influence Function
    def filter_params(name: str, param: nn.Parameter) -> bool:
        return target_layer_keyword in name and param.requires_grad

    logger.info("Initializing Empirical IF...")
    eif = EmpiricalIF(dl_train=train_loader, model=model, device=device, param_filter_fn=filter_params)

    logger.info("Querying influence...")
    scores = eif.query_influence(query_batch, lr=1e-4)

    logger.info("Results:")
    # 打印结果时，只截取 output 的前几十个字符展示
    for i, score in enumerate(scores):
        preview = train_texts[i]['output'].replace('\n', '\\n')[:50]
        print(f"Sample {i} | Score: {score:.6f} | Output: {preview}...")


if __name__ == '__main__':
    main()