from utils import ContrastiveLossWrapper, calc_pair_loss, grad_pair_loss
import torch
from typing import Callable, List, Tuple
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random

class PairWiseEmpiricalIF:
    """
    The model must implement a dual-input interface:
    - forward(doc_inputs, code_inputs) -> (doc_embeddings, code_embeddings)
        - doc_inputs: torch.Tensor of shape (batch_size, input_dim)
        - code_inputs: torch.Tensor of shape (batch_size, input_dim)
        - doc_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        - code_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
    
    Sample of dataloader: (doc_tensor, code_tensor)
        - doc_tensor: document representation vector
        - code_tensor: code representation vector
    """
    def __init__(
        self,
        dl_train: DataLoader,
        model: nn.Module,
        param_filter_fn: Callable[[str, nn.Parameter], bool] = None,
        criterion: nn.Module = ContrastiveLossWrapper(),
    ):
        self.dl_train = dl_train
        self.model = model
        self.criterion = criterion
        self.param_filter_fn = param_filter_fn
        self.train_samples = list(dl_train.dataset)
        self.num_train_samples = len(self.train_samples)

    @staticmethod
    def get_filtered_param_snapshot(
        model: nn.Module,
        param_filter_fn: Callable[[str, nn.Parameter], bool] = None
    ) -> List[torch.Tensor]:
        return [p.detach().clone() for n, p in model.named_parameters() if (param_filter_fn is None or param_filter_fn(n, p))]

    @staticmethod
    def restore_params(
        model: nn.Module,
        param_snapshot: List[torch.Tensor],
        param_filter_fn: Callable[[str, nn.Parameter], bool] = None
    ):
        idx = 0
        for n, p in model.named_parameters():
            if param_filter_fn is None or param_filter_fn(n, p):
                p.data.copy_(param_snapshot[idx].to(p.device))
                idx += 1

    @staticmethod
    def apply_gradient_update(
        model: nn.Module,
        grad_tensors: List[torch.Tensor],
        param_filter_fn: Callable[[str, nn.Parameter], bool],
        lr: float
    ):
        idx = 0
        for n, p in model.named_parameters():
            if param_filter_fn is None or param_filter_fn(n, p):
                grad_tensor = grad_tensors[idx].to(p.device)
                p.data -= lr * grad_tensor
                idx += 1

    def query_influence(
        self,
        input_pair: Tuple[torch.Tensor, torch.Tensor],
        query_is_positive: bool, # True for positive pair, False for negative pair
        lr: float = 1e-4,
        num_negative_samples: int = 16 # Number of negative samples per sample
    ) -> List[Tuple[int, int, float, str]]:
        """
        Compute the influence of all training samples on the query pair.
        Each training sample i samples num_negative_samples negative samples j.
        Returns: (doc_index, code_index, influence_value, "positive" or "negative")
        """
        # Ensure inputs are batch-shaped
        query_doc, query_code = input_pair
        query_doc = query_doc.unsqueeze(0) if query_doc.dim() == 1 else query_doc
        query_code = query_code.unsqueeze(0) if query_code.dim() == 1 else query_code
        
        # Computing gradient for the query pair
        query_grad = grad_pair_loss(
            self.model, self.criterion, query_doc, query_code, query_is_positive, self.param_filter_fn
        )

        # Backup original model parameters
        before_update = self.get_filtered_param_snapshot(self.model, self.param_filter_fn)

        # Compute training sample loss changes before/after perturbation
        all_influences = []

        # --- Positive pairs ---
        # Loss before perturbation
        print("Computing original loss for positive training pairs...")
        positive_losses_before = calc_pair_loss(self.model, self.criterion, self.train_samples, is_positive=True)

        # --- Negative pairs ---
        # For each doc_i, sample num_negative_samples negative code_j
        print(f"Sampling {num_negative_samples} negative samples per training document...")
        negative_sample_info = []
        possible_j_indices = list(range(self.num_train_samples))
        for i in range(self.num_train_samples):
            sampled_j_indices = random.sample([idx for idx in possible_j_indices if idx != i], num_negative_samples)
            doc_i = self.train_samples[i][0]
            for j in sampled_j_indices:
                code_j = self.train_samples[j][1]
                negative_sample_info.append({'i': i, 'j': j, 'data': (doc_i, code_j)})
        
        negative_samples_data = [info['data'] for info in negative_sample_info]
        
        # Loss before perturbation
        print("Computing original loss for negative training pairs...")
        negative_losses_before = calc_pair_loss(self.model, self.criterion, negative_samples_data, is_positive=False)

        # --- Simulate gradient descent ---
        print("Simulating gradient descent...")
        self.apply_gradient_update(self.model, query_grad, self.param_filter_fn, lr=lr)
        positive_losses_after_descent = calc_pair_loss(self.model, self.criterion, self.train_samples, is_positive=True)
        negative_losses_after_descent = calc_pair_loss(self.model, self.criterion, negative_samples_data, is_positive=False)
        self.restore_params(self.model, before_update, self.param_filter_fn)

        # --- Simulate gradient ascent ---
        print("Simulating gradient ascent...")
        self.apply_gradient_update(self.model, query_grad, self.param_filter_fn, lr=-lr)
        positive_losses_after_ascent = calc_pair_loss(self.model, self.criterion, self.train_samples, is_positive=True)
        negative_losses_after_ascent = calc_pair_loss(self.model, self.criterion, negative_samples_data, is_positive=False)
        self.restore_params(self.model, before_update, self.param_filter_fn)
        
        # 4. Compute influence values
        query_loss_before = calc_pair_loss(self.model, self.criterion, [(query_doc, query_code)], query_is_positive)[0]
        # Reapply gradient descent to compute query loss change
        self.apply_gradient_update(self.model, query_grad, self.param_filter_fn, lr=lr)
        query_loss_after_descent = calc_pair_loss(self.model, self.criterion, [(query_doc, query_code)], query_is_positive)[0]
        self.restore_params(self.model, before_update, self.param_filter_fn)
        # Reapply gradient ascent to compute query loss change
        self.apply_gradient_update(self.model, query_grad, self.param_filter_fn, lr=-lr)
        query_loss_after_ascent = calc_pair_loss(self.model, self.criterion, [(query_doc, query_code)], query_is_positive)[0]
        self.restore_params(self.model, before_update, self.param_filter_fn)
        
        query_loss_change_descent = query_loss_after_descent - query_loss_before
        query_loss_change_ascent = query_loss_after_ascent - query_loss_before

        # --- positive-pair influence ---
        for i in range(self.num_train_samples):
            pos_loss_change_descent = positive_losses_after_descent[i] - positive_losses_before[i]
            pos_loss_change_ascent = positive_losses_after_ascent[i] - positive_losses_before[i]
            influence_pos = (query_loss_change_descent * pos_loss_change_descent + 
                             query_loss_change_ascent * pos_loss_change_ascent) / 2
            # Return (doc_idx, code_idx, influence, type)
            all_influences.append((i, i, influence_pos, "positive"))

        # --- negative-pair influence ---
        for idx, info in enumerate(negative_sample_info):
            i, j = info['i'], info['j']
            neg_loss_change_descent = negative_losses_after_descent[idx] - negative_losses_before[idx]
            neg_loss_change_ascent = negative_losses_after_ascent[idx] - negative_losses_before[idx]
            influence_neg = (query_loss_change_descent * neg_loss_change_descent +
                             query_loss_change_ascent * neg_loss_change_ascent) / 2
            # Return (doc_idx, code_idx, influence, type)
            all_influences.append((i, j, influence_neg, "negative"))
            
        # Sort and return by influence value (index 2)
        all_influences.sort(key=lambda x: x[2], reverse=True)
        return all_influences


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Model configuration
    input_dim = 768  # e.g., BERT embedding dimension
    embedding_dim = 256
    batch_size = 32
    num_samples = 100
    
    print(f"Input dimension: {input_dim}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of training samples: {num_samples}")
    
    # Create synthetic training data
    # In practice, these would be real document and code embeddings
    doc_data = torch.randn(num_samples, input_dim)
    code_data = torch.randn(num_samples, input_dim)
    
    # Create dataset and dataloader
    # Dataset format: each item is (doc_tensor, code_tensor)
    train_dataset = TensorDataset(doc_data, code_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"doc_batch shape: {doc_data[:batch_size].shape}")
    print(f"code_batch shape: {code_data[:batch_size].shape}")
    
    # Initialize model
    class DualEncoderModel(nn.Module):
        """
        Example dual-encoder model that accepts both doc_inputs and code_inputs.
        Returns both doc_embeddings and code_embeddings.
        """
        def __init__(self, input_dim=768, embedding_dim=256):
            super().__init__()
            self.doc_encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim)
            )
            self.code_encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim)
            )
        
        def forward(self, doc_inputs, code_inputs):
            """
            Args:
                doc_inputs: torch.Tensor of shape (batch_size, input_dim)
                code_inputs: torch.Tensor of shape (batch_size, input_dim)
            Returns:
                doc_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
                code_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
            """
            doc_embeddings = self.doc_encoder(doc_inputs)
            code_embeddings = self.code_encoder(code_inputs)
            return doc_embeddings, code_embeddings
    
    model = DualEncoderModel(input_dim=input_dim, embedding_dim=embedding_dim)
    print(f"\nModel initialized: {model.__class__.__name__}")
    print("Model interface:")
    print("  - forward(doc_inputs, code_inputs) -> (doc_embeddings, code_embeddings)")
    print(f"  - Input shapes: ({batch_size}, {input_dim}) each")
    print(f"  - Output shapes: ({batch_size}, {embedding_dim}) each")
    
    # Parameter filter function (optional)
    def param_filter_fn(name, param):
        return 'encoder' in name
    
    print(f"\nParameter filter: focusing on final layers only")
    filtered_params = [(name, param.shape) for name, param in model.named_parameters() 
                      if param_filter_fn(name, param)]
    print(f"Filtered parameters: {len(filtered_params)} layers")
    for name, shape in filtered_params:
        print(f"  - {name}: {shape}")
    
    # Initialize PairWiseEmpiricalIF
    pairwise_if = PairWiseEmpiricalIF(
        dl_train=train_dataloader,
        model=model,
        param_filter_fn=param_filter_fn,
        criterion=ContrastiveLossWrapper()
    )
    
    print(f"\nPairWiseEmpiricalIF initialized")
    print(f"Training samples loaded: {pairwise_if.num_train_samples}")
    
    # Create a query pair for influence computation
    query_doc = torch.randn(input_dim)  # Single document embedding
    query_code = torch.randn(input_dim)  # Single code embedding
    query_is_positive = True  # This is a positive pair (similar doc-code)
    
    print(f"\nQuery pair created:")
    print(f"  - Query doc shape: {query_doc.shape}")
    print(f"  - Query code shape: {query_code.shape}")
    print(f"  - Is positive pair: {query_is_positive}")
    
    # Compute influence scores
    print(f"\nComputing influence scores...")
    print("This may take a while for the first run...")
    
    try:
        influence_scores = pairwise_if.query_influence(
            input_pair=(query_doc, query_code),
            query_is_positive=query_is_positive,
            lr=1e-3,
            num_negative_samples=8  # Reduced for faster testing
        )
        
        print(f"\nInfluence computation completed!")
        print(f"Total influence scores: {len(influence_scores)}")
        
        # Display top 5 most influential samples
        print(f"\nTop 5 most influential training samples:")
        for i, (doc_idx, code_idx, influence_val, sample_type) in enumerate(influence_scores[:5]):
            print(f"  {i+1}. Doc[{doc_idx}] + Code[{code_idx}] ({sample_type}): {influence_val:.6f}")
            
    except Exception as e:
        print(f"Error during influence computation: {e}")