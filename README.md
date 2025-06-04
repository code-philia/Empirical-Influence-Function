# Influence-Function
PyTorch Implementation of Famous Influence Function Methods.

## 🧠 What is Influence Function? 

The influence function (Understanding Black-box Predictions via Influence Functions, ICML 2017.) tells you:

> **How much does a single training point affect the model's prediction or loss on a specific test point?**

Imagine training your model, then removing or perturbing one training sample. The influence function estimates:

> **What would happen to the model’s prediction (or loss) on a test point if I changed that training sample?**

But it does this without re-training — using gradient and Hessian approximations instead.

## 🧪 How Influence Function (Understanding Black-box Predictions via Influence Functions, ICML 2017) Works?

$$
\text{Influence}(z_i, z_{\text{test}}) = - \nabla_\theta \mathcal{L}(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_i, \hat{\theta})
$$

Where $z_i$ is a training sample, $z_{\text{test}}$ is the test sample, $\hat{\theta}$ are the trained model parameters, $\mathcal{L}$ is the loss function.
$H_{\hat{\theta}}$ is the Hessian of the **total** training loss at $\hat{\theta}$, i.e. $H_{\hat{\theta}} = \frac{1}{n} \sum_{i=1}^{n} \nabla^2_\theta \mathcal{L}(z_i, \theta) \bigg|_{\theta = \hat{\theta}}$

- **Positive influence scores** → test loss goes up if we keep this training point → it's **harmful**
- **Negative influence scores** → test loss goes down if we keep this point → it's **helping**
  
## ❗ Limitations of the Original Influence Function

One limitation of the original Influence Function design is its computation bottleneck in the estimation of inverse Hessian. 
Even approximating with conjugate gradient or damping can be expensive and unstable.
In this repository, we also include two lightweight variants of Influence Function, which greatly speed up the computation.

| Paper | Tool Name | Venue |
| ---|---| ---|
| Estimating Training Data Influence by Tracing Gradient Descent | TracIn | NeurIPs 2020 |
| Debugging and Explaining Metric Learning Approach: An Influence Function Perspective | EmpiricalIF | NeurIPs 2022 |

### Intuitions of TracIn

$$
\text{Influence}(z_i, z_{\text{test}}) = \frac{1}{T} \sum_{t=1}^T \nabla_\theta \mathcal{L}(z_{\text{test}}, \hat{\theta})^\top \nabla_\theta \mathcal{L}(z_i, \hat{\theta})
$$

If we have collected $T$ checkpoints during the training process, TracIn estimates the influence of a training point on a test point by computing the **gradient alignment** (dot product) at each checkpoint:

- A **positive value** suggests that $z_i$ helped the model reduce test loss.
- A **negative value** suggests that $z_i$ hurt test performance (possibly harmful data).

TracIn is fundamentally a _**first-order**_ influence approximation, unlike the original Influence Function which involves _**second-order**_ Hessian terms.
One limitation of TracIn is that it requires recording intermediate checkpoints in order to have a good estimation.

### Intuitions of EmpiricalIF

$$
\text{Influence}(z_i, z_{\text{test}}) = \mathbb{E_{\delta}} \left[ \nabla_\theta \mathcal{L}(z_{\text{test}}, \hat{\theta} + \delta)^\top \nabla_\theta \mathcal{L}(z_i, \hat{\theta} + \delta) \right]
$$

Based on the final checkpoint, EmpiricalIF estimates the **gradient alignment** by perturbing $\hat{\theta}$ with $\delta$, where $\delta \sim \\{x \in \mathbb{R}^d \mid \left\lVert x \right\rVert = r\\}$:
- A **positive value** suggests that $z_i$ co-evolve with test $z_{\text{test}}$, i.e. helpful.
- A **negative value** suggests that $z_i$ conflict with test $z_{\text{test}}$, i.e. harmful.

EmpiricalIF is a _**single-checkpoint**_ relaxation of TracIn.
In practice, we find setting $\delta$ to be the steepest decent direction of testing and the steepest ascent direction of testing are sufficient for computing EmpiricalIF.

## 🛠️ Requirements
- Step 1: Install torch, torchvision compatible with your CUDA, see here: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- Step 2: 
```
pip install -r requirements.txt
```

## 💻 Instructions

Input Arguments:
- ``dl_train``: Training dataloader of type torch.utils.data.DataLoader
- ``model``: Model class of ``nn.Module`` type
- ``param_filter_fn``: Module names that participate in influence calculation (e.g. last linear layer)
- ``criterion``: Loss function

Outputs:
- Running ``IF.query_influence(test_input, test_target)`` will return a list of influence scores of size (|dl_train|,),  indicating how much each training sample contributes to this testing sample.
  
### Use Empirical IF
```python
from src.IF import EmpiricalIF

IF = EmpiricalIF(dl_train=trainloader,
                 model=resnet18,
                 param_filter_fn=lambda name, param: 'fc' in name,
                 criterion=nn.CrossEntropyLoss(reduction="none"))

for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores) # of size (|dl_train|,)
```

You can further validate the influence results by perturbing selected training samples (with highest and lowest influence values, due to time constraints) and observing how they affect the test loss. This serves as a reverse check of influence correctness.
```python
most_inf, least_inf = IF.reverse_check(
    query_input=test_input,
    query_target=test_target,
    influence_values=IF_scores,
    check_ratio=0.01  # top and bottom 1%
) # [(idx, influence_value, reverse_influence_value), ...], [(idx, influence_value, reverse_influence_value), ...]

for idx, orig_if, rev_if in most_inf:
    print(f"Top IF sample {idx}: IF={orig_if:.4f}, Reverse IF={rev_if:.4f}")

for idx, orig_if, rev_if in least_inf:
    print(f"Bottom IF sample {idx}: IF={orig_if:.4f}, Reverse IF={rev_if:.4f}")
```



### Use Original Influence Function
```python
from src.IF import BaseInfluenceFunction

IF = BaseInfluenceFunction(dl_train=trainloader,
                           model=resnet18,
                           param_filter_fn=lambda name, param: 'fc' in name,
                           criterion=nn.CrossEntropyLoss(reduction="none"))

for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)
```



### Use TracIn
```python
from src.IF import TracIn

IF = TracIn(dl_train=trainloader,
                           model=resnet18,
                           param_filter_fn=lambda name, param: 'fc' in name,
                           criterion=nn.CrossEntropyLoss(reduction="none"))


for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)
```

## If you Find our Repo Useful, Please Consider Cite our Paper 

```bibex
@article{liu2022debugging,
  title={Debugging and Explaining Metric Learning Approaches: An Influence Function Based Perspective},
  author={Liu, Ruofan and Lin, Yun and Yang, Xianglin and Dong, Jin Song},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={7824--7837},
  year={2022}
}
```
