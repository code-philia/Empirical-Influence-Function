# Influence-Function
Pytorch Implementation of Famous Influence Function Methods

## Requirements
- Step 1: Install torch, torchvision compatible with your CUDA, see here: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- Step 2: 
```
pip install -r requirements.txt
```

## Instructions

### Use Empirical IF
```python
from src.IF import EmpiricalIF
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

# Load the test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

resnet18 = models.resnet18(pretrained=True)
num_classes = 10  # CIFAR-10 has 10 classes
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)  # Replace the last layer

IF = EmpiricalIF(dl_train=trainloader,
                           model=resnet18,
                           param_filter_fn=lambda name, param: 'fc' in name,
                           criterion=nn.CrossEntropyLoss(reduction="none"))


for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)
```


### Use Original Influence Function
```python
from src.IF import BaseInfluenceFunction
# ...

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

# ...

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
# ...
# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

# ...

IF = TracIn(dl_train=trainloader,
                           model=resnet18,
                           param_filter_fn=lambda name, param: 'fc' in name,
                           criterion=nn.CrossEntropyLoss(reduction="none"))


for test_sample in testloader:
    test_input, test_target = test_sample
    IF_scores = IF.query_influence(test_input, test_target)
    print(IF_scores)
```