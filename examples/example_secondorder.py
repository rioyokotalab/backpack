import torch
import torchvision
# The main BackPACK functionalities
from backpack import backpack, extend
# The diagonal GGN extension
from backpack.extensions import *
# This layer did not exist in Pytorch 1.0:w
from backpack.core.layers import Flatten
from backpack.optim import DiagGGNOptimizer, KronGGNOptimizer

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-3
DAMPING = 1e-5
CURV_EMA_DECAY = 0.95
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
MAX_ITER = 100
#OPTIM = 'DiagGGN'
OPTIM = 'KronGGN'
#GGN_TYPE = 'mc'
GGN_TYPE = 'exact'
#GGN_TYPE = 'recursive'
torch.manual_seed(0)


"""
Step 1: Load data and create the model.

We're going to load the MNIST dataset,
and fit a 3-layer MLP with ReLU activations.
"""


mnist_loader = torch.utils.data.dataloader.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)
            )
        ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(20, 50, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    Flatten(), 
    # Pytorch <1.2 doesn't have a Flatten layer
    torch.nn.Linear(4*4*50, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
)

loss_function = torch.nn.CrossEntropyLoss()


def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


"""
Step 3: Tell BackPACK about the model and loss function, 
create the optimizer, and we will be ready to go
"""

extend(model)
extend(loss_function)

if OPTIM == 'DiagGGN':
    optimizer = DiagGGNOptimizer(
        model.parameters(),
        ggn_type=GGN_TYPE,
        lr=LR,
        damping=DAMPING,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        curv_ema_decay=CURV_EMA_DECAY
    )
elif OPTIM == 'KronGGN':
    optimizer = KronGGNOptimizer(
        model.parameters(),
        ggn_type=GGN_TYPE,
        lr=LR,
        damping=DAMPING,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        curv_ema_decay=CURV_EMA_DECAY
    )
else:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

for batch_idx, (x, y) in enumerate(mnist_loader):

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        return loss, output

    loss, output = optimizer.step(closure=closure)
    accuracy = get_accuracy(output, y)

    print(
        "Iteration %3.d/%d   " % (batch_idx, MAX_ITER) +
        "Minibatch Loss %.3f  " % (loss) +
        "Accuracy %.0f" % (accuracy * 100) + "%"
    )

    if batch_idx >= MAX_ITER:
        break
