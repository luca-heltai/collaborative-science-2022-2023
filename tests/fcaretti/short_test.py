import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

from utils.models import NeuralNetwork
from utils.training import train

print(torch.__version__)
print(np.__version__)
print('Passed import tests')

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

losses=[]
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    losses.append(train(train_dataloader, model, loss_fn, optimizer,device))
    
if losses[-1]<=losses[0]:
    print('Test passed: training loss is decreasing')
else:
    raise ValueError('Training loss is not decreasing')
