import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
lr = 1e-3
dropout = 0.25


# Model
class CNN3(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# Instantiate the model
model = CNN3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


