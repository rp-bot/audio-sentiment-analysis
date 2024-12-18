import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from pre_process import RavdessDataset
from dataset_utils import RAVDESS_MUSIC_PROCESSED


# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
LEARNING_RATE = 1e-3
DROPOUT = 0.25
BATCH_SIZE = 16


# DataLoader
ravdess_dataset = RavdessDataset(RAVDESS_MUSIC_PROCESSED)
data_loader = DataLoader(ravdess_dataset, BATCH_SIZE)


# Model
class CNN3(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# Instantiate the model
model = CNN3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


# Train Loop

# for batch in data_loader:
#     pass
