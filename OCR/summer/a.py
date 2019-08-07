from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms
import torch
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 15, 3)
        self.conv3 = nn.Conv2d(15, 20, 3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=20 * 4 * 4, out_features=248)
        self.fc2 = nn.Linear(in_features=248, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=27)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.eval()
net.load_state_dict(torch.load('OCR/summer/classifier'))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'Ba', 'Bhe', 'Dhau', 'Ga', 'Ja', 'Ka', 'Ko', 'Lu', 'Ma', 'Me', 'Na', 'Ra', 'Sa', 'Se',
           'Cha', 'Jha', 'Yan']
