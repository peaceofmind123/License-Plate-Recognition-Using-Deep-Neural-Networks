import torch
import torch.nn as nn
from torch.nn import functional as F
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

    @staticmethod
    def build_network():
        return Net()

    def load_weights(self, weight_path=os.path.join(os.getcwd(),'OCR', 'weights/classifier')):
        weights = torch.load(weight_path)
        self.load_state_dict(weights)
