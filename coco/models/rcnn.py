import torch
import torch.nn as nn

class rcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(512, 512, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(512, 15, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(512, 60, kernel_size=1, stride=1)

    def forward(self, x):
        relu = nn.ReLU()
        x = relu(self.conv1(x))
        clf = torch.sigmoid(self.conv2(x))
        reg = relu(self.conv3(x))
        return clf, reg