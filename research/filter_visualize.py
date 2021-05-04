import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import platform
import matplotlib.pyplot as plt
import numpy as np

# build a network model, 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


conv = Net()

# load weights if they haven't been loaded
# skip if you're directly importing a pretrained network
checkpoint = torch.load("/home/yoon/Yoon/pytorch/research/save_model/fashion_mnist.pth")
conv.load_state_dict(checkpoint)


# get the kernels from the first layer
# as per the name of the layer
kernels = conv.conv1.weight.detach().clone()

#check size for sanity check
print(kernels.size())
print(kernels)
# normalize to (0,1) range so that matplotlib
# can plot them
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
filter_img = torchvision.utils.make_grid(kernels, nrow = 12)
# change ordering since matplotlib requires images to 
# be (H, W, C)
plt.imshow(filter_img.permute(1, 2, 0))

# You can directly save the image as well using
