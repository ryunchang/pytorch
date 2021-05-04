
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

import time

label_tags = {
    0: 'T-Shirt', 
    1: 'Trouser', 
    2: 'Pullover', 
    3: 'Dress', 
    4: 'Coat', 
    5: 'Sandal', 
    6: 'Shirt',
    7: 'Sneaker', 
    8: 'Bag', 
    9: 'Ankle Boot' }
    
test_batch_size=1000
columns = 6
rows = 6
fig = plt.figure(figsize=(10,10))


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
        x = self.conv1(x)
        print('x = self.conv1(x) 호출완료')

        x = F.relu(x)
        print('x = F.relu(x)')
        
        x = self.pool(x)
        print('x = self.pool(x)')
        
        x = self.conv2(x)
        print('x = self.conv2(x)')
        
        x = F.relu(x)
        print('x = F.relu(x)')

        x = self.pool(x)
        print('x = self.pool(x)')

        x = x.view(-1, 12 * 4 * 4)
        print('x = x.view(-1, 12 * 4 * 4)')

        x = self.fc1(x)
        print('x = self.fc1(x)')

        x = F.relu(x)
        print('x = F.relu(x)')

        x = self.fc2(x)
        print('x = self.fc2(x)')

        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 12 * 4 * 4)
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        print("forward 완료. time sleep 중 입니다 Zzz..")
        #time.sleep(3)
        return x


# build a network model, 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.fc3 = nn.Linear(100,10)
#        self.conv1 = nn.Conv2d(1, 32, 5) #in, out, filtersize
#        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
#        self.conv2 = nn.Conv2d(32, 64, 5)
#        self.fc1 = nn.Linear(64 * 4 * 4, 1000)
#        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        print('@ forward 함수 진입!')
        x = self.conv1(x)
        print('@ x = self.conv1(x) 호출완료')
        x = F.relu(x)
        print('@ x = F.relu(x)')
        x = self.pool(x)
        print('@ x = self.pool(x)')
        x = self.conv2(x)
        print('@ x = self.conv2(x)')
        x = F.relu(x)
        print('@ x = F.relu(x)')
        x = self.pool(x)
        print('@ x = self.pool(x)')
        x = x.view(-1, 12 * 4 * 4)
        print('@ x = x.view(-1, 12 * 4 * 4)')
        x = self.fc1(x)
        print('@ x = self.fc1(x)')
        x = F.relu(x)
        print('@ x = F.relu(x)')
        x = self.fc2(x)
        print('@ x = self.fc2(x)')
        
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 64 * 4 * 4)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
        return x


use_cuda = torch.cuda.is_available()
print("use_cude : ", use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

nThreads = 1 if use_cuda else 2 
if platform.system() == 'Windows':
    nThreads =0 #if you use windows

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                        shuffle=False, num_workers=nThreads)

# model load
print("@이제 모델을 만듭니다.")
model = Net().to(device)
print("@모델을 생성 완료! model = Net().to(device)")
model.load_state_dict(torch.load("/home/yoon/Yoon/pytorch/research/save_model/fashion_mnist.pth"), strict=False) 
print("@load_state_dice 완료!")
model.eval()
print("@model.eval()완료!")

# inference
for i in range(1, columns*rows+1):
    data_idx = np.random.randint(len(testset))
    print("input image 만들고 to(device)")
    input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
    
    print("@ inference할 데이터를 넣겠습니다.")
    output = model(input_img)
    print("@ output = model(input_img) 완료!!")
    _, argmax = torch.max(output, 1)
    pred = label_tags[argmax.item()]
    label = label_tags[testset[data_idx][1]]
    
    fig.add_subplot(rows, columns, i)
    if pred == label:
        plt.title(pred + ', right !!')
        cmap = 'Blues'
    else:
        plt.title('Not ' + pred + ' but ' +  label)
        cmap = 'Reds'
    plot_img = testset[data_idx][0][0,:,:]
    plt.imshow(plot_img, cmap=cmap)
    plt.axis('off')
    
plt.show() 
