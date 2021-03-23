# pytorch basic classification 
# Fashion mnist data set

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import platform
import matplotlib.pyplot as plt
import numpy as np

import threading
import time


def imshow(img):
     npimg = img.numpy() #convert the tensor to numpy for displaying the image
     #for displaying the image, shape of the image should be height * width * channels 
     plt.imshow(np.transpose(npimg, (1, 2, 0))) 
     plt.show()

class Worker(threading.Thread):
    def __init__(self, name, args):
        super().__init__()
        self.name = name
        self._args = args

    def run(self):
        start_time = time.time()
        for epoch in range(1, self._args[5] + 1):
            print(self._args[2], "에서", epoch, "회 에폭 실행")
            train(self._args[0], self._args[1], self._args[2], self._args[3], self._args[4], epoch)
            test(self._args[1], self._args[2], self._args[3])
        stop_time = time.time()
        print("duration : ", stop_time - start_time)

# build a network model, 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss =0.0
    criterion = nn.CrossEntropyLoss() #defalut is mean of mini-batchsamples
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(device, ' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss/log_interval))
            running_loss =0.0



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion =  nn.CrossEntropyLoss(reduction='sum') #add all samples in a mini-batch
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss +=  loss.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    epochs = 5
    learning_rate = 0.001
    batch_size = 32
    test_batch_size=1000
    log_interval =100
    
    #print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    print("use_cude : ", use_cuda)
    
    #device = torch.device("cuda" if use_cuda else "cpu")
    device1 = "cpu"
    device2 = "cuda"

    nThreads = 1 if use_cuda else 2 
    if platform.system() == 'Windows':
        nThreads =0 #if you use windows
  
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=True,
        transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)

 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=nThreads)


    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                            shuffle=False, num_workers=nThreads)

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    # model
    model1 = Net().to(device1)
    model2 = Net().to(device2)
    
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer1 = optim.Adam(model1.parameters(),lr=learning_rate)
    optimizer2 = optim.Adam(model2.parameters(),lr=learning_rate)

    device_args1 = (log_interval, model1, device1, train_loader, optimizer1, epochs)
    device_args2 = (log_interval, model2, device2, train_loader, optimizer2, epochs)

    task1 = Worker(device1+" task", device_args1)
    task2 = Worker(device2+" task", device_args2)

    task1.start()
    task2.start()

if __name__ == '__main__':
    main()
