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

import time
import socket
import numpy
from torch.multiprocessing import Process, Queue

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

# Socket
RCV_HOST = '127.0.0.1'
RCV_PORT = 9999

SEND_HOST = '127.0.0.1'
SEND_PORT = 9998

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#send_socket.bind((SEND_HOST, SEND_PORT))

receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_socket.bind((RCV_HOST, RCV_PORT))

# CPU 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(100, 60, 5)
        self.fc1 = nn.Linear(300 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.fc3 = nn.Linear(100,10)

    def forward(self, x):
        rcv, addr = receive_socket.recvfrom(3136)
        rcv = np.frombuffer(rcv, dtype=np.float32)
        rcv = np.reshape(rcv, (1,1,28,28))
        x = torch.from_numpy(rcv)
        x = self.conv1(x)
        print(x.shape)
        snd = x.to("cpu").numpy().tobytes()
        send_socket.sendto(snd, (SEND_HOST, SEND_PORT))
        rcv, addr = receive_socket.recvfrom(57600)
        rcv = np.frombuffer(rcv, dtype=np.float32)
        rcv = np.reshape(rcv, (1,100,12,12))
        x = torch.from_numpy(rcv)
        x = self.conv2(x)
        snd = x.to("cpu").numpy().tobytes()
        send_socket.sendto(snd, (SEND_HOST, SEND_PORT))
        rcv, addr = receive_socket.recvfrom(102400)
        rcv = np.frombuffer(rcv, dtype=np.float32)
        rcv = np.reshape(rcv, (1,10))
        x = torch.from_numpy(rcv)
        return x


def inference(model, testset, device):
    fig = plt.figure(figsize=(10,10))
    plt.title(device, pad=50)
    for i in range(1, columns*rows+1):
        print("-----------------------------")
        data_idx = np.random.randint(len(testset))
        input_img = testset[data_idx][0].unsqueeze(dim=0).to(device) 
        output = model(input_img)
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
        print("-----------------------------")
    plt.show()      # If you want to measure inferencing time, comment out this line


def main():
    epochs = 3
    learning_rate = 0.001
    batch_size = 32
    test_batch_size=16
    log_interval =100
    cpu_pth_path = "/home/yoon/Yoon/pytorch/research/pth/cpu_20:80.pth"
    gpu_pth_path = "/home/yoon/Yoon/pytorch/research/pth/gpu_20:80.pth"

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
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                            shuffle=False, num_workers=nThreads)


    # model
    model = Net().to(device1)
    #model.share_memory()    # imshow example

    model.load_state_dict(torch.load(cpu_pth_path), strict=False) 
    model.eval()

    # Freeze model weights
    for param in model.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False

    a = time.time()
    inference(model, testset, device1)
    b = time.time()
    print("time : ", b - a)


if __name__ == '__main__':
    main()
