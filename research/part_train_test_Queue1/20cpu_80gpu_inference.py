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

# # CPU and MAIN
# class Net(nn.Module):
#     def __init__(self, shared_queue):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5) #in, out, filtersize
#         self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
#         self.conv2 = nn.Conv2d(100, 60, 5)
#         self.fc1 = nn.Linear(300 * 4 * 4, 1000)
#         self.fc2 = nn.Linear(1000, 10)
#         self.fc3 = nn.Linear(100,10)
#         self.q = shared_queue
    
#     def forward(self, x):
#         x = self.conv1(x)
#         print("cpu에서 기다리는 중")
#         y = self.q.get(True, None).to("cpu")
#         x = torch.cat((x,y), 1)
#         x = F.relu(x)
#         x = self.pool(x)
#         self.q.put(x)
#         x = self.conv2(x)
#         print("cpu 두번째 단계에서 기다리는 중")
#         y = self.q.get(True, None).to("cpu")
#         x = torch.cat((x,y),1)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = x.view(-1, 300 * 4 * 4)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         self.q.put(x)
#         return x

# # CUDA 
# class Net2(nn.Module):
#     def __init__(self, shared_queue):
#         super(Net2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 80, 5) #in, out, filtersize
#         self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
#         self.conv2 = nn.Conv2d(100, 240, 5)
#         self.fc1 = nn.Linear(300 * 4 * 4, 1000)
#         self.fc2 = nn.Linear(1000, 10)
#         self.fc3 = nn.Linear(100,10)
#         self.q = shared_queue
    
#     def forward(self, x):
#         x = self.conv1(x)
#         self.q.put(x)
#         time.sleep(0.005)
#         while(not self.q.empty()):
#             continue
#         x = self.q.get(True,None).to("cuda")
#         x = self.conv2(x)
#         self.q.put(x)
#         x = self.q.get(True, None).to("cuda")
        
#         return x

# CPU 
class Net(nn.Module):
    def __init__(self, shared_queue):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(100, 60, 5)
        self.fc1 = nn.Linear(300 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.fc3 = nn.Linear(100,10)
        self.q = shared_queue
    
    def forward(self, x):
        x = self.conv1(x)
        self.q.put(x)
        print("CPU에서 첫번째 인풋 넣음")
        time.sleep(0.005)
        while(not self.q.empty()):
            continue
        x = self.q.get(True,None).to("cpu")
        x = self.conv2(x)
        self.q.put(x)
        print("CPU에서 두번째 인풋 넣음")
        x = self.q.get(True, None).to("cpu")
        return x

# CUDA AND MAIN
class Net2(nn.Module):
    def __init__(self, shared_queue):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, 5) #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2) #2x2 pooling
        self.conv2 = nn.Conv2d(100, 240, 5)
        self.fc1 = nn.Linear(300 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.fc3 = nn.Linear(100,10)
        self.q = shared_queue
    
    def forward(self, x):
        x = self.conv1(x)
        print("CUDA에서 첫번째 인풋 기다림")
        y = self.q.get(True, None).to("cuda")
        x = torch.cat((x,y), 1)
        x = F.relu(x)
        x = self.pool(x)
        time.sleep(0.005)
        self.q.put(x)
        x = self.conv2(x)
        print("CUDA에서 두번째 인풋 기다림")
        y = self.q.get(True, None).to("cuda")
        x = torch.cat((x,y),1)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 300 * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        self.q.put(x)
        return x



def inference(model, testset, device, q):
    fig = plt.figure(figsize=(10,10))
    plt.title(device, pad=50)
    for i in range(1, columns*rows+1):
        print("-----------------------------")
        data_idx = np.random.randint(len(testset))
        if q.empty():
            print(device, "에서 첫인풋 넣음")
            q.put(data_idx)
        else:
            print(device, "에서 첫인풋 받음")
            data_idx = q.get(True, None)
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
    plt.show() 

def my_run(model, testset, device, pth_path, q):
    model.load_state_dict(torch.load(pth_path), strict=False) 
    model.eval()
    a = time.time()
    inference(model, testset, device, q)
    b = time.time()
    print("time : ", b - a)

def main():
    q = Queue()
    idx_q = Queue()
    epochs = 3
    learning_rate = 0.001
    batch_size = 32
    test_batch_size=16
    log_interval =100
    cpu_pth_path = "/home/yoon/Yoon/pytorch/research/part_train_test_Queue1/cpu.pth"
    gpu_pth_path = "/home/yoon/Yoon/pytorch/research/part_train_test_Queue1/gpu.pth"

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

    
    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # model
    model1 = Net(q).to(device1)
    model1.share_memory()    # imshow example
    model2 = Net2(q).to(device2)
    model2.share_memory()
    
    # Freeze model weights
    for param in model1.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False
    for param in model2.parameters():  # 전체 layer train해도 파라미터 안바뀌게 프리징
        param.requires_grad = False

    proc1 = Process(target=my_run, args=(model1, testset, device1, cpu_pth_path, idx_q))
    proc2 = Process(target=my_run, args=(model2, testset, device2, gpu_pth_path, idx_q))
    
    num_processes = (proc2, proc1) 
    processes = []
    
    for procs in num_processes:
        procs.start()
        processes.append(procs)
        time.sleep(2)

    for proc in processes:
        proc.join()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!
    main()
