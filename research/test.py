from __future__ import print_function
import torch
import numpy as np

#print(torch.cuda.is_available())
x = torch.tensor([[1,2],[3,4]])
if torch.cuda.is_available():
    device = torch.device("cuda") # a CUDA device object
    y = torch.ones_like(x, device=device) # directly create a tensor on GPU

print(x) 
print(y) # y는 GPU 메모리에 올라간다.
x = x.to(device) 
z = x + y   # x와 y의 위치가 다르니깐 x를 gpu로 보내어 계산
print(z)
print(z.to("cpu", torch.double)) # cpu로 옮길 수 있음 (데이터 타입 정의가능)


a = torch.ones(5)
print(a)
b = a.numpy() # numpy변환가능
print(b)
c = x.detach().cpu().numpy()    # x는 gpu에 있으므로 cpu로 detach작업해야함
print(c)

d = np.ones(5)
e = torch.from_numpy(d) # cpu 일때
f = torch.from_numpy(d).float().to(device)
# np.add(d, 1, out=d)
# print(d)
print(e)
print(f)

print("------")
print(x[0])

print([x[0]])
