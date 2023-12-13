import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
import time

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 


        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

   
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)  
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x) 
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '10000'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, epochs=10):
    setup(rank, world_size)


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, sampler=train_sampler)


    device = torch.device("cuda", rank)
    model = ComplexCNN().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    start_time = time.time()  
    model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    end_time = time.time()  
    total_time = end_time - start_time
    print(f"Total training time ddp: {total_time:.2f} seconds")
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print("availale gpus",world_size)
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
