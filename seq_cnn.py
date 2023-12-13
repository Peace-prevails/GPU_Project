import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.cuda.profiler as profiler

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


device = torch.device("cpu")
model = ComplexCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"Current device: {device}")

def train(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    start_time = time.time() 

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    end_time = time.time()  
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")


# Profile the model
profiler.start()
train(model, device, train_loader, optimizer, criterion)
profiler.stop()
