import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        # Create random data: Tensors of shape [1, 28, 28] and scalar labels from 0 to 9
        self.data = [(torch.rand(1, 28, 28), torch.randint(0, 10, (1,)).item()) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

