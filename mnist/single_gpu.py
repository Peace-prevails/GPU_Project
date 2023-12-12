import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from datautils import MyTrainDataset
from my_model import Net

import torch.cuda.profiler as profiler


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        # profiler.start()
        source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
        self.optimizer.zero_grad()
        output = self.model(source)
        # profiler.stop()
        loss = F.nll_loss(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.model.train()
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "epoch_%d_checkpoint.pt" % epoch
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        # scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        for epoch in range(1, max_epochs + 1):
            self._run_epoch(epoch)
            # scheduler.step()
            # if epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)
        # self._save_checkpoint(epoch)


def load_train_objs(dataset_size=2048, batch_size=32, use_cuda=True, use_mps=False, lr=1.0):
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
    # print(train_kwargs)

    # offical dataset
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)

    my_dataset = MyTrainDataset(dataset_size)

    train_loader = DataLoader(my_dataset, **train_kwargs)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    return train_loader, model, optimizer


def main(device, total_epochs, save_every, dataset_size, batch_size):
    train_data, model, optimizer = load_train_objs(dataset_size=dataset_size, batch_size=batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    
    start_train_time = time.time()
    trainer.train(total_epochs)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    print("total_epochs:%d, train_time:%d" % (total_epochs, train_time))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--dataset_size', default=2048, type=int, help='Fake dataset size (default: 2048)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.dataset_size, args.batch_size)
