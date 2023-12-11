import torch
from torchsummary import summary
from my_model import Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model = model.to(device)
input_size = (1, 28, 28)

summary(model, input_size)