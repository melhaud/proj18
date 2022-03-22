from IPython import display
import matplotib.pyplot as plt
import numpy as np
from lightly.loss import BarlowTwinsLoss
from lightly.data import LightlyDataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from torch import cuda
import torch
from torch import nn
from torch.optim import SGD
from src.barlow_twins import model
from src.utils import custom_collate_fn, get_classes
device = "cuda" if cuda.is_available() else "cpu"
model.to(device)

cifar10_train = CIFAR10("datasets/cifar10", download=True, train=True)
cifar10_test = CIFAR10("datasets/cifar10", download=True, train=False)
classes_ids_train = get_classes(cifar10_train) # long!
classes_ids_test = get_classes(cifar10_test)
dataset = LightlyDataset.from_torch_dataset(Subset(cifar10_train, classes_ids_train['dog']))
# dataset = LightlyDataset.from_torch_dataset(cifar10)

collate_fn = custom_collate_fn()

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

criterion = BarlowTwinsLoss()
optimizer = SGD(model.parameters(), momentum=0.9, lr=0.06)
# model.load_state_dict(torch.load('file'))

# plotter
def interactive_plot(x_range, avg_loss):
    fig, ax = plt.subplots(figsize=(12, 6))

    # color = 'tab:red'
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg loss', color=color)
    ax.plot(x_range, avg_loss, 'r',  ls = '--') #label = 'val_loss',
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid()
    plt.show()

print("Starting Training")
epochs = range(400)
avg_losses = []
for epoch in epochs:
    total_loss = 0

    for (x0, x1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    avg_losses.append(avg_loss.cpu().detach())
    # save the model every 20 epochs
    if epoch // 20 == 0:
        torch.save(model.state_dict(), f'weights_{epoch}_{avg_loss:.3f}')
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    display.clear_output(wait=True)
    interactive_plot(np.arange(epoch+1), avg_losses)
