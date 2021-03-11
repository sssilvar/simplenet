import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch, log_interval=10, dry_run=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


# ===============================================================================================
# Training routine
# ===============================================================================================

import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def training_routine(model, batch_size=48, lr=1e-3, epochs=2, gamma=0.7):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('~/.data/', train=True, download=True,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()

    print(model)

