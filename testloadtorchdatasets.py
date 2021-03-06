from torchvision import datasets, transforms
import torch

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.Scale(64),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **{})

cifar10_train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./cifar10_data/torch_cifar10data', download=True,
                           transform=transforms.Compose([
                               transforms.Scale(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])),
    batch_size=64, shuffle=True, **{})

mb_size=64
print train_loader.dataset.train_data.size()[1]*train_loader.dataset.train_data.size()[2]
print train_loader.dataset.train_labels.size()

for batch_idx, (data, target) in enumerate(train_loader):
    print data.size()
    #print data.resize_(mb_size, 28*28)
    # print data.view(mb_size, 28*28)
    # if batch_idx == 64:
    #     break