import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pycrayon import CrayonClient
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torchvision import datasets, transforms
import torchvision.utils as vutils
from util.network_util import weight_init
from util.train_util import (link_data, draft_data)
from util.vision_util import (create_sigle_experiment)

# loading datasets
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar10_data/torch_cifar10data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(64),
                       transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=True, num_workers=2)

# global setting
cc = CrayonClient(hostname="localhost")
cc.remove_all_experiments()
mb_size = 64
z_dim = 100
x_dim_w, x_dim_h =train_loader.dataset.train_data.shape[1:3]
print x_dim_w
resize_w, resize_h =64, 64
x_dim = 3
train_size = train_loader.dataset.train_data.shape[0]
lr = 2e-4
cnt = 0

cuda = False
niter = 24

# manual_seed = random.randint(1, 10000)
# print("Random Seed: ", manual_seed)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)

# build gans
netD = build_netD(config['D'][3], 3)
netG = build_netG(config['G'][4], 100)

print netD, netG

# init gans
netD.apply(weight_init)
netG.apply(weight_init)

# build gans's solver
G_solver = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
D_solver = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# build exps of netG and netD
G_exp = create_sigle_experiment(cc, 'G_loss')
D_exp = create_sigle_experiment(cc, 'D_loss')

# announce input
x = torch.FloatTensor(mb_size, x_dim, resize_w, resize_h)
z = torch.FloatTensor(mb_size, z_dim, 1, 1)
label = torch.FloatTensor(mb_size)

# annouce loss-style
criterion = nn.BCELoss()

# init input in cuda, then convert floattensor to variable
if cuda:
    netD.cuda()
    netG.cuda()
    criterion = criterion.cuda()
    label = label.cuda()
    x, z = x.cuda(), z.cuda()

x = Variable(x)
z = Variable(z)
label = Variable(label)

# training
for it in range(niter):
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        mb_size = data.size(0)
        x.data.resize_(data.size()).copy_(data)
        label.data.resize_(mb_size).fill_(1)

        D_real = netD(x)
        errD_real = criterion(D_real, label)
        errD_real.backward()
        z.data.resize_(mb_size, z_dim, 1, 1)
        z.data.normal_(0, 1)
        fake = netG(z)
        D_fake = netD(fake.detach())
        label.data.fill_(0)
        errD_fake = criterion(D_fake, label)
        errD_fake.backward()
        D_loss = errD_real + errD_fake
        D_exp.add_scalar_value('D_loss', D_loss.data[0], step=batch_idx + it * train_size)
        D_solver.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        D_fake = netD(fake)
        label.data.fill_(1)
        G_loss = criterion(D_fake, label)
        G_exp.add_scalar_value('G_loss', G_loss.data[0], step=batch_idx + it * train_size)
        G_loss.backward()
        G_solver.step()
        if batch_idx % 100 == 0:
            print "D_loss:{}-G_loss:{}".format(D_loss.data[0],G_loss.data[0])

    if  it % 2 == 0:
        z.data.resize_(mb_size, z_dim, 1, 1)
        z.data.normal_(0, 1)
        fake = netG(z)
        vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % ('./out', it),
                    normalize=True)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('./out', it))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./out', it))
        cnt += 1

