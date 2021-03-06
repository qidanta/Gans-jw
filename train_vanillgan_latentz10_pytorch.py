import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from pycrayon import CrayonClient
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torchvision import datasets, transforms
from util.network_util import weight_init
from util.train_util import (link_data)
from util.vision_util import (create_sigle_experiment)

# loading datasets
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=True, **{})

# global setting
cc = CrayonClient(hostname="localhost")
cc.remove_all_experiments()
mb_size = 64
z_dim = 10
h_dim = 128
x_dim_w, x_dim_h =train_loader.dataset.train_data.size()[1:3]
resize_w, resize_h =64, 64
x_dim = x_dim_w * x_dim_h
train_size = train_loader.dataset.train_data.size()[0]
y_dim = 10
lr = 1e-3
cnt = 0

cuda = False
niter = 24

# build gans
netD = build_netD(config['D'][2], x_dim)
netG = build_netG(config['G'][1], z_dim)


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
x = torch.FloatTensor(mb_size, x_dim)
z = torch.FloatTensor(mb_size, z_dim)

# init input in cuda, then convert floattensor to variable
if cuda:
    netD.cuda()
    netG.cuda()
    x, z = x.cuda(), z.cuda()

x = Variable(x)
z = Variable(z)

# training
for it in range(niter):
    for batch_idx, (data, target) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        x.data.resize_(data.size()).copy_(data)
        x.data.resize_(mb_size, x_dim)
        z.data.resize_(mb_size, z_dim).normal_(0, 1)

        D_real = netD(x)
        fake = netG(z)      
        D_fake = netD(fake)

        D_loss = -(torch.mean(torch.log(D_real)) + torch.mean(torch.log(1 - D_fake)))
        D_loss.backward(retain_variables = True)
        D_exp.add_scalar_value('D_loss', D_loss.data[0], step=batch_idx + it * train_size)
        D_solver.step()

        ############################
        # (2) Update G network: maximize log(1 - D(G(z)))
        ###########################
        netG.zero_grad()
        D_fake = netD(fake)
        G_loss = -torch.mean(torch.log(D_fake))
        G_exp.add_scalar_value('G_loss', G_loss.data[0], step=batch_idx + it * train_size)
        G_loss.backward(retain_variables = True)
        G_solver.step()

    if  it % 2 == 0:
        z.data.resize_(mb_size, z_dim).normal_(0, 1)
        samples = netG(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for index, sample in  enumerate(samples):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')
        
        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('./out', it))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./out', it))
        cnt += 1
        plt.close(fig)


