import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn
import torchvision.utils as vutils
import torch.optim as optim
from pycrayon import CrayonClient
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torchvision import datasets, transforms
from util.network_util import create_nets, init_network, weight_init
from util.solver_util import create_couple2one_optims, create_optims
from util.train_util import (compute_dloss, compute_gloss,
                             create_netG_indeps_sample, mutil_steps, netD_fake, resize_data)
from util.vision_util import (add2experiments, create_experiments,
                              create_sigle_experiment)
mb_size = 1
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=mb_size, shuffle=True, **{})

z_dim = 100
h_dim = 128
x_dim_w, x_dim_h =train_loader.dataset.train_data.size()[1:3] 
x_dim = x_dim_w*x_dim_h
train_size = train_loader.dataset.train_data.size()[0]
y_dim = 10
lr = 1e-3
cnt = 0
nets_num = 10

cuda = False


cc = CrayonClient(hostname="localhost")
cc.remove_all_experiments()
D_exp = create_sigle_experiment(cc, 'D_loss')
G_exps = create_experiments(cc, 10)

netG_share = build_netG(config['G'][2], z_dim)
netG_indeps = create_nets(config['G'], h_dim, nets_num)
netD = build_netD(config['D'][2], x_dim)

init_network(netG_share)
init_network(netG_indeps)
init_network(netD)

D_solver = optim.Adam(netD.parameters(), lr=lr)
G_share_solver = optim.Adam(netG_share.parameters(), lr=lr)
G_indep_solver = create_optims(netG_indeps, [lr,])
G_solvers = create_couple2one_optims(netG_share, netG_indeps, [lr,])

X = torch.FloatTensor(mb_size, x_dim)
z = torch.FloatTensor(mb_size, z_dim)
label = torch.FloatTensor(mb_size)

if cuda:
    X, z = X.cuda(), z.cuda()
    label = label.cuda()

X = Variable(X)
z = Variable(z)
label = Variable(label)

for it in range(10):
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        netD.zero_grad()
        mb_size = data.size(0)
        X.data.resize_(data.size()).copy_(data)
        X.data.resize_(mb_size, x_dim)
        z.data.resize_(mb_size, z_dim).normal_(0, 1)
        label.data.resize_(mb_size)

        G_share_sample = netG_share(z)
        G_indep_sample = create_netG_indeps_sample(netG_indeps, G_share_sample)

        ############################
        # (1) Update D network: 
        ###########################
        D_real = netD(X)
        D_fake = netD_fake(G_indep_sample, netD)
        D_loss = compute_dloss(D_real, D_fake, label)
        D_exp.add_scalar_value('D_loss', D_loss.data[0], step=batch_idx + it * train_size)
        D_loss.backward(retain_variables = True)
        D_solver.step()

        ############################
        # (2) Update G network: 
        ###########################
        D_fake = netD_fake(G_indep_sample, netD)
        G_losses, index = compute_gloss(D_fake, label)
        mutil_backward(G_losses, netG_share, netG_indeps, index)
        mutil_steps(G_losses, G_share_solver, G_indep_solver, index)
        add2experiments(G_losses, G_exps, step=batch_idx + it * train_size)

    if it % 2 == 0:
        z.data.resize_(mb_size, z_dim).normal_(0, 1)
        G_share_sample = netG_share(z)
        G_indep_sample = create_netG_indeps_sample(netG_indeps, G_share_sample)
        #resize_data(G_indep_sample, x_dim_w, mb_size)
        for index_of_sampels, samples in enumerate(G_indep_sample):
            fig = plt.figure(figsize=(1, 1))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=0.05, hspace=0.05)

            prefix = 'iter_{}netG_{}st_'.format(cnt, index_of_sampels)
            samples = samples.data.numpy()
            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')
            
            plt.savefig('out/{}.png'.format(prefix + str(cnt)), bbox_inches='tight')
            plt.close(fig)
        for index in range(len(G_indep_sample)):
            torch.save(netG_indeps[index].state_dict(), '%s/netG_indep_epoch_%d.pth' % ('./out', it))
        torch.save(netG_share.state_dict(), '%s/netG_share_epoch_%d.pth' % ('./out', it))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./out', it))
        cnt += 1
