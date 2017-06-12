import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pycrayon import CrayonClient
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torchvision import datasets, transforms
from util.network_util import create_nets, init_network, weight_init
from util.solver_util import create_couple2one_optims, create_optims
from util.train_util import (compute_dloss, compute_gloss, find_best_netG
                             create_netG_indeps_sample, mutil_steps, netD_fake)
from util.vision_util import (add2experiments, create_experiments,
                              create_sigle_experiment)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **{})
mb_size = 1
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
entropy = nn.BCEloss()

if cuda:
    X, z = X.cuda(), z.cuda()
    label = label.cuda()
    entropy = entropy.cuda()

X = Variable(X)
z = Variable(z)
label = Variable(label)

for it in range(2):
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
        # real-loss
        label.data.fill_(1)
        real_loss = entropy(D_real, label)
        real_loss.backward()
        # real-like-loss
        best_fake_porp_index = find_best_netG(D_fake)
        real_like_prop = D_fake[best_fake_porp_index]
        label.data.resize_(1)
        real_like_loss = entropy(real_like_prop, label)
        real_like_loss.backward()
        # reset-fake-loss
        netG_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        label.data.resize_(mb_size).fill_(0)
        fake_losses = []
        for index in range(len(D_fake)):
            fake_loss = entropy(D_fake[index], label)
            fake_losses.append(fake_loss)
            if index != best_fake_porp_index:
                fake_loss.backward()
        reset_fake_loss = (sum(fake_losses[i] for i in netG_num) - fake_losses[best_netG_index]) / (len(netG_num) - 1)
        D_loss = real_loss + real_like_loss + reset_fake_loss
        D_exp.add_scalar_value('D_loss', D_loss.data[0], step=batch_idx + it * train_size)
        D_solver.step()

        ############################
        # (2) Update G network: 
        ###########################
        D_fake = netD_fake(G_indep_sample, netD)
        G_losses, index = compute_gloss(D_fake, label)
        mutil_steps(G_losses, G_share_solver, G_indep_solver, best_fake_porp_index)
        add2experiments(G_losses, G_exps, step=batch_idx + it * train_size)

    
    if it % 2 == 0:
        z.data.resize_(mb_size, z_dim).normal_(0, 1)
        G_share_sample = netG_share(z)
        G_indep_sample = create_netG_indeps_sample(netG_indeps, G_share_sample)
        for index_of_sampels, samples in enumerate(G_indep_sample):
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            prefix = 'iter_{}netG_{}st_'.format(cnt, index_of_sampels)
            samples = samples.data.numpy()[:16]
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
        for index_of_sampels in range(len(G_indep_sample)):
            torch.save(netG_indeps[index_of_sampels].state_dict(), '%s/netG_indep_epoch_%d.pth' % ('./out', it))
        torch.save(netG_share.state_dict(), '%s/netG_share_epoch_%d.pth' % ('./out', it))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./out', it))
        cnt += 1
