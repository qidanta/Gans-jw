import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from pycrayon import CrayonClient
from torch.autograd import Variable

import torchvision.utils as vutils
from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torchvision import datasets, transforms
from util.network_util import create_nets, init_network, weight_init
from util.solver_util import create_couple2one_optims, create_optims
from util.train_util import (compute_dloss, compute_gloss,
                             create_netG_indeps_sample,
                             create_netG_share_sample, mutil_backward,
                             compute_mean,
                             mutil_steps, netD_fake, resize_data)
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
display_cnt = 100
iter = 2
nets_num = 10

cuda = False
netD_continue_trian = True


cc = CrayonClient(hostname="localhost")
cc.remove_all_experiments()
D_exp = create_sigle_experiment(cc, 'D_loss')
D_preb_real = create_sigle_experiment(cc, 'preb_real')
D_preb_fake = create_sigle_experiment(cc, 'preb_fake')
G_exps = create_experiments(cc, 10)

netG_indeps = create_nets(config['G'][2], z_dim, nets_num)
netG_share = build_netG(config['G'][3], h_dim)
netD = build_netD(config['D'][2], x_dim)

print netG_indeps
print netG_share

init_network(netG_share)
init_network(netG_indeps)
init_network(netD)

D_solver = optim.Adam(netD.parameters(), lr=lr)
G_share_solver = optim.Adam(netG_share.parameters(), lr=lr)
G_indep_solver = create_optims(netG_indeps, [lr, (0.5, 0.999)])

X = torch.FloatTensor(mb_size, x_dim)
z = torch.FloatTensor(mb_size, z_dim)
label = torch.FloatTensor(mb_size)

if cuda:
    X, z = X.cuda(), z.cuda()
    label = label.cuda()

X = Variable(X)
z = Variable(z)
label = Variable(label)

if netD_continue_trian:
    netD.load_state_dict(torch.load('./result/vanillgan_mnist_latent100/out/netD_epoch_22.pth'))

for it in range(iter):
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        ################
        #  (0) process data
        ################
        netD.zero_grad()
        mb_size = data.size(0)
        X.data.resize_(data.size()).copy_(data)
        X.data.resize_(mb_size, x_dim)
        z.data.resize_(mb_size, z_dim).normal_(0, 1)
        label.data.resize_(mb_size)
        cnt = batch_idx + it * train_size

        G_indep_sample = create_netG_indeps_sample(netG_indeps, z)
        G_share_sample = create_netG_share_sample(netG_share, G_indep_sample)

        ############################
        # (1) Update D network:  vaillgan loss
        ###########################
        if not netD_continue_trian:
            D_real = netD(X)
            D_fake = netD_fake(G_share_sample, netD)
            D_loss = compute_dloss(D_real, D_fake, label)
            D_exp.add_scalar_value('D_loss', D_loss.data[0], step=cnt)
            D_loss.backward(retain_variables = True)
            D_solver.step()
        else:
            D_real = netD(X).data.numpy()[0][0]
            D_fake = netD_fake(G_share_sample, netD)
            D_fake = compute_mean(D_fake, 10.0).data.numpy()[0]
            print 'D_real:{}----/D_fake:{}'.format(D_real, D_fake)

        ############################
        # (2) Update G network: 
        ###########################
        D_fake = netD_fake(G_share_sample, netD)
        G_losses, index = compute_gloss(D_fake, label)
        mutil_backward(G_losses, netG_share, netG_indeps, index)
        mutil_steps(G_losses, G_share_solver, G_indep_solver, index)
        add2experiments(G_losses, G_exps, step=cnt)
        if cnt % display_cnt == 0:
            # display result
            
            z.data.resize_(mb_size, z_dim).normal_(0, 1)
            G_indep_sample = create_netG_indeps_sample(netG_indeps, z)
            G_share_sample = create_netG_share_sample(netG_share, G_indep_sample)
            
            for index, sample in enumerate(G_share_sample):

                prefix = 'iter_{}netG_{}st_'.format(cnt, index)
                sample = sample.data.numpy()
                
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')
                # save displayed-result
                plt.savefig('out/{}.png'.format(prefix + str(cnt)), bbox_inches='tight')

            for index in range(nets_num):
                torch.save(netG_indeps[index].state_dict(), '%s/netG_indep_epoch_%d.pth' % ('./out', cnt))
            torch.save(netG_share.state_dict(), '%s/netG_share_epoch_%d.pth' % ('./out', cnt))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./out', cnt))
