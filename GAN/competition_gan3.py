import argparse
import os
import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import glog as log
from torch.autograd import Variable

import torchvision.utils as vutils
from baseModel import _baseModel
from D.netD_fm import build_netDFM
from G.netG_fm import build_netGFM
from util.network_util import create_nets, init_network
from util.solver_util import create_optims
from util.train_util import (find_best_netG, find_best_netG_v1dot1,
                             find_best_netG_v1dot2, link2condition_data, compute_fm_loss)
from util.vision_util import create_sigle_experiment


#v3 keywords: fm_loss
class _competitionGan(_baseModel):
    '''feature match gans, compare G's each layer out and D's each layer out
    ...this is loss that how to compute

    @Params:
    - opt: from `train_`.py  files
    - x_dim: noise dim
    - z_dim: img's channels(dim)
    - condtition_D: for ssl
    - mb_size: the batch size of training
    - Lambda: the weight of some loss
    - savepath: where to save the path
    '''
    def __init__(self, opt):
        super(_competitionGan, self).__init__(opt)
        self.opt = opt
        self.x_dim = opt.x_dim
        self.z_dim = opt.z_dim
        self.condition_D = opt.condition_D
        self.mb_size = opt.mb_size
        self.Lambda = opt.Lambda
        self.continue_train = opt.continue_train
        self.train = opt.train
        self.test = True if self.continue_train and self.train else False
        self.savepath = '{}{}/'.format(opt.savepath, opt.gans_type)
        self.cnt = 0

        self.netG = build_netGFM(opt.g_model, opt.z_dim, gans_type=opt.gans_type)
        self.netD = build_netDFM(opt.d_model, opt.x_dim, opt.condition_D, gans_type=opt.gans_type)


        X = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        Z = torch.FloatTensor(opt.mb_size, opt.z_dim, 1, 1)

        real_like_sample = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        fake_like_sample = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        
        label = torch.FloatTensor(opt.mb_size)
        self.criterionGAN = torch.nn.BCELoss()
        self.criterionL1 = torch.nn.L1Loss()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.criterionGAN.cuda()
            self.criterionL1.cuda()
            X, Z = X.cuda(), Z.cuda()
            real_like_sample, fake_like_sample = real_like_sample.cuda(), fake_like_sample.cuda()
            label = label.cuda()

        self.X = Variable(X)
        self.Z = Variable(Z)
        self.real_like_sample = Variable(real_like_sample)
        self.fake_like_sample = Variable(fake_like_sample)
        self.label= Variable(label)

        info.log("Train: {}  Continue: {}  Test: {}".format(self.train, self.continue_train, self.test))

        if self.opt.cc:
            self.create_tensorboard()
            self.index_exp = create_sigle_experiment(self.cc, 'index')
        self.D_solver = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.G_solver = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

        if opt.train == False:
            self.load_networkG(self.opt.g_network_path)
            self.load_networkD(self.opt.d_network_path)
        else:
            init_network(self.netD)
            init_network(self.netG)
    
    def draft_data(self, input, target):
        '''process input-data'''
        self.mb_size = input.size(0)
        self.target = target
        self.X.data.resize_(input.size()).copy_(input)
        self.Z.data.resize_(self.mb_size, self.z_dim, 1, 1).normal_(0, 1)
        self.label.data.resize_(self.mb_size)

    def backward_D(self):
        '''net D backward'''
        self.fake, self.fake_fm = self.netG(self.Z)
        self.D_fake, _ = self.netD(self.fake)

        self.D_real, self.real_fm  = self.netD(self.X)
        self.real_fm.reverse()

        # real-backward
        self.label.data.fill_(1)
        self.loss_D_real = self.criterionGAN(self.D_real, self.label)
        self.loss_D_real.backward(retain_variables=True)

        # fake-backward
        self.label.data.fill_(0)
        self.loss_D_fake = self.criterionGAN(self.D_fake, self.label)
        self.loss_D_fake.backward(retain_variables=True)

        # lambda
        self.loss_D = self.loss_D_real + self.loss_D_fake
        #self.loss_D.backward(retain_variables=True)
        self.cnt = self.cnt + 1

    def backward_G(self):  
        '''net G backward'''
        D_fake, _ = self.netD(self.fake)

        self.label.data.fill_(1)
        self.loss_G_real = self.criterionGAN(D_fake, self.label)
        self.fake_like_sample = self.fake
        #self.loss_G_real.backward(retain_variables=True)

        self.fm_loss = compute_fm_loss(self.real_fm, self.fake_fm, cuda=self.opt.cuda)
        #self.fm_loss.backward(retain_variables=True)
        self.loss_G = self.loss_G_real * 0.1 + self.fm_loss * 0.9

        self.loss_G.backward(retain_variables=True)
        self.best_netG_index = 0
    
    def train(self, input, target):
        '''train model gan, by backward G/D'''
        self.draft_data(input, target)

        self.netD.zero_grad()
        self.backward_D()
        self.D_solver.step()

        self.netG.zero_grad()
        self.backward_G()
        self.G_solver.step()
        self.visual()
    
    def test(self, cnt):
        self.Z.data.resize_(self.opt.mb_size, self.z_dim).normal_(0, 1) 
        fake = self.netG.forward(self.Z)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.save_image(fake, cnt, self.savepath)
    
    def save_network(self, it, savepath):
        log.info("Saving netG - [epochs: {}  cnt: {}  index: {}] in {}".format(it, self.cnt, self.best_netG_index, self.savepath))
        torch.save(self.netG.state_dict(), '{}/netG_epoch{}_index{}.pth' .format(savepath, it, self.best_netG_index))
        log.info("Saving netD - [epochs: {}  cnt: {}] in {}".format(it, self.cnt, self.savepath))
        torch.save(self.netD.state_dict(), '{}/netD_epoch{}.pth' .format(savepath, it))

    def save_image(self, fake, it , savepath):
        '''save result of netG output

        @Params:
        - fake: the output of netG
        - it: number of iterations
        - savepath: in savepath, save network parameter
        '''
        if self.opt.cuda:
            samples = fake.data.cpu()
            samples = samples.resize_(self.mb_size, self.opt.x_dim, self.opt.img_size, self.opt.img_size).numpy()[:16]
        else:
            samples = fake.data.resize_(self.mb_size, self.opt.x_dim, self.opt.img_size, self.opt.img_size).numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for index, sample in  enumerate(samples):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.opt.img_size, self.opt.img_size), cmap='Greys_r')
        
        if self.train:
            log.info("Saving TRIMG - [epochs: {}  cnt: {}  index: {}] in {}".format(it, self.cnt, self.best_netG_index, self.savepath))
            plt.savefig(self.savepath+ '/TRIMG_epoch{}_index{}.png'.format(str(it), self.best_netG_index), bbox_inches='tight')
        else:
            log.info("Saving TESTIMG - [epochs: {}  cnt: {}] in {}".format(it, self.cnt, self.savepath))
            plt.savefig(self.savepath+ '/TESTIMG_epoch{}.png'.format(str(it)), bbox_inches='tight')
        plt.close()
    
    def store(self, epoch):
        log.info("*" * 50)
        log.info("Epoch: {}  Iters: {}".format(epoch, self.opt.niter))
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.save_network(epoch, self.savepath)
        self.save_image(self.fake_like_sample, epoch, self.savepath)

    def visual(self):
        if self.cc:
            self.D_exp.add_scalar_value('D_loss', self.loss_D.data[0], step=self.cnt)
            self.G_exp.add_scalar_value('G_loss', self.loss_G.data[0], step=self.cnt)
            self.index_exp.add_scalar_value('index', self.best_netG_index, step=self.cnt)
    
    def __str__(self):
        netG = self.netG.__str__()
        netD = self.netD.__str__()
        return 'Gan:\n' + '{}_{}{}'.format(self.opt.gans_type, netG, netD)

    def gan_type(self):
        '''print what the gan's type
        '''
        return {'G': self.opt.g_model, 'D': self.opt.d_model}
    
    def info(self):
        '''print what the gan's type
        '''
        return {'CG': self.opt.g_model, 'D': self.opt.d_model, 'X': self.X.data.size(), 'Z': self.Z.data.size(), 'label': self.label.data.size()}

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dim', default=784, help='channels of input data')
    parser.add_argument('--z_dim', default=100, help='noise channels')
    parser.add_argument('--mb_size', default=1, help='the size of each batch')
    parser.add_argument('--img_size', default=28, help='for mnsit datasets, it is 28px')
    parser.add_argument('--savepath', default='out/', help='the folder saved .pth and img results')
    parser.add_argument('--display_it', default=500, help='iter how many times, display the result')
    parser.add_argument('--g_model', default='fc_competition', help='what kind of netG')
    parser.add_argument('--d_model', default='fc_competition', help='what kind of netD')
    parser.add_argument('--niter', default=5, help='iter how many times')
    parser.add_argument('--train', default=True, help='trian or test')
    parser.add_argument('--cc', default=False, help='use tensorboard or not')
    parser.add_argument('--cuda', default=False, help='use cuda or not')
    parser.add_argument('--nums',  default=10, help='how many netG to compete!')
    parser.add_argument('--Lambda',  default=10, help='belong to competition')
    parser.add_argument('--random', default=False, help='random the best netG index')
    parser.add_argument('--find_version', default='v1.1', help='v1.0: index by maxium fake_prob | v1.1: minium L1 distance between real and fake img |v1.2: index by minium distance between real and fake prob')
    parser.add_argument('--condition_D', default=10)
    parser.add_argument('--g_network_path', default='/home/eric/Desktop/Project-PY/pro-py27/01GANs/02competitiongan/result/competitionganv1.4/out/netD_epoch_4600.pth')
    parser.add_argument('--d_network_path', default='/home/eric/Desktop/Project-PY/pro-py27/01GANs/02competitiongan/result/competitionganv1.4/out/netD_epoch_3200.pth')

    opt = parser.parse_args()
    if opt.find_version == 'v1.0':
        print True
    gans = _competitionGan(opt)
    print gans
