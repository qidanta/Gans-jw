import os

import torch
from pycrayon import CrayonClient

import torchvision.utils as vutils
from util.vision_util import create_sigle_experiment


class _baseModel(object):
    '''Base Model combine netG and netD to became a gans's model

    - Attributes:
    @opt: options for config gans'model
    @train: train or test
    @cc: crayon client or not
    @cuda: use cuda or not
    '''

    def __init__(self, opt):
        self.opt = opt
        self.istrain = opt.train
        self.cc = CrayonClient(hostname="localhost") if opt.cc else opt.cc
        self.cuda = opt.cuda

    def create_tensorboard(self):
        '''use docker create tensorboard
        '''
        if self.cc:
            self.cc.remove_all_experiments()
            self.D_exp = create_sigle_experiment(self.cc, 'D_loss')
            self.G_exp = create_sigle_experiment(self.cc, 'G_loss')
    
    def draft_data(self, input):
        '''input from datasetsloader, put those into X/Z
        '''
        pass

    def backward_D(self):
        '''backwrad netD
        '''
        pass

    def train(self):
        '''train gans
        '''
        pass

    def test(self):
        '''test gans
        '''
        pass

    def save_network(self, it, savepath):
        '''save checkpoints of netG and netD in savepath

        - Params:
        @it: number of iterations
        @savepath: in savepath, save network parameter
        '''
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (savepath, it))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (savepath, it))
    
    def load_networkG(self, g_network_path):
        '''load network parameters of netG and netD

        - Params:
        @g_network_path: the path of netG
        '''
        self.netG.load_state_dict(torch.load(d_network_path))
    def load_networkD(self, d_network_path):
        '''load network parameters of netG and netD

        - Params:
        @d_network_path: the path of netG
        '''
        self.netD.load_state_dict(torch.load(d_network_path))

    def save_image(self, fake, it , savepath):
        '''save result of netG output

        - Params:
        @fake: the output of netG
        @it: number of iterations
        @savepath: in savepath, save network parameter
        '''
        vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (savepath, it))
