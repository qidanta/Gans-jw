import argparse
import torch

from util.network_util import weight_init
from baseModel import _baseModel
from G.netG import build_netG()
from D.netD import build_netD()

class _cycleGan(_baseModel):
    def __init__(self, opt):
        super(_testGan, self).__init__(opt)
        self.netG = build_netG(opt.g_model, opt.z_dim)
        self.netD = build_netD(opt.d_model, opt.x_dim)

        netD.apply(weight_init)
        netG.apply(weight_init)

        x = torch.FloatTensor(opt.mb_size, opt.x_dim, opt.img_size, opt.img_size)
        z = torch.FloatTensor(opt.mb_size, opt.z_dim, 1, 1)
        label = torch.FloatTensor(opt.mb_size)

        if self.cuda:
            netD.cuda()
            netG.cuda()
            x, z = x.cuda(), z.cuda()
            label = label.cuda()

    def __str__(self):
        netG = self.netG.__str__()
        netD = self.netD.__str__()
        return 'Gan:\n' + '{}{}'.format(netG, netD)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dim', default=784, help='channels of input data')
    parser.add_argument('--z_dim', default=100)
    parser.add_argument('--g_model', default='fc_competition')
    parser.add_argument('--d_model', default='fc_competition')
    parser.add_argument('--train', default=True, help='trian or test')
    parser.add_argument('--cc', default=False)
    parser.add_argument('--cuda', default=False)
    opt = parser.parse_args()
    print opt.train
    gans = _testGan(opt)
    print gans