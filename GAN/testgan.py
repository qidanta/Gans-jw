import argparse
from baseModel import _baseModel

class _testGan(_baseModel):
    def __init__(self, opt):
        super(_testGan, self).__init__(opt)
        self.test = True

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