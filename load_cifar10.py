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
from util.train_util import (link_data, draft_data)
from util.vision_util import (create_sigle_experiment)

# loading datasets
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar10_data/torch_cifar10data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(64),
                       transforms.ToTensor()
                   ])),
    batch_size=64, shuffle=True, **{})