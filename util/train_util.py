import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

# gernerate samples
def create_netG_indeps_sample(netG_indep, input, condition=0):
    '''create netG_indep_sample by netG in netG_indep

    in v1.0, condition for netG, not support now.

    - Params:
    @netG_indep: a list of netG
    @input: noise input from netG_share outputs
    @condition: condition for netG create sample

    - Returns:
    a list of samples created by netG_indep
    '''
    G_Indep_sample = []
    for net in netG_indep:
        G_Indep_sample.append(net(input))
    return G_Indep_sample

def create_netG_share_sample(netG_share, input, condition=0):
    '''create netG_indep_sample by netG in netG_indep

    in v1.0, condition for netG, not support now.
    in v1.2, mutil lower layers(netG indeps), sigle high layer(netG share)

    - Params:
    @netG_share: sigle high layer's netG
    @input: a list of input from netG_indep outputs
    @condition: condition for netG create sample

    - Returns:
    a list of samples created by netG_indep
    '''
    G_share_sample = []
    for sample in input:
        G_share_sample.append(netG_share(sample))
    return G_share_sample


# computing loss
def netD_fake(indep_samples, netD):
    '''import fake samples, computing those fake prop

    - Params:
    @indep_samples: fake samples from netG

    - Returns:
    prop of those fake samples
    '''
    fake_prop = []
    for sample in indep_samples:
        fake_prop.append(netD(sample))
    return fake_prop

def compute_fake_loss(fake_prop, label):
    '''compute loss of netG by offical funcs

    - Params:
    @fake_prop: the prop of fake picture(sample created by netG)
    @label: BCEloss's label

    - Returns:
    the loss of netG
     '''
    fake_losses = []
    entropy = nn.BCELoss()
    for fake in fake_prop:
         fake_loss  = entropy(fake, label)
         fake_losses.append(fake_loss)
    return fake_losses

def competitin_cross_entry(index, target, dim=10):
    '''computer nn.CrossEntryLoss between index and target
    
    - Params:
    @index : int type, expand to (1, 10)
    @taget: from dataloader
    @dim: control index expand to (1, dim)

    - Returns
    crossEnteyloss
    '''
    label = torch.zeros(1, dim)
    label = Variable(label)
    label.data[0][index] = 1
    entropy = nn.CrossEntropyLoss()
    print -entropy(label, target)
    return -entropy(label, target)

def compute_fm_loss(real_feats, fake_feats, criterion='HingeEmbeddingLoss'):
    '''compute distance bwtween real_feats and fake_feats, instead of l1loss

    - Params:
    @real_feats: real img's features, **not the last output of netD, and is hidden-layers's output**
    @fake_feats: same as upone, but just from fake imgs
    @criterion: criterion type, defalyt is `HingeEmbeddingLoss`
    '''
    if criterion == 'HingeEmbeddingLoss':
        criterion = nn.HingeEmbeddingLoss()
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean() - fake_feat.mean()) * (real_feat.mean() - fake_feat.mean())
        loss = criterion(l2, Variable(torch.ones(l2.size())))
        losses += loss
    return losses

# find best netG
def find_best_netG(fake_prop):
    '''v1.0 find the best netG from netG_indep by its max netG_prop

    - Params:
    @fake_prop: the prop of fake imgs

    - Returns:
    the index of the best netG in neG_indep
    '''
    fake_losses = np.array([(torch.mean(fake)).data.numpy()[0] for fake in fake_prop])
    return np.argmax(fake_losses)

def find_best_netG_v1dot1(entropy, fake, origin):
    '''the min of (fake - origin), the index will be

    - Params:
    @entropy: such as nn.L1loss
    @fake: fake samples
    @origin: the origin samples

    - Returns
    the index of best fake sample
    '''
    tmp = []
    for i, sample in enumerate(fake):
        tmp.append(entropy(sample, origin))
    gaps = np.array([gap.data.numpy()[0] for gap in tmp])
    return np.argmin(gaps)

def find_best_netG_v1dot2(real_prob, fake_prob):
    '''find the one in fake_prob approch real_prob

    - Params:
    @real_prob: netD(X), X = real img
    @fake_prob: a list of netD(fake), fake = fake samples

    - Returns:
    find the index in fake_prob approch real_prob
    '''
    gaps = []
    for fake in fake_prob:
        gap = real_prob - fake
        gap[gap<0] = -gap
        gaps.append(gap)
    gaps = np.array([(torch.mean(fake)).data.numpy()[0] for fake in gaps])
    return np.argmin(gaps)

def find_best_netG_v1dot3(class_prob, target):
    '''find the one in fake_prob approch real_prob

    - Params:
    @class_prob: a list of Variable type, each one is (mb, size, 10) for mnist, from netC
    @target: Variable type, from dataloader. is the goal of netG

    - Returns:
    find the index
    '''
    target_prob = []
    index = target.data[0]
    for i in range(len(class_prob)):
        target_prob.append(class_prob[i].data[0][index])
    target_prob = np.array(target_prob)
    return np.argmax(target_prob)

def compute_dloss(real_prop, fake_prop, label):
    '''v1.0 compute loss of netG and netD by offical funcs

    take the best-prop fake_prop as real-like prop, the real-like prop and real prop as real prop
    the rest of fake_prop as fake_prop

    - Params:
    @real_prop: the prop of dis real imgs
    @fake_prop: the prop of dis fake imgs
    @label: BCEloss's label

    - Returns: 
    the loss of 
    netD: log(D(x)) + log(1 - D(G(z)))
    '''
    netG_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    best_netG_index = find_best_netG(fake_prop)
    real_like_prop = fake_prop[best_netG_index]
    entropy = nn.BCELoss()

    label.data.fill_(0)
    fake_losses = compute_fake_loss(fake_prop, label)
    label.data.fill_(1)
    real_true_loss = entropy(real_prop, label)
    real_like_loss = entropy(real_like_prop, label)
    rest_fake_loss = (sum(fake_losses[i] for i in netG_num) - fake_losses[best_netG_index]) / (len(netG_num) - 1)
    real_loss = real_true_loss + real_like_loss + rest_fake_loss

    return real_loss

def compute_gloss(fake_prop, label):
    '''compute loss of netG by compute_fake_loss funcs

    - Params:
    @fake_prop: the prop of fake picture(sample created by netG)
    @label: BCEloss's label

    - Returns:
    the loss of 
    netG: log(D(G(z)))
     '''

    label.data.fill_(1)
    best_netG_index = find_best_netG(fake_prop)
    fake_losses = compute_fake_loss(fake_prop, label)
    return fake_losses, best_netG_index

# compute-loss cyclegan

# backward&step
def mutil_backward(netG_losses, net_share, net_indeps, index=None):
    '''mutil  backward() for netG_losses, let netG_losses[index].backward() as lastOne
       ... netG_share will backward only followed by netG_losses[index]

    - Params:
    @netG_losses: netG_losses
    @net_share: in v1.2 mean high level layer netG
    @net_indeps: in v1.2 mean low level layer netG
    @index: netG_losses[index] out of backward()
    '''
    for i in range(len(netG_losses)):
        if i == index:
            continue
        net_share.zero_grad()
        net_indeps[i].zero_grad()
        netG_losses[i].backward(retain_variables = True)

    if index != None:
        net_share.zero_grad()
        net_indeps[index].zero_grad()
        netG_losses[index].backward(retain_variables = True)     

def mutil_steps(netG_losses, net_share, net_indeps, index=None):
    '''v1.0 mutil step() for mutil net_solver

    - Params: 
    @netG_losses: loss for netG
    @net_indeps: mutil independly netG, each netG is net_indep
    @net_share: shared netG
    @index: net_indeps[index] be the lastOne to step()

    - Returns:
    no returns
    '''
    for i in range(len(net_indeps)):
        if  i == index:
            continue
        net_indeps[i].step()
    if index != None:
        net_indeps[index].step()
        net_share.step()

# pre-process data
def link_data(data, times, dim):
    '''expand the data

    - Params:
    @data: the data flow in netG
    @dim: the dim-index of data
    @times: torch.cat([data, data]) times's times by the order of dim

    - Returns:
    dim = 1, time =3, the dim of data = (64, 1, 28, 28)
    return the dim of data = (64, 1*(times + 1), 28, 28)
    '''
    temp = data
    for i in range(times):
        data = torch.cat([data, temp], dim)
    return data

def link2condition_data(data, target, classes=10):
    '''add condition to data

    - Params:
    @data: input from dataloader for net, for mnist, (64, x_dim)
    @target dataloader, for mnist, (64, 1)
    @classes: if define, for mnist's condition, it will become (64, classes), not (64, 10)

    - Return:
     same data type as data, for mnist, data.size from (64, x_dim) become (64, x_dim + classes)
    '''
    mb_size = data.size(0)
    condition = torch.zeros(mb_size, classes)
    for i in range(mb_size):
        if isinstance(target, int):
            index = target
        elif isinstance(target, Variable):
            index = int(target.data[i])
        else:
            index = int(target[i])
        condition[i][index] = 1
    print condition
    return torch.cat([data, condition], 1)

def draft_data(fake_samples, cuda):
    '''if cuda is true, draft data from GPU. Otherwise, change nothing

    - Params:
    @fake_samples: netG(z)
    @cuda: true or false

    - Return
    a floattensor
    '''
    if cuda:
        return fake_samples.data.cpu()
    else:
        return fake_samples

def resize_data(samples, img_size, mb_size):
    '''resize samples to (img_size, img_size)

    - Params:
    @samples: img samples, a list of Variale(pytorch) type
    @img_size: finall img size. len of sample(in samples) > img_size*img*size
    @mb_size: how many imgs in sample(in samples)
    '''
    for i in range(len(samples)):
        samples[i].data.resize_((mb_size, img_size, img_size))
        print samples[i].data.size()


# tools
def compute_mean(input, scale):
    '''compute mean of input

    - Params:
    @input: input data
    @scale: sum/scale = mean

    - Returns
    the mean of input data
    '''
    mean = 0
    for data in input:
        mean += torch.mean(data)
    return mean/scale