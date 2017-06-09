import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

def testparams(a, b, c, d=1):
    print a
    print b
    print c
    print d

def testzip():
    i = [1]*3
    jlist = [2, 3, 4]
    print zip(i, jlist)

def testzip2():
    i = [1]*3
    jlist = [2, 3, 4]
    ziplist = [ (iindex, jindex) for iindex, jindex in zip(i, jlist) ]
    print ziplist

def testtype():
    item = (1, 2)
    print type(item)

def testrandom():
    print np.random.rand(1)*0.1

def testadd():
    a = torch.randn(4)
    b = torch.randn(4)
    z = torch.zeros(1)
    print z
    print (a + b)/2
    # testsum
    elist = []
    elist.append(a)
    elist.append(b)
    b = sum(elist)
    print b

def testremovelist():
    a = [1, 2, 3, 4]
    length = range(len(a))
    extra = list(length).remove(2)
    print length

def testdir():
    print dir(list)

def testcontinue():
    a = [1, 2, 3, 4]
    for i in a:
        if i == 2:
            continue
        print i

def testyeild():
    a = [1, 2, 3]
    for i in a:
        yield i

def testzip3():
    result = testyeild()
    a = [1, 2, 3]
    print zip(result, a)

def list2npmax():
    a = np.array([1, 2, 3])
    print type(a)

def listresize():
    a = [1, 2, 3, 4]
    return a[1:3]

def testtuple():
    a = (1, 2)
    print type(a) == tuple

def testbackward():
    x = Variable(torch.ones(2, 2), requires_grad=True)
    y = x + 2
    z = Variable(torch.ones(2, 2), requires_grad=True)
    f = y * z 
    out = f.mean()
    print 'first\n'
    print x.grad
    out.backward()
    print 'second\n'
    print x.grad
    print 'second z\n'
    print z.grad
def testrange():
    for i in range(1):
        print i
def testbce():
    real = torch.ones(4, 1) * 0.2
    fake = torch.ones(4, 1) * 0.2
    real_target = torch.ones(4, 1)
    fake_target = torch.zeros(4, 1)
    real = Variable(real)
    fake = Variable(fake)
    real_target = Variable(real_target)
    fake_target = Variable(fake_target)
    criterion = nn.BCELoss()
    real_bce_loss = criterion(real, real_target)
    fake_bce_loss =  criterion(fake, fake_target)
    print fake_bce_loss
    real_loss = torch.mean(torch.log(real))
    fake_loss = torch.mean(torch.log(1 - fake))
    print fake_loss

def testjs():
    cd, ab = [0.75], [0.25]
    for i in range(100):
        cd.append(random.uniform(0,1))
        ab.append(random.uniform(0,1))
    length = len(cd)
    cd = np.array(cd, dtype='float')
    ab = np.array(ab, dtype='float')
    js = -0.5*(np.sum(cd*np.log(cd/(0.5*(cd+ab))))+np.sum(ab*np.log(ab/(0.5*(cd+ab)))))
    print js
    print np.log(2)

def JSD():
    P, Q = [0.25, 0, 0.75], [0.75, 0.25, 0]
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    print _P, _Q
    print 0.75*np.log(1.5)+0.25*np.log(0.5)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def testtype():
    for index in range(10):
        print index

def testtorchsum():
    a = torch.ones(2,3)
    b = torch.ones(2,3)
    sumlist = [a, b]
    print mean(sumlist)

def testrandomint():
    i = random.randint(0, 4)
    for j in range(10):
        print random.randint(0, 4)

def testcondition():
    c = torch.zeros(1, 10)
    print c
    c[1] = 1
    print c

def reducearrray():
    a = np.array([1, 2, 3])
    b = 3
    print a - b

def testfilter():
    b = np.ones((64,1))
    a = Variable(torch.ones(64,1) + 1)
    a[a>1]=-a
    print a

def testindex():
    condition = torch.ones(5, 10)
    target = torch.zeros(1,10)
    print target[0][1]

def testint():
    a = 1
    print isinstance(a, int)

def testL1():
    X = torch.ones(2, 784)
    X_split = torch.ones(1, 784)
    sample = torch.FloatTensor(2, 784)
    sample_split_up = torch.FloatTensor(1, 784)
    sample_split_down = torch.FloatTensor(1, 784)
    X, sample = Variable(X), Variable(sample)
    X_split, sample_split_up, sample_split_down = Variable(X_split), Variable(sample_split_up), Variable(sample_split_down)
    sample.data.normal_(0, 1)
    sample_split_up.data.copy_(sample.data[0, :])
    sample_split_down.data.copy_(sample.data[1, :])
    crientL1 = torch.nn.L1Loss()
    all_loss = crientL1(sample, X)
    split_up_loss = crientL1(sample_split_up, X_split)
    split_down_loss = crientL1(sample_split_down, X_split)
    print all_loss
    print (split_down_loss + split_up_loss)/2

def testmean():
    X = torch.ones(2, 784)
    X_split = torch.ones(1, 784)
    x_list = (X, X_split)
    print X.mean(0)



if __name__ == '__main__':
    config = [1, 2, 3]
    testmean()
    #testtype()
    #list2npmax()
    #testzip3()
    #result = testyeild()
    #testcontinue()
    #testzip2()
    #testdir()
    #testremovelist()
    #testadd()
    #testrandom()
    #testtype()
    #testzip()
    #testparams(*config)