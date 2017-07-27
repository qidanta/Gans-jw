# Small Project for me
> just gan!!!!

## HOW TO USE IT?

* in `pytorch` and `python2`
* datasets is `mnist` and `cifar10`
* download [crayon](https://github.com/torrvision/crayon) for visual, command `sudo docker run -d -p 8888:8888 -p 8889:8889 alband/crayon`

## master

* no condition version.
* each netG rise in a competition, and the best netG limits the rest of the network
* **use netG best-one and real-data as real-prop, and others are fake-prop**

### Pre-works - master

* [x] -nets
    * [x] - move create_nets to G.netG
    * [x] - init different weights
* [x] - solver
    * [x] - create mutil solvers for mutil netG_share+netG_indep
    * [x] - create mutil solvers for mutil netG_indep
    * [x] - create solver for netG_share
    * [x] - create solver for mutil netD
* [x] - train
    * [x] - init network's weights and bias of netG and netD
    * [x] - compute prop of fake for each fake sample
    * [x] - find out best netG

### Future-works-master

* [ ] modify batch size=1

### Arch of Gans

![Arch of gans](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/noise-Z.png)

* Step1: any netG share low layer, and indepently higher layers. Generate samples as normal Gan's do
* Step2: netD distinguish true images and fake samples, then and the best netG by the higher prob of fake samples
* Step3: netG_share backward&step followed by best netG, netG_indep backward&step normaly

### Loss format

![netG_Loss](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/netG_loss.gif)

![netD_Loss](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/netD_loss.gif)

## v1.0

v1.0 version of gans always conv!

* base on dcgans, cyclegan
* base on [mygans](https://github.com/JiangWeixian/GANS)

### Pre-works

* [x] - nets
    * [x] build dcgans
    * [x] init conv networks
    * [x] add layers's size to nets
* [x] - training
    * [x] - add cuda to gans
* [x] - vision
    * [x] - add tensorflowboard support by crayon
* [x] - test_mnist
    * [x] - change g_loss_format(in ./test_mnist_dcgan.py)
* [x] - test(pytorch/examples/dcgan.py)
    * [x] - download datasets-cifar10
    * [x] - run this demo


## v1.1

* mutil netG networks
* in this version, no share layers. 
* backward and step just in norm way! best netG loss will be contained in **lambda part**


### Pre-works
* [x] - nets
    * [x] netG: mutil netG
    * [x] netD: single One
    * [x] GAN: edit in class way
* [x] - training
    * [x] batch_size: 64

this version: 
* each netG complete each other
* find best netG index by argmax prob of netG

make some change for find best netG index

* v1.1 - compare netG outputs with origin pictures
* v1.2 - compare fake prob with real prob, try to let fake prob approach real one, **didn't work well!**
* v1.3 - in backward D, take real-like as real one, the rest of netG as fakes
* v1.4 - in backward D, take real-like as fake one, the rest of netG as fakes too!
    * v1.4.1 - take best netG and other netGs in differenet ways. add normal loss_fake to other netG
    * v1.4.2 - batch_size=1, remove l1-distance in backward_G for other netG.
    * v1.4.3 - in v1.1 way backward d
    * v1.4.4 - 
        * batch-size=1
        * take batch size as fake label
        * only step one netG
* v1.5 - read paper<triple gan>, i think we should add  condition to netD!

those are some rule we should follow after times:

* use v1.1 to find best netG
* only modifilde backward's D&G

## v3

* **keywords** - fm_loss

### TODO

* [x] - add continue train version in `conpetition_gan3_2.py`