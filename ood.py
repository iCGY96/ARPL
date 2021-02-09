import os
import sys
import argparse
import datetime
import time
import csv
import os.path as osp
import numpy as np
import warnings
import importlib
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.utils as vutils

import datasets.datasets as datasets
from models.models import ConvNet
from models.resnet import ResNet34
from models.resnetABN import resnet34ABN
from models import gan
from utils import Logger, save_networks, save_GAN, load_networks, mkdir_if_missing
from core import train, train_cs, test

parser = argparse.ArgumentParser("ARPLoss")

# dataset
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--out-dataset', type=str, default='kmnist')
parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--loss', type=str, default='ARPLoss')

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='cnn')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Samples", default=False)

args = parser.parse_args()
options = vars(args)

sys.stdout = Logger(osp.join(options['outf'], 'logs.txt'))

def main():
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    feat_dim = 2 if 'cnn' in options['model'] else 512

    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    dataset = datasets.create(options['dataset'], **options)
    out_dataset = datasets.create(options['out_dataset'], **options)

    trainloader, testloader = dataset.trainloader, dataset.testloader
    outloader = out_dataset.testloader

    options.update(
        {
            'num_classes': dataset.num_classes
        }
    )

    print("Creating model: {}".format(options['model']))
    if 'cnn' in options['model']:
        net = ConvNet(num_classes=dataset.num_classes)
    else:
        if options['cs']:
            net = resnet34ABN(num_classes=dataset.num_classes, num_bns=2)
        else:
            net = ResNet34(dataset.num_classes)


    if options['cs']:
        print("Creating GAN")
        nz = options['nz']
        netG = gan.Generator32(1, nz, 64, 3) # ngpu, nz, ngf, nc
        netD = gan.Discriminator32(1, 3, 64) # ngpu, nc, ndf
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()
    
    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['dataset'], options['loss'], str(options['weight_pl']), str(options['cs']))
    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
        return

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    if options['cs']:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
 
    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

    start_time = time.time()

    score_now = 0.0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        if options['cs']:
            train_cs(net, netD, netG, criterion, criterionD,
                optimizer, optimizerD, optimizerG,
                trainloader, epoch=epoch, **options)

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test")
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

            save_networks(net, model_path, file_name, criterion=criterion)
            if options['cs']: 
                save_GAN(netG, netD, model_path, file_name)
                fake = netG(fixed_noise)
                GAN_path = os.path.join(model_path, 'samples')
                mkdir_if_missing(GAN_path)
                vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(GAN_path, epoch), normalize=True)

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

if __name__ == '__main__':
    main()

