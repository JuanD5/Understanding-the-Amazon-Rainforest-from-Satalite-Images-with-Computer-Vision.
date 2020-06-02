##---------------------------archivo principal -------------------------------##
##-----Jessica Castillo, Juan David Garcia, Juan Francisco Suescun------------##

## Importamos las librerias
from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb 
#import debug 

import copy
import time
import argparse
import os.path as osp
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from dataloader import AmazonDataset, Rescale, AmazonDatasetNIR
import model
import utils
from tensorboard_logger import configure, log_value
from sklearn.metrics import fbeta_score

from tqdm import tqdm
from tqdm.auto import trange

# Argumentos

parser = argparse.ArgumentParser(description='PyTorch resnet18 for Image Multiclassification')
parser.add_argument("--model", type=str, default='AmazonResNet101', help="model: AmazonResNet101")
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument("-v", action='store_true', help="verbose")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument("--patience", type=int, default=20, help="early stopping patience")                    
parser.add_argument('--input_size', type=int, default=256, metavar='N',
                    help='Input image size (default: 256)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', type=str, default='1',
                    help='GPU(s) to use (default: 1)')
parser.add_argument('--nir_channel',type = str, default= 'normal',
                    help = 'Representation options: NIR-R-G, NIR-R-B, NDVI-spectral, NDVI-calculated, NDWI, NIR-combined')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cuda = not args.no_cuda and torch.cuda.is_available() # use cuda

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'pin_memory': True} if args.cuda else {}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# -------------------------- LOADING THE DATA --------------------------
# Data augmentation and normalization for training
# Just normalization for validation

train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

print("Initializing Datasets and Dataloaders...")
data_path = '/home/jlcastillo/Database_real/train-jpg'

# Create training, validation and test datasets
train_dataset = AmazonDataset('csv/train.csv', data_path,'csv/labels.txt', args.nir_channel, transform = train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

#Val
val_dataset = AmazonDataset('csv/val.csv', data_path,'csv/labels.txt', args.nir_channel,transform = train_transforms)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# check the size of your datatset
dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(val_dataset)

print('Training dataset size:', dataset_sizes['train'])
print('Validation dataset size:', dataset_sizes['val'])


# -------------------------- MODEL --------------------------
## URL`s a los pesos
#RESNET_18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
#RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'



###### Para salvar modelos y logs ##################
# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup tensorboard folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
	run += 1
	current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)

# Tensorboard viz. tensorboard --logdir {yourlogdir}. Requires tensorflow.
configure(current_dir, flush_secs=5)


##########################################################################################
#Model names:
model_names = sorted(name for name in model.__dict__
    if name.startswith("Planet")
    and callable(model.__dict__[name]))


def train(net, loader, criterion, optimizer, verbose = False):
    net.train()
    running_loss = 0
    running_accuracy = 0
    for i, (X,y) in enumerate(tqdm(loader,desc='Train')):
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        output = net(X)
        #output = sigmoid(output)
        #if i ==76:
        #    pdb.set_trace()
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()                                     
        acc = utils.get_multilabel_accuracy(output, y)
        running_accuracy += acc
        if i%400 == 0 and verbose:
            pct = float(i+1)/len(loader)
            curr_loss = running_loss/(i+1)
            curr_acc = running_accuracy/(i+1)
            print('Complete: {:.2f}, Loss: {:.2f}, Accuracy: {:.4f}'.format(pct*100,
                        curr_loss, curr_acc))
    return running_loss/len(loader), running_accuracy/len(loader)

def validate(net, loader, criterion):
    net.eval()
    running_loss = 0
    running_accuracy = 0
    targets = torch.FloatTensor(0,17) # For fscore calculation
    predictions = torch.FloatTensor(0,17)
    for i, (X,y) in enumerate(tqdm(loader,desc='Val')):
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        acc = utils.get_multilabel_accuracy(output, y)
        targets = torch.cat((targets, y.cpu().data), 0)
        predictions = torch.cat((predictions,output.cpu().data), 0)
        running_loss += loss.item()
        running_accuracy += acc
    fscore = fbeta_score(targets.numpy(), predictions.numpy() > 0.23,
                beta=2, average='samples')
    return running_loss/len(loader), running_accuracy/len(loader), fscore


if __name__ == '__main__':
    net = model.__dict__[args.model]()
    criterion = torch.nn.BCELoss()

    if args.cuda:
        net, criterion = net.cuda(), criterion.cuda()
    # early stopping parameters
    patience = args.patience
    best_loss = 1e4

    # Print model to logfile
    print(net, file=logfile)

    # Change optimizer for finetuning
    if args.model=='AmazonSimpleNet':
        optimizer = optim.Adam(net.parameters())
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            #best_acc1 = checkpoint['best_acc1']
            #if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                #best_acc1 = best_acc1.to(args.gpu)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for e in range(args.start_epoch, args.epochs):
        start = time.time()
        train_loss, train_acc = train(net, train_loader, criterion, optimizer, args.v)
        val_loss, val_acc, fscore = validate(net, val_loader, criterion)
        end = time.time()

        # print stats
        stats ="""Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t fscore: {:.3f}\t
                time: {:.1f}s""".format( e, train_loss, train_acc, val_loss,
                val_acc, fscore, end-start)
        print(stats)
        print(stats, file=logfile)
        log_value('train_loss', train_loss, e)
        log_value('val_loss', val_loss, e)
        log_value('fscore', fscore, e)

        #early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            utils.save_model({
                'epoch': e + 1,
                'arch': args.model,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss
            }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break