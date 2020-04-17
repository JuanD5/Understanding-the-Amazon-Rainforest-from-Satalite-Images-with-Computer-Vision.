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
from dataloader import AmazonDataset, Rescale
import model
import utils
# Argumentos

parser = argparse.ArgumentParser(description='PyTorch resnet18 for Image Multiclassification')
parser.add_argument("--model", type=str, default='AmazonSimpleNet', help="model: AmazonSimpleNet")
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument("-v", action='store_true', help="verbose")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")                    
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cuda = not args.no_cuda and torch.cuda.is_available() # use cuda

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'pin_memory': True} if args.cuda else {}


# -------------------------- LOADING THE DATA --------------------------
# Data augmentation and normalization for training
# Just normalization for validation

print("Initializing Datasets and Dataloaders...")
data_path = '/home/jlcastillo/Proyecto/Database/Dataset/train-jpg'
# Create training, validation and test datasets
train_dataset = AmazonDataset('train.csv', data_path,'labels.txt', transform = transforms.Compose([Rescale((args.input_size, args.input_size)), transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

#Val
val_dataset = AmazonDataset('val.csv', data_path,'labels.txt',transform = transforms.Compose([Rescale((args.input_size, args.input_size)), transforms.ToTensor()]))
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# TEST call your dataset function def __init__(self, csv_file, data_path, transform=None)
test_dataset = AmazonDataset('test.csv', data_path, 'labels.txt',transform = transforms.Compose([Rescale((args.input_size, args.input_size)), transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# check the size of your datatset
dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(val_dataset)
dataset_sizes['test'] = len(test_dataset)
print('Training dataset size:', dataset_sizes['train'])
print('Validation dataset size:', dataset_sizes['val'])
print('Test dataset size:', dataset_sizes['test'])

# -------------------------- MODEL --------------------------
## URL`s a los pesos
RESNET_18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

"""
model = models.resnet18(num_classes=17)
#resnet18.classifier = [nn.Linear(resnet18.fc.in_features, 17)]

#resnet101 = models.resnet101(pretrained = True, progress = True)

for param in model.parameters():
    param.requires_grad = True

if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True
else:
        
    ## Cargamos los pesos pre entrenados a las redes
    state = model_zoo.load_url(RESNET_18)
     # eliminate fully connected layers weights (trained for 1000 categories)
    state = {x: state[x] for x in state if not x.startswith('fc')}

    # current weights (not the pretrained model)
    model_state = model.state_dict()
    # update state_dict with the pretrained model
    model_state.update(state)
    # load weights into the model
    model.load_state_dict(model_state)
"""
"""
state101 = model_zoo.load_url(RESNET_101)
# current weights (not the pretrained model)
model_state101 = resnet101.state_dict()
# update state_dict with the pretrained model
model_state101.update(state18)
# load weights into the model
resnet101.load_state_dict(model_state101)
"""
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

##########################################################################################
#Model names:
model_names = sorted(name for name in model.__dict__
    if name.startswith("Planet")
    and callable(model.__dict__[name]))

def train(net, loader, criterion, optimizer, verbose = False):
    net.train()
    running_loss = 0
    running_accuracy = 0

    for i, (X,y) in enumerate(loader):
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        output = net(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
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
    for i, (X,y) in enumerate(loader):
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        acc = utils.get_multilabel_accuracy(output, y)
        targets = torch.cat((targets, y.cpu().data), 0)
        predictions = torch.cat((predictions,output.cpu().data), 0)
        running_loss += loss.data[0]
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

    for e in range(args.epochs):
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
                'arch': args.model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break