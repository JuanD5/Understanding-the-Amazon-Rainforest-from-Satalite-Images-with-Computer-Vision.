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

# Argumentos

parser = argparse.ArgumentParser(description='PyTorch resnet18 for Image Multiclassification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--input_size', type=int, default=256, metavar='N',
                    help='Input image size (default: 256)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


#kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


# -------------------------- LOADING THE DATA --------------------------
# Data augmentation and normalization for training
# Just normalization for validation

print("Initializing Datasets and Dataloaders...")
data_path = '/home/jlcastillo/Proyecto/Database/Dataset/train-jpg'
# Create training and test datasets
train_dataset = AmazonDataset('train.csv', data_path, transform = transforms.Compose([Rescale((args.input_size, args.input_size)), transforms.ToTensor()]))
##ToTensor normaliza
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# TEST call your dataset function def __init__(self, csv_file, data_path, transform=None)
test_dataset = AmazonDataset('test.csv', data_path, transform = transforms.Compose([Rescale((args.input_size, args.input_size)), transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# check the size of your datatset
dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['test'] = len(test_dataset)
print('Training dataset size:', dataset_sizes['train'])
print('Test dataset size:', dataset_sizes['test'])

# -------------------------- MODEL --------------------------
resnet18 = models.resnet18(pretrained=True, progress = True)
resnet101 = models.resnet101(pretrained = True, progress = True)

## URL`s a los pesos
RESNET_18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

## Cargamos los pesos pre entrenados a las redes
state18 = model_zoo.load_url(RESNET_18)
# current weights (not the pretrained model)
model_state18 = resnet18.state_dict()
# update state_dict with the pretrained model
model_state18.update(state18)
# load weights into the model
resnet18.load_state_dict(model_state18)


state101 = model_zoo.load_url(RESNET_101)
# current weights (not the pretrained model)
model_state101 = resnet101.state_dict()
# update state_dict with the pretrained model
model_state101.update(state18)
# load weights into the model
resnet101.load_state_dict(model_state101)


