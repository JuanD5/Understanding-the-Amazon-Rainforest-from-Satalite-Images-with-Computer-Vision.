import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import pdb
import os
from spectral import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler


def get_conv_out(layer, input_size):
    w, h = input_size
    F = layer.kernel_size
    S = layer.stride
    P = layer.padding
    w2= (w-F[0]+2*P[0])/S[0]+1
    h2 =(h-F[1]+2*P[1])/S[1]+1
    return w2,h2

def get_pool_out(layer,input_size):
    w, h = input_size
    F = layer.kernel_size
    S = layer.stride
    P = layer.padding
    w2 = (w-F)/S+1
    h2 = (h-F)/S+1
    return w2,h2

def calculate_feature_size(model, input_size):
    for layer in model:
        if type(layer) == nn.Conv2d:
            input_size = get_conv_out(layer, input_size)
        elif type(layer) == nn.MaxPool2d:
            input_size = get_pool_out(layer, input_size)
    return input_size





###         Training Utils

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_multilabel_accuracy(pred, target):
    """ Calculate multilabel accuracy.
        Turn prediction tensor in binary. Compare with target.
        Calculate common elements. To be used for calculating running
        accuracy and total accuracy in training.
    """
    pred = pred > 0.5
    r = (pred == target.bool())
    acc = r.float().cpu().sum().item()
    return acc/(pred.size()[1]*pred.size()[0])

def save_model(model_state,filename):
    """ Save model """
    # TODO: add it as checkpoint
    torch.save(model_state,filename)


def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger
    
def adjust_learning_rate(lr0,optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr0 * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





###                PIL Image Transformations

class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotation(object):
    """Rotate PIL.Image randomly (90/180/270 degrees)with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be rotated.
        Returns:
            PIL.Image: Randomly rotated image.
        """
        if np.random.random() < 0.5:
            deg = np.random.randint(1,3)*90.
            return img.rotate(deg)
        return img

class RandomTranslation(object):
    """Translates PIL.Image randomly (0-10 pixels) with a probability of 0.5."""

    def __init__(self,max_vshift=10, max_hshift=10):
        self.max_vshift = max_vshift
        self.max_hshift = max_hshift

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be translated.
        Returns:
            PIL.Image: Randomly translated image.
        """
        if np.random.random() < 0.5:
            hshift = np.random.randint(-self.max_hshift,self.max_hshift)
            vshift = np.random.randint(-self.max_vshift,self.max_vshift)
            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift))
        return img



##INFRARED UTILS

def infrared_channel_converter(path,kind = 'NDVI-calculated' ):
    """
    Takes the path to a .tif image and returns the RGB image and the desired Infra red respresentation.
    Default representation set to NDVI-calculated.
    Representation options: normal, NIR-R-G, NIR-R-B, NDVI-spectral, NDVI-calculated,NDWI
    """
    img = io.imread(path)
    img_rgb = get_rgb(img, [2, 1, 0]) # RGB
    RG = get_rgb(img, [3, 2, 1]) # NIR-R-G

    if (kind == 'normal'):
        inf_out = img[:,:,3]
    if (kind == 'NIR-R-G'):
        inf_out = RG # NIR-R-G
    if (kind == 'NIR-R-B'):
        inf_out = get_rgb(img, [3, 2, 0]) # NIR-R-B
    if (kind == 'NDVI-spectral'):
        inf_out = ndvi(img, 2, 3)

    #calculate NDVI and NDWI with spectral module adjusted bands
    
    np.seterr(all='warn') # divide by zero, NaN values
    
    if (kind == 'NDVI-calculated'):
        inf_out = (RG[:, :, 0] - RG[:, :, 1]) / (RG[:, :, 0] + RG[:, :, 1]) # (NIR - RED) / (NIR + RED)
    if (kind == 'NDWI'):
        inf_out = (RG[:, :, 2] - RG[:, :, 0]) / (RG[:, :, 2] + RG[:, :, 0]) # (GREEN - NIR) / (GREEN + NIR)
    if (kind == 'NIR-combined'):
        spectral = ndvi(img, 2, 3)
        calculated = (RG[:, :, 0] - RG[:, :, 1]) / (RG[:, :, 0] + RG[:, :, 1])
        ndwi = (RG[:, :, 2] - RG[:, :, 0]) / (RG[:, :, 2] + RG[:, :, 0])
        inf_out = np.dstack((spectral,calculated,ndwi))

    return img_rgb,inf_out