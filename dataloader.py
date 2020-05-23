from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from utils import infrared_channel_converter

def get_labels(fname):
    with open(fname,'r') as f:
        labels = [t.strip() for t in f.read().split(',')]
    labels2idx = {t:i for i,t in enumerate(labels)}
    idx2labels = {i:t for i,t in enumerate(labels)}
    return labels,labels2idx,idx2labels

class AmazonDataset(Dataset):

    def __init__(self,csv_file,root_dir,labels_file,nir_channel,transform=None):
        """
        Args: 
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        nir_channel(string): Type of NIR 4th channel desired - options: NIR-R-G, NIR-R-B, NDVI-spectral, NDVI-calculated,NDWI
        transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filenames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.nir_channel = nir_channel
        self.transform = transform

        self.labels, self.labels2idx, self.idx2labels = get_labels(labels_file)
        self.n_labels = len(self.labels)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

    def __getitem__(self,idx):

        sample = self.filenames.iloc[int(idx)]
        img_name = sample['image_name']
        #label = sample['tags']

        #image = Image.open(os.path.join(self.root_dir,img_name+'.tif'))
        rgb_image, nir_image = infrared_channel_converter(os.path.join(self.root_dir,img_name+'.tif'), self.nir_channel)

        image = np.dstack((rgb_image,nir_image))
        labels = self.filenames.ix[idx, 1]
        target = torch.zeros(self.n_labels)
        label_idx = torch.LongTensor([self.labels2idx[tag] for tag in labels.split(' ')])
        target[label_idx] = 1

        if self.transform:
            image = self.transform(image)
        
        return image, target

class TestAmazonDataset(Dataset):

    def __init__(self, csv_file, root_dir, labels_file, nir_channel,transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.labels, self.labels2idx, self.idx2labels = get_labels(labels_file)
        self.n_labels = len(self.labels)
        self.nir_channel = nir_channel
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.ix[idx, 0]+'.tif')
        #img = Image.open(img_name + '.jpg').convert('RGB')
        rgb_image, nir_image = infrared_channel_converter(img_name,self.nir_channel)
        img = np.concatenate((rgb_image,nir_image), axis = 2)

        labels = self.data.ix[idx, 1]
        target = torch.zeros(self.n_labels)
        label_idx = torch.LongTensor([self.labels2idx[tag] for tag in labels.split(' ')])
        target[label_idx] = 1
        if self.transform:
            img = self.transform(img)
        return img, target


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, X):
        image = X

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode = 'constant')
        return img

