from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb 
from PIL import Image

def get_labels(fname):
    with open(fname,'r') as f:
        labels = [t.strip() for t in f.read().split(',')]
    labels2idx = {t:i for i,t in enumerate(labels)}
    idx2labels = {i:t for i,t in enumerate(labels)}
    return labels,labels2idx,idx2labels

class AmazonDataset(Dataset):

    def __init__(self,csv_file,root_dir,labels_file,transform=None):
        """
        Args: 
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.filenames = pd.read_csv(csv_file)
        self.root_dir = root_dir
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

        image = io.imread(os.path.join(self.root_dir,img_name))

        labels = self.filenames.ix[idx, 1]
        target = torch.zeros(self.n_labels)
        label_idx = torch.LongTensor([self.labels2idx[tag] for tag in labels.split(' ')])
        target[label_idx] = 1

        if self.transform:
            image = self.transform(image)
        
        return image, target


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

