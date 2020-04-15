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

#pdb.set_trace()
# For reading the csv_file with the annotations of the classes. 
#amazon_classes = pd.read_csv('/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv')
# Esto devuelve algo asi: 
        #train_0                                       haze primary
        #train_1                    agriculture clear primary water
        #train_2                                      clear primary
#  Nombre de la imagen con su correspondiente clase.         

def rename (csv_file):
    csv_file = pd.read_csv('/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv')
    amazon_classes = csv_file
    for i in len(amazon_classes):
        name = amazon_classes.iloc[i,0]
        new_name = name + '.jpg'
        amazon_classes.iloc[i,0] = new_name
    return amazon_classes    





class AmazonDataset(Dataset):

    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args: 
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.classes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.classes)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.classes.iloc[idx,0])
        image = io.imread(img_name)
        category = self.classes.iloc[idx,1:]
        sample = {'image':image,'category':category}
        if self.transform:
            sample = self.transform(sample)

        return sample    
    
            

amazon_dataset = AmazonDataset(csv_file='/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv',
                                    root_dir='/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train-jpg')

fig = plt.figure()

for i in range(len(amazon_dataset)):

    sample = amazon_dataset[i]+'.jpg' # TOCA ARREGLAR ESTO PORQUE SOLO LLEGA HASTA EL PEDAZO DE EL PATH/train_0 # No encuentra la imagen porque no tiene extensión .jpg

    print(i, sample['image'].shape, sample['category'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    if i == 3:
        plt.show()
        break










