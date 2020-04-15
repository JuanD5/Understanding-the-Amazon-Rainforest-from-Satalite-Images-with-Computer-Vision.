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
        pdb.set_trace()
        new_name = name + '.jpg'
        amazon_classes.iloc[i,0] = new_name
    return amazon_classes  