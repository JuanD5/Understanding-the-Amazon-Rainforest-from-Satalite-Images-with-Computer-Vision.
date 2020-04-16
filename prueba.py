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
   

## Guardamos el dataframe con la información de labels para facilitar visualización
labels_df = pd.read_csv('/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv')

label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)


# Add features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
labels_df.head()

## Hacemos un histograma con la cantidad de veces que aparece cada categoria

labels_df[label_list].sum().sort_values().plot.bar()

porcentaje_division = 0.8

pdb.set_trace()
dataset_train = labels_df.iloc[:, :round(porcentaje_division*(len(labels_df)))]
dataset_test = labels_df.iloc[:, round(porcentaje_division* len(labels_df)):]
