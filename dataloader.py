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
   

class AmazonDataset(Dataset):

    def __init__(self,csv_file,root_dir,transform=None):
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

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

    def __getitem__(self,idx):

        sample = self.filenames.iloc[int(idx)]
        img_name = sample['image_names']
        label = sample['tags']

        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)
        return image, label


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

"""
# For reading the csv_file with the annotations of the classes. 
#amazon_classes = pd.read_csv('/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv')
# Esto devuelve algo asi: 
        #train_0                                       haze primary
        #train_1                    agriculture clear primary water
        #train_2                                      clear primary
#  Nombre de la imagen con su correspondiente clase.     

def rename (csv_file):
    file_name = pd.read_csv(csv_file)
    amazon_classes = file_name
    for i in range(len(amazon_classes)):
        name = amazon_classes.iloc[i,0]
        new_name = name + '.jpg'
        amazon_classes.iloc[i,0] = new_name
    return amazon_classes    


amazon_dataset = AmazonDataset(csv_file = rename('/home/jlcastillo/Proyecto/Database/Dataset/train_classes.csv'),
                                    root_dir='/home/jlcastillo/Proyecto/Database/Dataset/train-jpg')

"""
"""fig = plt.figure()

for i in range(len(amazon_dataset)):

    pdb.set_trace()
    sample = amazon_dataset[i] 
    print(i, sample['image'].shape, sample['category'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    if i == 3:
        plt.show()
        break"""

"""
##--- Las imagenes quedan guardadas en amazon_dataset, NOMBRE Y ANOTACIONES----#


## Guardamos el dataframe con la informacion de labels para facilitar visualizacion
labels_df = pd.read_csv('/home/jlcastillo/Proyecto/Database/Dataset/train_classes.csv')

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

dataset_train = labels_df.iloc[:, :porcentaje_division* round(len(labels_df))]
dataset_test = labels_df[:, porcentaje_division* round(len(labels_df)):]



"""