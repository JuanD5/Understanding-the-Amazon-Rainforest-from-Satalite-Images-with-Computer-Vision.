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

        self.classes = csv_file
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


amazon_dataset = AmazonDataset(csv_file = rename('/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv'),
                                    root_dir='/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train-jpg')


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


##--- Las imagenes quedan guardadas en amazon_dataset, NOMBRE Y ANOTACIONES----#

label_list = []
for tag_str in amazon_dataset.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)


# Add features for every label
for label in label_list:
    amazon_dataset[label] = amazon_dataset['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
amazon_dataset.head()

## Hacemos un histograma con la cantidad de veces que aparece cada categoria

amazon_dataset[label_list].sum().sort_values().plot.bar()



