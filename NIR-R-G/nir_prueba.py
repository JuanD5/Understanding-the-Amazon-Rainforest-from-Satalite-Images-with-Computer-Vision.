import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import pdb
import os
from spectral import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('agg')
import pylab as plt

##INFRARED UTILS

def infrared_channel_converter(path,kind = 'NDVI-calculated' ):
    """
    Takes the path to a .tif image and returns the RGB image and the desired Infra red respresentation.
    Default representation set to NDVI-calculated.
    Representation options: NIR-R-G, NIR-R-B, NDVI-spectral, NDVI-calculated,NDWI
    """
    img = io.imread(path)
    img_rgb = get_rgb(img, [2, 1, 0]) # RGB
    RG = get_rgb(img, [3, 2, 1]) # NIR-R-G
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

    return img_rgb,inf_out

root_dir = os.path.join(os.getcwd(), '../imagenes_prueba')
img_name = 'train_0'
nir_channel = 'NIR-R-B'

image = Image.open(os.path.join(root_dir,img_name+'.tif'))  
rgb_image, nir_image = infrared_channel_converter(os.path.join(root_dir,img_name+'.tif'), nir_channel)
plt.imshow(rgb_image)
plt.savefig('a.png')
plt.imshow(nir_image)

plt.savefig('b.png')
image = np.dstack((rgb_image,nir_image))
