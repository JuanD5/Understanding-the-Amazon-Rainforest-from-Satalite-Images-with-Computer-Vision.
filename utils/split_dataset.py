import os
import pdb
import numpy as np
from sklearn.model_selection import train_test_split
import os
import csv
from PIL import Image
import glob
import shutil
from shutil import copyfile
import argparse
import cv2
import pdb 

### SCRIPT PARA DIVIDIR LA BASE DE DATOS EN TRAIN Y TEST. 80% TRAIN Y 20% TEST. 

# Parameters

ap = argparse.ArgumentParser()
ap.add_argument("-val_size", "--val_size", type=float,default=0.2, help="Size of the validation set.")
args = vars(ap.parse_args())

path = '/home/jlcastillo/Database_real/Copy_train_tif'


if not  os.path.exists('/home/jlcastillo/Database_real/Split_train_tif'):# si no existe una carpeta que se llame asi 
        os.mkdir('/home/jlcastillo/Database_real/Split_train_tif') # la crea 

image_list=[]

for filename in os.listdir(path):
    image_list.append(filename)

train,val = train_test_split(image_list, train_size = 0.8 ,test_size = args["val_size"],random_state = None)


for im_train in train:
    if not  os.path.exists('/home/jlcastillo/Database_real/Split_train_tif/train'):# si no existe una carpeta que se llame asi 
        os.mkdir('/home/jlcastillo/Database_real/Split_train_tif/train') # la crea 
    shutil.copy2(os.path.join(path,im_train),'/home/jlcastillo/Database_real/Split_train_tif/train') # copiamos las imagenes a la carpeta de train 

for im_val in val:
    if not  os.path.exists('/home/jlcastillo/Database_real/Split_train_tif/val'):# si no existe una carpeta que se llame asi 
        os.mkdir('/home/jlcastillo/Database_real/Split_train_tif/val') # la crea 
    shutil.copy2(os.path.join(path,im_val),'/home/jlcastillo/Database_real/Split_train_tif/val') # copiamos las imagenes a la carpeta de val 

                
    

        



        

    

    








