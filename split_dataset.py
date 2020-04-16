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
ap.add_argument("-test_size", "--test_size", type=float,default=0.2, help="Size of the test set.")
args = vars(ap.parse_args())

path = '/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train-jpg'


if not  os.path.exists(os.path.join(os.getcwd(),'Split_Amazon_Dataset')):# si no existe una carpeta que se llame asi 
        os.mkdir(os.path.join(os.getcwd(),'Split_Amazon_Dataset')) # la crea 


image_list=[]
for filename in os.listdir(path):

        image_list.append(filename)

        train,test = train_test_split(image_list,test_size = args["test_size"])

        for im_train in train:
            if not  os.path.exists(os.path.join(path,'Split_Amazon_Dataset','train')):# si no existe una carpeta que se llame asi 
                os.mkdir(os.path.join(path,'Split_Amazon_Dataset','train')) # la crea 
            shutil.copy2(os.path.join(path,im_train),os.path.join(path,'Split_Amazon_Dataset','train')) # copiamos las imagenes a la carpeta de train 

        for im_test in test:
            if not  os.path.exists(os.path.join(path,'Split_Amazon_Dataset','test')):# si no existe una carpeta que se llame asi 
                os.mkdir(os.path.join(path,'Split_Amazon_Dataset','test')) # la crea 
            shutil.copy2(os.path.join(path,im_test),os.path.join(path,'Split_Amazon_Dataset','test')) # copiamos las imagenes a la carpeta de test 

        
    

        



        

    

    








