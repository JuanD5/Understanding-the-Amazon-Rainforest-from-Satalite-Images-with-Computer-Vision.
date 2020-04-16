import csv
import pandas as pd 
import pdb

path = '/home/jlcastillo/Proyecto/Understanding-The-Amazon/Dataset/train_classes.csv'

def rename (csv_file):
    file_name = pd.read_csv(csv_file)
    amazon_classes = file_name
    for i in range(len(amazon_classes)):
        name = amazon_classes.iloc[i,0]
        new_name = name + '.jpg'
        amazon_classes.iloc[i,0] = new_name
    return amazon_classes    

dataframe = rename(path)

data = []  
first_row = ['image_name','tags']
data.append(first_row) 


for i in range(len(dataframe)):
    row = dataframe.iloc[i]
    new_row = row.tolist()
    data.append(new_row)



with open ('train.csv','w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerows(data)                            
    

    










    



       


