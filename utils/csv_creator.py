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

datat = []  
first_row = ['image_name','tags']
datat.append(first_row) 


num_total = len(dataframe)
num_train = round(0.8*num_total)

for i in range(num_total):
    if i < num_train:
       row = dataframe.iloc[i]
       new_row = row.tolist()
       data.append(new_row)            

    else:
        rowt = dataframe.iloc[i]
        new_rowt = rowt.tolist()
        datat.append(new_rowt) 

        
with open ('train.csv','w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerows(data)                            
    
with open ('test.csv','w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerows(datat) 
    










    



       


