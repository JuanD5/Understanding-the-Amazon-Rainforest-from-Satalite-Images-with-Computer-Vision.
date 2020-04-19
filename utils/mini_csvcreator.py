import csv
import pandas as pd 
import pdb

path = '/home/jlcastillo/Proyecto/Database/Dataset/train_classes.csv'

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

datav = []  
first_row = ['image_name','tags']
datav.append(first_row) 

datat = []  
first_row = ['image_name','tags']
datat.append(first_row) 


num_total = len(dataframe)
num_train = 20
num_val = 10
print('Total: '+str(num_total))
print('Train: '+str(num_train))
print('Val: '+ str(round(0.2*num_total)))
print('Test: '+ str(round(0.1*num_total)))
for i in range(num_train+num_val):
    if i < num_train:
       row = dataframe.iloc[i]
       new_row = row.tolist()
       data.append(new_row)            

    else:
        rowv = dataframe.iloc[i]
        new_rowv = rowv.tolist()
        datav.append(new_rowv)

        
with open ('mini_train.csv','w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerows(data)                            
    
with open ('mini_val.csv','w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerows(datav) 
    