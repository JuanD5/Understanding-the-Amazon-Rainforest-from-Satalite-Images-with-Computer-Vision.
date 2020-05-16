import os
import pandas as pd
import csv

path = '/home/jlcastillo/Database_real/test_full'

"""
with open ('train.csv','w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerows(data)  
"""

with open('test_v2.csv', 'w') as file:
  writer = csv.writer(file)
  writer.writerow(['image_name'])
  for path, dirs, files in os.walk(path):
    for filename in files:
        writer.writerow([filename])