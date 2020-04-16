import pandas as pd
import matplotlib.pyplot as plt

#Importamos los datos de los csv's

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Build list with unique labels
label_list = []
for tag_str in train_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)


train_df.head()

for label in label_list:
    train_df[label] = train_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


# Build list with unique labels
label_list = []
for tag_str in test_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

for label in label_list:
    test_df[label] = test_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


plt.figure(1)
plt.title('Distribución Labels en el set de Entrenamiento')
plt.xlabel('Label')
plt.ylabel('Apariciones')
train_df[label_list].sum().sort_values().plot.bar()

plt.figure(2)
plt.title('Distribución Labels en el set de Validación')
plt.xlabel('Label')
plt.ylabel('Apariciones')
test_df[label_list].sum().sort_values().plot.bar()