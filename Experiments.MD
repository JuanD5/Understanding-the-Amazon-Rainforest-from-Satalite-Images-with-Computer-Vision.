# Experimentos:

Lista para acceder mas rápido a cada uno:

1. [Run_0](#run0): Nuestro baseline.
2. [Run_1](#run1): Nuestro baseline con pesos pre-entrenados de ResNet-18.



### - Baseline:

<a name="run0"></a>

Nombre del modelo: `AmazonSimpleNet-run-0.pth.tar`

Nombre de la carpeta del logfile: `AmazonSimpleNet/run-0`

Tiempo aproximado: 18 minutos/época. 

F-score: 0.860, en la última época. 

**Parámetros:**

+ --model MODEL]  AmazonSimpleNet
+ [--batch-size N]  2
+ [--test-batch-size N] 128
+ [--epochs N] 15
+ [--lr LR]  0.01
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 5 
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt



### - Baseline ResNet18 pre-entrenado *-[Jess] Corriendo*

<a name="run1"></a>

Nombre del modelo: 

Nombre de la carpeta del logfile: 

Tiempo aproximado: 

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  2
+ [--test-batch-size N] 128
+ [--epochs N] 15
+ [--lr LR]  0.01
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 5 
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt