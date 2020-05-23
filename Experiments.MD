# Experimentos:

<a name="up"></a>

Lista para acceder mas rápido a cada uno:

1. [Baseline](#run1): Nuestro baseline.
2. [Baseline ResNet](#run2): Nuestro baseline con pesos pre-entrenados de ResNet-18. Corrió hasta la época 8.
3. [Baseline ResNet](#run3): Nuestro baseline con pesos pre-entrenados de ResNet-18. Este sí corrió completo.
4. [ResNet101](#run4): Modelo pre-entrenado. 
5. [ResNet101](#run5): Modelo pre-entrenado en más épocas. 
6. [ResNet101](#run6): Modelo pre-entrenado, sin validar.
7. [ResNet101](#run7): Modelo pre-entrenado, validando y usando infrarrojo. 

### - [Baseline](#up):

<a name="run1"></a>

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



### - [Baseline ResNet18 pre-entrenado](#up) 

<a name="run2"></a>

Nombre del modelo: `AmazonResNet18-run-1.pth`

Nombre de la carpeta del logfile: `AmazonResNet18/run-1`

Tiempo aproximado: 23 minutos/época. 

F-score: 0.470, en la última época. 

**No corrió completo, se quedó sin paciencia y se detuvo en la época 8.**

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  1
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



### - [Baseline ResNet18 pre-entrenado](#up)

<a name="run3"></a>

Nombre del modelo: `AmazonResNet18-run-2.pth`

Nombre de la carpeta del logfile: `AmazonResNet18/run-2`

Tiempo aproximado: 18 minutos/época

F-score: 0.89, en la última época. 

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  2
+ [--test-batch-size N] 128
+ [--epochs N] 15
+ [--lr LR]  0.01
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] **10** 
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt

### - [Baseline ResNet101 pre-entrenado](#up)

<a name="run4"></a>

Nombre del modelo: `AmazonResNet101-run-0.pth`

Nombre de la carpeta del logfile: `AmazonResNet101/run-0`

Tiempo aproximado: 50 minutos/ época

F-score: 0.936

**Parámetros:**

+ --model MODEL] AmazonResNet101
+ [--batch-size N]  4
+ [--test-batch-size N] 128
+ [--epochs N] 15
+ [--lr LR]  0.01
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] **15** 
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt

### - [ResNet101 pre-entrenado](#up)

<a name="run5"></a>

Nombre del modelo: `AmazonResNet101-run-1.pth`

Nombre de la carpeta del logfile: `AmazonResNet101/run-1`

Archivo de predicción: `pred0.cvs`

F-score: **Oficial:** 0.84604

**Parámetros:**

+ --model MODEL] AmazonResNet101
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 40
+ [--lr LR]  0.01
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] **15** 
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt

### - [ResNet101 pre-entrenado, validando, usando infrarrojo](#up)

<a name="run7"></a>

Base de datos de train (80%) y Validación (20%). Canal de infrarrojo: NDVI-spectral.

Nombre del modelo: `AmazonResNet101-run-3.pth.tar`

Nombre de la carpeta del logfile: `AmazonResNet101/run-3`

Archivo de predicción: 

F-score: **Oficial:** 

**Parámetros:**

+ --model MODEL] AmazonResNet101
+ [--batch-size N]  4
+ [--test-batch-size N] 128
+ [--epochs N] 40
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] **15** 
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: NDVI-spectral

