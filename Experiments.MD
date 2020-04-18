# Experimentos:

Lista para acceder mas rápido a cada uno:

1. [Baseline](#run0): Nuestro baseline.
2. [Baseline ResNet](#run1): Nuestro baseline con pesos pre-entrenados de ResNet-18. Corrió hasta la época 8.
3. [Baseline ResNet](#run2): Nuestro baseline con pesos pre-entrenados de ResNet-18.



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



### - Baseline ResNet18 pre-entrenado 

<a name="run1"></a>

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



### - Baseline ResNet18 pre-entrenado *-[Jess] Corriendo un nuevo intento*

<a name="run2"></a>

Nombre del modelo: `AmazonResNet18-run-2.pth`

Nombre de la carpeta del logfile: `AmazonResNet18/run-2`

Tiempo aproximado: 

F-score:

**No corrió completo, se quedó sin paciencia y se detuvo en la época 8.**

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