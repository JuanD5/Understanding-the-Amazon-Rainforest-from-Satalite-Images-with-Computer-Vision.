# Experimentos:

<a name="up"></a>

1. [Baseline](#run1): Nuestro baseline. **Evaluado.**
2. [Baseline ResNet](#run2): Nuestro baseline con pesos pre-entrenados de ResNet-18. **Corrió hasta la época 8.**
3. [Baseline ResNet](#run3): Nuestro baseline con pesos pre-entrenados de ResNet-18. Este sí corrió completo. **Evaluado.**
4. [Baseline ResNet en 40](#run4): Nuestro baseline con pesos pre-entrenados de ResNet-18 en 40 épocas. Evaluado.**
5. [ResNet101](#run5): Modelo pre-entrenado. **Evaluado.**
6. [ResNet101](#run6): Modelo pre-entrenado, sin validar, en más épocas (40). **Evaluado.**
7. [ResNet18 NDVI](#run7): Modelo pre-entrenado, validando y usando infrarrojo.  **Evaluado.**
8. [ResNet18 NIR-R-G](#run8) : Modelo pre-entrenado, validando y usando infrarrojo NIR-R-G.  **Evaluado.**
9. [ResNet18 NIR-R-B](#run9) : Modelo pre-entrenado, validando y usando infrarrojo NIR-R-B.  **Evaluado.**
10. [ResNet18 NIR-R-B en 40 épocas](#run10): Modelo pre-entrenado, validando y usando infrarrojo NIR-R-B.  **Corrió hasta la época 7.**
11. [ResNet18, RGB + NIR normal](#run11): Modelo pre-entrenado, validando y usando infrarrojo normal.  **Evaluado.**
12. [Inception V3](#run12): Modelo pre-entrenado en Inception usando jpgs. **Evaluado.**
13. [Inception V3 completo](#run13): Modelo pre-entrenado en Inception usando jpgs. **Evaluado.**

### - [Baseline](#up):

<a name="run1"></a>

Nombre del modelo: `AmazonSimpleNet-run-0.pth.tar`

Nombre de la carpeta del logfile: `AmazonSimpleNet/run-0`

Tiempo aproximado: 18 minutos/época. 

F-score: **Oficial:** 0.86607

**Parámetros:**

+ --model MODEL]  AmazonSimpleNet
+ [--batch-size N]  2
+ [--test-batch-size N] 128
+ [--epochs N] 20
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
+ [--epochs N] 20
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

Archivo de predicción: `pred_resnet18.cvs`

F-score: **Oficial:** 0.88836

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  2
+ [--test-batch-size N] 128
+ [--epochs N] 20
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

### - [Baseline ResNet18 pre-entrenado en 40 épocas](#up)

<a name="run3"></a>

Nombre del modelo: `AmazonResNet18-run-4.pth`

Nombre de la carpeta del logfile: `AmazonResNet18/run-4`

Tiempo aproximado: 18 minutos/época

Archivo de predicción: `pred_resnet18-40.cvs`

F-score: **Oficial:** 0.89848

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 40
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
+ [--epochs N] 20
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

### - [ResNet101 pre-entrenado, sin validar](#up)

<a name="run6"></a>

Base de datos de train completa, sin realizar validación.

Nombre del modelo: `AmazonResNet101-run-2.pth.tar`

Nombre de la carpeta del logfile: `AmazonResNet101/run-2`

Archivo de predicción: `pred_40epochs.cvs`

F-score: **Oficial:** 0.89468

**Parámetros:**

+ --model MODEL] AmazonResNet101
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 40
+ [--lr LR]  0.01
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 15
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt

### - [ResNet18 pre-entrenado, validando, usando infrarrojo](#up)

<a name="run7"></a>

Base de datos de train (80%) y Validación (20%). Canal de infrarrojo: NDVI-spectral.

Nombre del modelo: `AmazonResNet18-run-0.pth.tar`

Nombre de la carpeta del logfile: `AmazonResNet18/run-0`

Archivo de predicción: `pred_resnet18inf.csv`

F-score: **Oficial:** 0.78410

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  4
+ [--test-batch-size N] 128
+ [--epochs N] 20
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: NDVI-spectral

### - [ResNet18 pre-entrenado, validando, usando infrarrojo NIR-R-G](#up)

<a name="run8"></a>

Base de datos de train (80%) y Validación (20%). Canal de infrarrojo: NIR-R-G.

Nombre del modelo: `AmazonNIRResNet18-run-0.pth.tar`

Nombre de la carpeta del logfile: `AmazonNIRResNet18/run-0`

Archivo de predicción: `pred_resnet18nir.csv`

F-score: **Oficial:**  0.87581

**Parámetros:**

+ --model MODEL] AmazonNIRResNet18
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 20
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: NIR-R-G

### - [ResNet18 pre-entrenado, validando, usando infrarrojo NIR-R-B](#up)

<a name="run9"></a>

Base de datos de train (80%) y Validación (20%). Canal de infrarrojo: NIR-R-G.

Nombre del modelo: `AmazonNIRResNet18-run-1.pth.tar`

Nombre de la carpeta del logfile: `AmazonNIRResNet18/run-1`

Archivo de predicción: `pred_resnet18nir2.csv`

F-score: **Oficial:** 0.87836

**Parámetros:**

+ --model MODEL] AmazonNIRResNet18
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 20
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: NIR-R-G

### - [ResNet18 pre-entrenado en 40 épocas, validando, usando infrarrojo NIR-R-B](#up)

<a name="run10"></a>

Base de datos de train (80%) y Validación (20%). Canal de infrarrojo: NIR-R-G.

Nombre del modelo: `AmazonNIRResNet18-run-2.pth.tar`

Nombre de la carpeta del logfile: `AmazonNIRResNet18/run-2`

Archivo de predicción: `pred_resnet18nir22.csv`

F-score: **Oficial:** 

**Parámetros:**

+ --model MODEL] AmazonNIRResNet18
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 40
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: NIR-R-G

### [ResNet18 pre-entrenado, validando, usando infrarrojo normal](#up)

<a name="run11"></a>

Base de datos de train (80%) y Validación (20%). Canal de infrarrojo: Normal.

Nombre del modelo: `AmazonResNet18-run-3.pth.tar`

Nombre de la carpeta del logfile: `AmazonResNet18/run-3`

Archivo de predicción: `pred_NIRnormal.csv`

F-score: **Oficial:** 0.61550

**Parámetros:**

+ --model MODEL] AmazonResNet18
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 20
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: Normal

### [Inception V3: Modelo pre-entrenado en Inception usando jpgs. ](#up)

<a name="run12"></a>

Base de datos de train (80%) y Validación (20%). Volvemos a las jpg.

Nombre del modelo: `AmazonInceptionV3-run-0.pth.tar`

Nombre de la carpeta del logfile: `AmazonInceptionV3/run-0`

Archivo de predicción: `pred_inception.csv`

F-score: **Oficial:** 0.90601

**Parámetros:**

+ --model MODEL] AmazonInceptionV3
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 20
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: Normal

### [Inception V3 completo: Modelo pre-entrenado en Inception usando jpgs. ](#up)

<a name="run13"></a>

Base de datos de train (80%) y Validación (20%). Volvemos a las jpg.

Nombre del modelo: `AmazonInceptionV3-run-1.pth.tar`

Nombre de la carpeta del logfile: `AmazonInceptionV3/run-1`

Archivo de predicción: `pred_Inception40.csv`

F-score: **Oficial:** 0.89549

**Parámetros:**

+ --model MODEL] AmazonInceptionV3
+ [--batch-size N]  8
+ [--test-batch-size N] 128
+ [--epochs N] 20
+ [--lr LR]  0.001
+ *[-v]*
+ [--momentum M] 0.5
+ [--gamma M] 2
+ [--patience PATIENCE] 20
+ [--input_size N] 256 
+ *[--no_cuda]*
+ [--seed S] 1
+ [--log-interval N] 10
+ [--save SAVE] model.pt
+ --NVDI channel: Normal