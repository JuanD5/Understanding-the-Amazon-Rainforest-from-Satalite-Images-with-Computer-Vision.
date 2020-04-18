# Understanding-the-Amazon-Rainforest-from-Satalite-Images-with-Computer-Vision.

### Computer Vision Kaggle challenge to correctly classify multi-label satelite images

##### J.Castillo, J.D.García, J.F.Suescún, Universidad de Los Andes, Bogotá, Colombia 2020.



## Cositas a tener en cuenta:

Github va a almacenar los logfiles, pero no los modelos salvados. Entonces, si en caso de que lleguemos a necesitarlos, yo los tengo en mi usario en la carpeta *save-models*. 

### 1. ¿Cómo leer los archivos de logfile o ver el entrenamiento en tiempo real?

*Se requiere Tensorflow*

Este es un ejemplo en mi carpeta del repositorio. En sus casos, deberían cambiar el nombre del usuario al principio, y cambiar la ruta si quieren usar la copia de su repo, o dejar la ruta si quieren usar la mía (eso no va a afectar, creo):

1. Correr la siguiente línea:    `tensorboard --logdir==jlcastillo@bcv001:/home/jlcastillo/Proyecto/Understanding-the-Amazon-Rainforest-from-Satalite-Images-with-Computer-Vision./logs/AmazonSimpleNet/run-0/ --port=16007`

   Es decir, ahí lo importante es poner la ruta de la carpeta que contiene el experimento que quieren visualizar.

2. En sus computadores, se meten a Chrome y se van a la siguiente dirección: [Tensorboard]( http://bcv001:16007)



### 2. Experimentos

Hay un archivo que se llama **Experiments.MD**.  La idea es registrar aquí todos los detalles de los experimentos: Los parámetros, lo que se usó, lo que se cambió. El código salva los modelos y logfiles en carpetas consecutivas, tipo *run_0, run_1, etc.*, para el logfile, y .tars para los modelos. Lo que yo sugiero es o cambiarle el nombre, o registrar inmediatamente en el Markdown qué experimento corresponde a dicha carpeta. 

Pasa mucho que uno empieza a correr algo y dice: "Pucha, se me olvidó cambiar tal cosa", y detiene el experimento. A veces, se alcanzan a crear las carpetas, y lo mejor es eliminarlas de una vez.  Tener el Markdown siempre actualizado permite precisamente no borrar lo que no se tiene que borrar.

 