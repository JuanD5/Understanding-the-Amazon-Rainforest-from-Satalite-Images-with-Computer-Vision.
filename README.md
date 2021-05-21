# Understanding-the-Amazon-Rainforest-from-Satalite-Images-with-Computer-Vision.

### Computer Vision Kaggle challenge to correctly classify multi-label satelite images

##### J.Castillo, J.D.García, J.F.Suescún, Universidad de Los Andes, Bogotá, Colombia 2020.


## Requerimientos:
-Pytorch: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`

-Spectral: `conda install -c conda-forge spectral`

-Torch summary: ` pip install torchsummary`

-TensorFlow: `conda install -c conda-forge tensorflow`

-Tensorboard_logger: `pip install tensorboard_logger`

 

## 3. Test

Para hacer el test, se corre:

`python -W ignore test.py --model saved-models/AmazonResNet101-run-3.pth.tar --output_file "pred_x.csv" --batch_size 8`

Para hacer la evaluación en Kaggle, se corre:

`kaggle competitions submit -c planet-understanding-the-amazon-from-space -f pred_x.csv -m "Run-x"`
