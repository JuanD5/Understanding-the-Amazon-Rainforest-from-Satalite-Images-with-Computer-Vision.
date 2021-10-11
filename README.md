# Understanding-the-Amazon-Rainforest-from-Satalite-Images-with-Computer-Vision.

### Computer Vision Kaggle challenge of multi-label classification of  satelite images

##### J.Castillo, J.D.García, J.F.Suescún, Universidad de Los Andes, Bogotá, Colombia 2020.


## Requierements:
-Pytorch: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`

-Spectral: `conda install -c conda-forge spectral`

-Torch summary: ` pip install torchsummary`

-TensorFlow: `conda install -c conda-forge tensorflow`

-Tensorboard_logger: `pip install tensorboard_logger`

 

## 3. Test

Run:

`python -W ignore test.py --model saved-models/AmazonResNet101-run-3.pth.tar --output_file "pred_x.csv" --batch_size 8`

For testing in Kaggle:

`kaggle competitions submit -c planet-understanding-the-amazon-from-space -f pred_x.csv -m "Run-x"`
