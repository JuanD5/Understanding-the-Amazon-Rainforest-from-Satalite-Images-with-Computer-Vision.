import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import model
import dataloader
import utils
import argparse
from sklearn.metrics import fbeta_score
import pandas as pd
from dataloader import AmazonDataset, AmazonDatasetNIR
from tqdm import tqdm
import numpy as np


#Arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument("--path", type=str, default = '/home/jlcastillo/Database_real/val',
                    help="If the code doesn't run due to the dataset location, you can use this argument to fix it")
parser.add_argument("--nocuda", action='store_true', help="No cuda used")
args = parser.parse_args()


#Other arguments:
path = args.path #General path
labels_file = os.path.join('csv', 'labels.txt')
test_path = path #Path to perform prediction (test real).
test_submission_file = os.path.join('csv','sample_submission_v2.csv') 
val_path = os.path.join(path,'val') #Path to perform test (requirements of the project) over validation test, and demo.
val_file = os.path.join('csv', 'val.csv')

batch_size = args.batch_size

#Model:
best_model = args.model

cuda = not args.nocuda and torch.cuda.is_available() # use cuda
print('...predicting on cuda: {}'.format(cuda))


#Transforms:
test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

kwargs = {'pin_memory': True} if cuda else {}
#Dataset and dataloader

test_set = AmazonDataset('csv/val.csv',test_path,
                labels_file,'normal', test_transforms)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=0, **kwargs)


def predict(net, loader,description):
    net.eval()
    predictions = torch.FloatTensor(0, 17)
    for i, (X,_) in enumerate(tqdm(loader,desc=description)):
        if cuda:
            X = X.cuda()
        X = Variable(X, volatile=True)
        output = net(X)
        predictions = torch.cat((predictions, output.cpu().data), 0)
    return predictions

def fscore(prediction):
    """ Get the fscore of the validation set. Gives a good indication
    of score on public leaderboard"""
    target = torch.FloatTensor(0, 17)
    for i, (_,y) in enumerate(tqdm(test_loader,desc='Calculating F2-Score')):
        target = torch.cat((target, y), 0)
    fscore = fbeta_score(target.numpy(), prediction.numpy() > 0.23,
                beta=2, average='samples')
    return fscore

if __name__ == '__main__':
    loaded_model = torch.load(best_model)
    net = model.__dict__[loaded_model['arch']]()
    net.load_state_dict(loaded_model['state_dict'])
    print('...loaded {}'.format(loaded_model['arch']))
    if cuda:
        net = net.cuda()
    # predict on the val set

    test_results = predict(net, test_loader,'Val')
    ev = fscore(test_results)
    print('F-score = {}'.format(ev))