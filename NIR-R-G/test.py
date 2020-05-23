import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
import dataloader
import utils
import argparse
from sklearn.metrics import fbeta_score
import pandas as pd
from dataloader import TestAmazonDataset
from dataloader import AmazonDataset
import dataloader 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="saved model")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--scale", type=int, default=256, help="image scaling")
parser.add_argument("--nocuda", action='store_true', help="no cuda used")
parser.add_argument("--nworkers", type=int, default=0, help="number of workers")
parser.add_argument("--output_file", type=str, default="pred.csv", help="output file")
parser.add_argument('--nir_channel',type = str, default= 'NDVI-calculated',
                    help = 'Representation options: NIR-R-G, NIR-R-B, NDVI-spectral, NDVI-calculated,NDWI')
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda
print('...predicting on cuda: {}'.format(cuda))

# Define transformations
# If using pretrained models normalization should also be added.
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225])
test_transforms = transforms.Compose([transforms.ToTensor()])
#val_transforms = transforms.Compose([transforms.Scale(args.scale),
#                        transforms.ToTensor()])

# Create dataloaders
kwargs = {'pin_memory': True} if cuda else {}
testset = AmazonDataset('csv/sample_submission_v2.csv', '/home/jlcastillo/Database_real/test-tif-v2',
                'csv/labels.txt', args.nir_channel, test_transforms)
test_loader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)

"""
valset = data.PlanetData('data/val_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', val_transforms)
val_loader = DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)
"""

def fscore(prediction):
    """ Get the fscore of the validation set. Gives a good indication
    of score on public leaderboard"""
    target = torch.FloatTensor(0, 17)
    for i, (_,y) in enumerate(val_loader):
        target = torch.cat((target, y), 0)
    fscore = fbeta_score(target.numpy(), prediction.numpy() > 0.23,
                beta=2, average='samples')
    return fscore

def predict(net, loader):
    net.eval()
    predictions = torch.FloatTensor(0, 17)
    for i, (X,_) in enumerate(tqdm(loader,desc='Predicting')):
        if cuda:
            X = X.cuda()
        X = Variable(X, volatile=True)
        output = net(X)
        predictions = torch.cat((predictions, output.cpu().data), 0)
    return predictions

if __name__ == '__main__':
    loaded_model = torch.load(args.model)
    net = model.__dict__[loaded_model['arch']]()
    net.load_state_dict(loaded_model['state_dict'])
    print('...loaded {}'.format(loaded_model['arch']))
    if cuda:
        net = net.cuda()

    # predict on the test set
    y_test = predict(net, test_loader)

    # Ready dataframe for submission.
    labels, _, _ = dataloader.get_labels('csv/labels.txt')
    y_test = y_test.numpy()
    y_test = pd.DataFrame(y_test, columns = labels)

    # Populate the submission csv
    predictions = []
    for i in tqdm(range(y_test.shape[0]),desc='Writing file'):
        a = y_test.ix[[i]]
        a = a.apply(lambda x: x > 0.24, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        predictions.append(' '.join(list(a.index)))
    df_test = pd.read_csv('csv/sample_submission_v2.csv')
    df_test['tags'] = pd.Series(predictions).values
    df_test.to_csv(args.output_file, index=False)
