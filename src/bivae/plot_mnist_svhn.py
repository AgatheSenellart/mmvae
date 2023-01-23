import argparse
import datetime
import json
import random
import sys
from pathlib import Path
import os, glob
from tqdm import tqdm

import numpy as np
import torch
import wandb

import models
from models.samplers import GaussianMixtureSampler
from utils import Logger, Timer, unpack_data, update_dict_list, get_mean_std, print_mean_std
from torchvision.utils import make_grid, save_image

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = 'cuda'
n_samples = 50

wandb.init(project = 'plot_celeba_samples' , entity="asenellart") 


"""A script to plot samples from the different models and saving all the attributes """

models_to_evaluate = ['jean_zay_models/jmvae/mnist_svhn', 'jean_zay_models/jmvae_nf/mnist_svhn']
model_dicts = []

# load args from disk if pretrained model path is given
for model_name in models_to_evaluate:
    print(model_name)
    day_path = max(glob.glob(os.path.join('../experiments/' + model_name, '*/')), key=os.path.getmtime)
    model_path = max(glob.glob(os.path.join(day_path, '*/')), key=os.path.getmtime)
    with open(model_path + 'args.json', 'r') as fcc_file:
        # Load the args
        wandb.init(project = 'plot_celeba_samples' , entity="asenellart") 

        args = argparse.Namespace()
        args.__dict__.update(json.load(fcc_file))
        # Get the model class
        modelC = getattr(models, 'VAE_{}'.format(args.model))
        # Create instance and load the state dict
        model_i = modelC(args).to(device)
        print('Loading model {} from {}'.format(model_i.modelName, model_path))
        model_i.load_state_dict(torch.load(model_path + '/model.pt'))
        # Save everything
        model_dicts.append(dict(args = args,path=model_path, model=model_i))




# Save everything in '../experiments/compare_celeba/'

# set up run path

runPath = Path('../experiments/compare_ms')
runPath.mkdir(parents=True, exist_ok=True)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)


train_loader, test_loader, val_loader = model_dicts[0]['model'].getDataLoaders(batch_size=4, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")


i,j=0,1

def compare_samples():

    # Take the first batch of the test_dataloader as conditional samples
    for n,dataT in enumerate(test_loader):
        data = unpack_data(dataT, device=device)
        classes = dataT[0][1], dataT[1][1]
        # Sample from the conditional distribution for each model 
        samples = []
        for m in model_dicts:
            model = m['model']
            model.eval()
            comp = model.sample_from_conditional(data, runPath, 0, n=5,return_res=True)[i][j] # s[i][j] is a list with n tensors (n_batch, c, w, h)
            samples.append(comp)
            # print(comp.shape)
        samples = torch.cat(samples, dim=2)
        save_image(samples, str(runPath) + f'/samples_{i}_{j}_{n}.png')

    

    return 



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        compare_samples()