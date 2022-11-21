# Base JMVAE-NF class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
import wandb
from numpy.random import randint
import numpy as np
from bivae.utils import get_mean, kl_divergence, negative_entropy, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF, my_VAE_MAF, VAE_MAF_Config
from torchnet.dataset import TensorDataset, ResampleDataset
from torch.utils.data import DataLoader
from bivae.utils import extract_rayon
from ..nn import Encoder_VAE_SVHN,Decoder_VAE_SVHN
import matplotlib.pyplot as plt


from bivae.dataloaders import CIRCLES_SQUARES_DL
from ..nn import DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis.classifiers.classifier_empty_full import load_classifier_circles, load_classifier_squares




class JMVAE_NF_CIRCLES(JMVAE_NF):
    def __init__(self, params):
        params.input_dim = (1,32,32)
        

        joint_encoder = DoubleHeadJoint(512, params,params,Encoder_VAE_SVHN,Encoder_VAE_SVHN, params)
        
        if params.no_nf:
            vae = my_VAE
            vae_config = VAEConfig
        else:
            vae = my_VAE_IAF if params.flow=='iaf' else my_VAE_MAF
            vae_config = VAE_IAF_Config if params.flow == 'iaf' else VAE_MAF_Config
        
        flow_config = {'n_made_blocks' : 2} if not params.no_nf else {}
        wandb.config.update(flow_config)
        vae_config = vae_config(params.input_dim, params.latent_dim,**flow_config )

        encoder1, encoder2 = None, None
        decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
             vae(model_config=vae_config, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config, encoder=encoder2, decoder=decoder2)

        ])
        super(JMVAE_NF_CIRCLES, self).__init__(params, joint_encoder, vaes)
        self.modelName = 'jmvae_nf_circles_squares'

        self.vaes[0].modelName = 'squares'
        self.vaes[1].modelName = 'circles'
        self.to_tensor = False

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform=None):
        # handle merging individual datasets appropriately in sub-class
        # load base datasets
        dl = CIRCLES_SQUARES_DL(self.data_path)
        train, test, val = dl.getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def analyse_rayons(self,data, r0, r1, runPath, epoch, filters):
        m,s,zxy = self.analyse_joint_posterior(data,n_samples=len(data[0]))
        zx, zy = self.analyse_uni_posterior(data,n_samples=len(data[0]))
        plot_embeddings_colorbars(zxy,zxy,r0,r1,'{}/embedding_rayon_joint{:03}.png'.format(runPath,epoch), filters)
        wandb.log({'joint_embedding' : wandb.Image('{}/embedding_rayon_joint{:03}.png'.format(runPath,epoch))})
        plot_embeddings_colorbars(zx, zy,r0,r1,'{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch), filters)
        wandb.log({'uni_embedding' : wandb.Image('{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch))})

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        JMVAE_NF.sample_from_conditional(self,data, runPath, epoch,n)
        if epoch == self.max_epochs:
            self.conditional_rdist(data, runPath,epoch)

    def conditional_rdist(self,data,runPath,epoch,n=100):
        bdata = [d[:8] for d in data]
        samples = self._sample_from_conditional(bdata,n)
        samples = torch.cat([torch.stack(samples[0][1]), torch.stack(samples[1][0])], dim=1)
        r = extract_rayon(samples)
        plot_hist(r,'{}/hist_{:03d}.png'.format(runPath, epoch))
        wandb.log({'histograms' : wandb.Image('{}/hist_{:03d}.png'.format(runPath, epoch))})

    def extract_hist_values(self,samples):
        samples = torch.cat([torch.stack(samples[0][1]), torch.stack(samples[1][0])], dim=1)
        return extract_rayon(samples), (0,1), 10

    def compute_metrics(self, data, runPath, epoch, classes=None,freq=10):
        m = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)

        # Compute cross accuracy of generation
        bdata = [d[:100] for d in data]
        samples = self._sample_from_conditional(bdata, n=100)

        preds1 = self.classifier2(torch.stack(samples[0][1]))
        preds0 = self.classifier1(torch.stack(samples[1][0]))

        labels0 = torch.argmax(preds0, dim=-1).reshape(100, 100)
        labels1 = torch.argmax(preds1, dim=-1).reshape(100, 100)
        classes_mul = torch.stack([classes[0][:100] for _ in np.arange(100)]).cuda()
        acc1 = torch.sum(labels1 == classes_mul)/(100*100)
        acc0 = torch.sum(labels0 == classes_mul)/(100*100)

        bdata = [d[:100] for d in data]
        samples = self._sample_from_conditional(bdata, n=100)
        r, range, bins = self.extract_hist_values(samples)
        sm =  {'neg_entropy' : negative_entropy(r.cpu(), range, bins), 'acc0' : acc0, 'acc1' : acc1}
        update_details(sm,m)

        # print('Eval metrics : ', sm)
        return sm

    def set_classifiers(self):

        self.classifier1 = load_classifier_squares()
        self.classifier2 = load_classifier_circles()
    
    
    def visualize_poe(self, data,runPath, n_data=4, N=30):
        
        
        bdata = [torch.cat([d[:n_data]]*N) for d in data]
        
        # Sample from the unimodal posteriors
        u_z = [self.vaes[m].forward(bdata[m]).z.reshape(N, n_data,2).permute(1,0) for m in range(self.mod)]

        
        # Sample from the joint posterior
        j_z = self.forward(bdata)[-1].reshape(N, n_data,2).permute(1,0)
        
        # Sample from the product of expert posterior
        
        poe_z = self.sample_from_poe_subset([0,1], 1,bdata, mcmc_steps=100, n_lf=10, eps_lf=0.01).reshape(N, n_data,2).permute(1,0)
        
        # Plot
        fig, axs = plt.subplots(2,n_data, sharex=True, sharey=True)
        
        for i in range(n_data):
            # On the first row plot the true joint posterior, and unimodal posteriors
            axs[0][i].scatter(u_z[0][i, :,0], u_z[0][i,:,1]) # First modality
            axs[0][i].scatter(u_z[1][i, :,0], u_z[1][i,:,1]) # second modality
            axs[0][i].scatter(j_z[i,:,0], j_z[i,:,1])
            

            # On the second, the poe and the unimodal posteriors
            axs[1][i].scatter(u_z[0][i, :,0], u_z[0][i,:,1]) # First modality
            axs[1][i].scatter(u_z[1][i, :,0], u_z[1][i,:,1]) # second modality
            axs[1][i].scatter(poe_z[i,:,0], poe_z[i,:,1])
        
        fig.savefig('{}/product_of_posteriors.png'.format(runPath))
            