#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
import time
import numpy as np
import scipy.io as sio
import os

import argparse


from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR

import torch.nn as nn
import torch

from transOptModel import TransOpt


from utils import *
from trans_opt_objectives import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./results/TOVAE', help='folder name')
parser.add_argument('--epoch', type=int, default=15, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--z_dim', type=int, default=2, help='Dimension of the latent space')
parser.add_argument('--x_dim', type=int, default=20, help='Dimension of the input space')
parser.add_argument('--c_samp', type=int, default=1, help='Number of samples from the c distribution')
parser.add_argument('--num_anchor', type=int, default=3, help='Number of anchor points per class')
parser.add_argument('--M', type=int, default=4, help='Number of dictionary elements')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--anchor_lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--lr_psi', type=float, default=0.00005, help='learning rate for Psi')
parser.add_argument('--b1', type=float, default=0.5, help='adam: momentum term')
parser.add_argument('--b2', type=float, default=0.999, help='adam: momentum term')
parser.add_argument('--recon_weight', type=float, default=0.01, help='Weight of the reconstruction term of the loss function')
parser.add_argument('--prior_weight', type=float, default=1.0, help='Weight of the prior term of the loss function')
parser.add_argument('--post_TO_weight', type=float, default=1.0, help='Weight of the posterior reconstruction term of the loss function')
parser.add_argument('--post_l1_weight', type=float, default=1.0, help='Weight of the posterior l1  term of the loss function')
parser.add_argument('--prior_l1_weight', type=float, default=0.01, help='Weight of the posterior l1  term of the loss function')
parser.add_argument('--post_cInfer_weight', type=float, default= 0.000001, help='Weight of the prior on the l1 term during inference in the posterior')
parser.add_argument('--prior_cInfer_weight', type=float, default=0.000005, help='Weight of the prior on the l1 term during inference in the prior')
parser.add_argument('--gamma', type=float, default=0.01, help='gd: weight on dictionary element')
parser.add_argument('--zeta', type=float, default=0.001, help='gd: weight on coefficient regularizer')
opt = parser.parse_args()
print(opt)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

    
class Sample_c(nn.Module):
    # Sample c from a laplacian distribution
    def __init__(self):
        super(Sample_c, self).__init__()
       
    def forward(self, batch_size,M,zeta):
        u = torch.rand(batch_size,M)-0.5
        c = -torch.div(torch.mul(torch.sign(u),torch.log(1.0-2.0*torch.abs(u))),zeta)
        return c


batch_size = opt.batch_size
x_dim = opt.x_dim
z_dim = opt.z_dim
N = opt.z_dim
N_use = N*N
M = opt.M
kappa = 8
scale = 1.0
decay = 0.99
titrate_decay = 0.9992
zeta = opt.zeta
lr_psi = opt.lr_psi
to_noise_std_later = 0.001
num_c_samp  = opt.c_samp
num_anchor = opt.num_anchor
num_class = 1
weight_scale = 1.0
num_pretrain_steps =0
data_use = 'concen_circle'
closest_anchor_flag = 1
alternate_steps_flag = 1
post_l1_weight = 100.0
to_noise_std = 0.001
t_use = np.arange(0,1.0,0.01)
numRestart = 4

learn_anchor_flag = 0
test_size = batch_size
lr_anchor = opt.anchor_lr
prior_mult = np.power(2,-(z_dim/2.0+M))*np.power(np.pi,-z_dim/2.0)
stepUse = 3900
batch_orig = 30

np.seterr(all='ignore')
save_folder = opt.model + '_' + data_use + '_M' + str(M) + '_A' + str(opt.num_anchor) + '_Nc' + str(opt.c_samp) + '_batch' + str(batch_orig) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
M = 1
start_time = time.time()
counter = 0


matLoad = sio.loadmat('./pretrained_models/concen_circle/mapMat_circleHighDim_nonLinear_z2_x20.mat')
mapMat = matLoad['mapMat']

opt.post_TO_weight = opt.post_TO_weight/weight_scale
opt.prior_weight = opt.prior_weight/weight_scale
opt.post_l1_weight = opt.post_l1_weight/weight_scale
opt.prior_l1_weight = opt.prior_l1_weight/weight_scale


nTrain = 1000
noise_std = 0.01

from fullyConnectedModel import Encoder
from fullyConnectedModel import Decoder

encoder = Encoder(x_dim,z_dim,M)
decoder = Decoder(x_dim,z_dim)

transNet = TransOpt()
sampler_c = Sample_c()



# Load a pretrained file - make sure the model parameters fit the parameters of rht pre-trained models
checkpoint = torch.load('./pretrained_models/concen_circle/network_batch30_zdim2.pt')
# Or specify a network that's been trained independently
#checkpoint = torch.load(save_folder + 'network_batch' + str(batch_orig) + '_zdim' + str(z_dim) + '_zeta' + str(zeta) + '_step' + str(stepUse) + '.pt')

encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
decoder.load_state_dict(checkpoint['model_state_dict_decoder'])

psi_load = sio.loadmat('./pretrained_models/concen_circle/spreadInferenceTest.mat')
Psi_read = psi_load['Psi_new']
Psi_read = np.expand_dims(Psi_read[:,0],axis=1)
anchors_read = psi_load['anchors']
# Define the loss functions
mse_loss = torch.nn.MSELoss(reduction = 'mean')
mse_loss_sum = torch.nn.MSELoss(reduction = 'sum')
latent_mse_loss = torch.nn.MSELoss(reduction = 'sum')




## Load dictionary
normalize_val = 1.0
Psi = Variable(torch.mul(torch.randn(N_use, M, dtype=torch.double),0.01), requires_grad=True)
Psi.data = torch.from_numpy(Psi_read)
#Psi = checkpoint['Psi']
Psi_use = Psi.detach().numpy()



anchors = Variable(torch.randn(num_anchor*num_class,x_dim), requires_grad=True)
anchors.data = torch.from_numpy(anchors_read)
#anchors = checkpoint['anchors']

nTest = 150
sample_X,sample_orig,sample_labels = create_circle_data(nTest//2,noise_std,mapMat,np.array([0.5,1]))

sample_X_torch = torch.from_numpy(sample_X)
sample_X_torch = sample_X_torch.float()

z_mu = encoder(sample_X_torch)
z_coeff = sampler_c(nTest,M,post_l1_weight)
z = transNet(z_mu.double(),z_coeff.double(),Psi,to_noise_std)
z_np= z.detach().numpy()




numPtPairs = 10
t_int_gen = 0.01
t_gen_vals = np.arange(0.0,1.0,t_int_gen)
z_path_all = np.zeros((numPtPairs,len(t_gen_vals),z_dim))
z0_path_store = np.zeros((numPtPairs,z_dim))
z1_path_store = np.zeros((numPtPairs,z_dim))
z0Idx_store = np.zeros((numPtPairs))
z1Idx_store = np.zeros((numPtPairs))
for k in range(0,numPtPairs):
    
    z0Idx = int(np.random.uniform(0,nTest))   
    idxUse = np.where((sample_labels == sample_labels[z0Idx]))[0]
    z0Idx_store[k] = z0Idx
    z1TempIdx = int(np.random.uniform(0,len(idxUse))) 
    z1Idx = idxUse[z1TempIdx]
    z1Idx_store[k] = z1Idx
    z0 = z_np[z0Idx,:]
    z0_path_store[k,:] = z0
    z1 = z_np[z1Idx,:]
    z1_path_store[k,:] = z1
    E_single = np.zeros((numRestart))     
    c_est_store = np.zeros((numRestart,M))
    nit_restart = np.zeros((numRestart))
    for r_idx in range(0,numRestart):
        rangeMin = -200 + r_idx*100
        rangeMax = rangeMin + 100
        c_est_store[r_idx,:],E_single[r_idx],nit_restart[r_idx] = infer_transOpt_coeff(z0,z1,Psi_use.astype('double'),0.0,rangeMin,rangeMax)
    minIdx = np.argmin(E_single)
    c_est = c_est_store[minIdx,:]
    
    t_count = 0
    for t_use in t_gen_vals:
        coeff_use = c_est*t_use
        coeff_input = np.expand_dims(coeff_use,axis=0)
        latent_0 = np.expand_dims(z0,axis=0)
        z_est = transNet(torch.from_numpy(latent_0),torch.from_numpy(coeff_input),Psi,to_noise_std)
        z_est_np = z_est.detach().numpy()
        z_path_all[k,t_count,:] = z_est_np
        t_count = t_count +1
        

a_mu = encoder(anchors)
a_mu_np = a_mu.detach().numpy()
num_c_samp = 100
z_anchor_samp = np.zeros((num_c_samp,num_anchor*2,z_dim))
for k in range(0,num_c_samp):
    z_coeff_anchor = sampler_c(num_anchor*2,M,post_l1_weight*0.001)
    z_scale_anchor = transNet(a_mu.double(),z_coeff_anchor.double(),Psi,to_noise_std*31.622)
    z_anchor_samp[k,:,:] = z_scale_anchor.detach().numpy()
sio.savemat(save_folder + '/circleDataTests.mat',{'sample_X':sample_X,'sample_labels':sample_labels,'z_np':z_np,'sample_orig':sample_orig,
                                                  'z0_path_store':z0_path_store,'z1_path_store':z1_path_store,'z_path_all':z_path_all,
                                                  'anchors':a_mu_np,'z_anchor_samp':z_anchor_samp,'z0Idx_store':z0Idx_store,'z1Idx_store':z1Idx_store});





