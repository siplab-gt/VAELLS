#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import os
import time
import numpy as np
import scipy.io as sio
import scipy as sp

import argparse

from torch.autograd import Variable

import torch.nn as nn
import torch

from transOptModel import TransOpt

from utils import *
from trans_opt_objectives import *

parser = argparse.ArgumentParser()  
parser.add_argument('--model', type=str, default='/storage/home/hcoda1/6/mnorko3/p-crozell3-0/projects/VAELLS/results/natDigitsLL/VAELLS', help='folder name')
parser.add_argument('--epoch', type=int, default=75, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=100, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--z_dim', type=int, default=6, help='Dimension of the latent space')
parser.add_argument('--x_dim', type=int, default=20, help='Dimension of the input space')
parser.add_argument('--c_samp', type=int, default=1, help='Number of samples from the c distribution')
parser.add_argument('--num_anchor', type=int, default=2, help='Number of anchor points per class')
parser.add_argument('--M', type=int, default=4, help='Number of dictionary elements')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--anchor_lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--lr_psi', type=float, default=0.00001, help='learning rate for Psi')
parser.add_argument('--b1', type=float, default=0.5, help='adam: momentum term')
parser.add_argument('--b2', type=float, default=0.999, help='adam: momentum term')
parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight of the reconstruction term of the loss function')
parser.add_argument('--prior_weight', type=float, default=1.0, help='Weight of the prior term of the loss function')
parser.add_argument('--post_TO_weight', type=float, default=1.0, help='Weight of the posterior reconstruction term of the loss function')
parser.add_argument('--post_l1_weight', type=float, default=1.0, help='Weight of the posterior l1  term of the loss function')
parser.add_argument('--prior_l1_weight', type=float, default=0.01, help='Weight of the posterior l1  term of the loss function')
parser.add_argument('--post_cInfer_weight', type=float, default= 0.000001, help='Weight of the prior on the l1 term during inference in the posterior')
parser.add_argument('--prior_cInfer_weight', type=float, default=0.000001, help='Weight of the prior on the l1 term during inference in the prior')
parser.add_argument('--gamma', type=float, default=0.01, help='gd: weight on dictionary element')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--zeta', type=float, default=0.001, help='gd: weight on coefficient regularizer')
parser.add_argument('--binarize_flag',type = bool,default = False, help='flag of weather or not to binarize the image data')
parser.add_argument('--num_net_steps', type=int, default=20, help='Number of steps on only the network weights')
parser.add_argument('--num_psi_steps', type=int, default=60, help='Number of steps on only the psi weights')
parser.add_argument('--numRestart', type=int, default=1, help='number of restarts for coefficient inference')
parser.add_argument('--priorWeight_nsteps', type=float, default= 0.0001, help='Weight of the prior during the network weight update steps')
parser.add_argument('--netWeights_psteps', type=float, default=0.0001, help='Weight of the reconstruction loss during the transport operator weight update steps')
parser.add_argument('--to_noise_std', type=float, default=0.001, help='Noise for sampling gaussian noise in latent space')
parser.add_argument('--num_pretrain_steps', type=int, default=30000, help='Number of steps to train the network with no VAE component')
parser.add_argument('--data_use', type=str,default = 'natDigits',help='Specify which dataset to use [swiss2D_identity,swiss2D,rotDigits,natDigits]')
parser.add_argument('--alternate_steps_flag', type=int, default=1, help='[0/1] to specify whether to alternate between steps updating net weights and psi weights ')
parser.add_argument('--closest_anchor_flag', type=int, default=1, help='[0/1] to specify whether to alternate between steps updating net weights and psi weights ')
parser.add_argument('--stepUse', type=int, default=33500, help='Trainign step from the saved file that you want to run')
parser.add_argument('--startPt', type=int, default=0, help='Starting index of test data to compute metrics for')
parser.add_argument('--num_samp', type=int, default=100, help='Number of latent vectors sampled for each test point')
parser.add_argument('--numTestPts', type=int, default=250, help='Number of tets points to compute metric')
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

# Define variables from input parameters
batch_size = opt.batch_size
input_h = opt.img_size
input_w = opt.img_size
x_dim = opt.x_dim
z_dim = opt.z_dim
N = opt.z_dim
N_use = N*N
M = opt.M
zeta = opt.zeta
lr_psi = opt.lr_psi
num_c_samp  = opt.c_samp
num_anchor = opt.num_anchor
num_pretrain_steps =opt.num_pretrain_steps
data_use = opt.data_use
prior_l1_weight = opt.prior_l1_weight*opt.priorWeight_nsteps
prior_weight = opt.prior_weight*opt.priorWeight_nsteps
recon_weight = opt.recon_weight
numRestart = opt.numRestart
num_net_steps = opt.num_net_steps
num_psi_steps = opt.num_psi_steps
alternate_steps_flag = opt.alternate_steps_flag
lr_anchor = opt.anchor_lr

scale = 10.0 # This doesn't do anything at this point but I need to get rid of the scaling parameters

# Define decay in psi learning rate with unsuccessful steps or after a certain number of steps
decay = 0.99            # Decay after psi steps that increase the objective
titrate_decay = 0.9992  # Titration decay
titrate_steps = num_pretrain_steps + 15000    # number of steps after which titration decay occurs
max_psi_lr = 0.008      # max psi learning rate allowed

# Specify the sampling spread and variance in the pretrained steps
#post_l1_weight = 100.0
#to_noise_std = 0.0

# Parameters for finding the closest anchor points
closest_anchor_flag = opt.closest_anchor_flag
t_use = np.arange(0,1.0,0.01)

# Initialize the counts for alternating steps
net_count = 0
psi_count = num_psi_steps +1

# Specify which classes we want to train on
class_use = np.array([0,1,2,3,4,5,6,7,8,9])
class_use_str = np.array2string(class_use)    


test_size = batch_size
batch_orig = 32

if data_use == 'rotDigits':
    from test_metrics_MNIST_rotDigit import *
elif data_use == 'natDigits':
    from test_metrics_MNIST_natDigit_singleAnchor import *

# May need to adjust the structure of these saving paths depending on the data you want to 
if alternate_steps_flag == 0:
    save_folder = opt.model +  '_' + data_use + '_pre' + str(num_pretrain_steps) + '_CA' + str(closest_anchor_flag) + '_M' + str(M) + '_z' + str(z_dim) + '_A' + str(opt.num_anchor)  + '_batch' + str(opt.batch_size) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '_lr' + str(opt.lr)  +'_toN' + str(opt.to_noise_std) + '/'
else:
    save_folder = opt.model + '_'+ data_use + '_pre' + str(num_pretrain_steps) + '_CA' + str(closest_anchor_flag) + '_M' + str(M) + '_z' + str(z_dim) + '_A' + str(opt.num_anchor)  + '_batch' + str(opt.batch_size) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '_lr' + str(opt.lr) + '_nst' + str(num_net_steps) + 'pst' + str(num_psi_steps) +  '_toN' + str(opt.to_noise_std) + '/'


if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
start_time = time.time()
counter = 1

mse_loss = torch.nn.MSELoss(reduction = 'mean')
mse_loss_sum = torch.nn.MSELoss(reduction = 'sum')
latent_mse_loss = torch.nn.MSELoss(reduction = 'sum')

from covNetModel import Encoder
from covNetModel import Decoder

num_samp = opt.num_samp 
encoder = Encoder(z_dim,opt.c_dim,opt.img_size)
decoder = Decoder(z_dim,opt.c_dim,opt.img_size)

transNet = TransOpt()
sampler_c = Sample_c()
# Load a pretrained file - make sure the model parameters fit the parameters of rht pre-trained models
if data_use == 'rotDigits':
    checkpoint = torch.load('./pretrained_models/rotMNIST/network_batch32_zdim10_M' + str(opt.M) + '_A' + str(opt.num_anchor) + '.pt')
elif data_use == 'natDigits':
    checkpoint = torch.load('./pretrained_models/natMNIST/network_M' + str(opt.M) + '_A' + str(opt.num_anchor) + '_toN' + str(opt.to_noise_std) + '.pt')
# Or specify a network that's been trained independently
#checkpoint = torch.load(save_folder + 'network_batch' + str(batch_orig) + '_zdim' + str(opt.z_dim) +  '_step' + str(opt.stepUse) + '.pt')
encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
decoder.load_state_dict(checkpoint['model_state_dict_decoder'])


sample_X,sample_labels,_ = load_mnist("test")


Psi = Variable(torch.mul(torch.randn(N_use, M, dtype=torch.double),0.01), requires_grad=True)
Psi = checkpoint['Psi']
Psi_use = Psi.detach().numpy()

if data_use == 'natDigits':
    anchors = checkpoint['anchors']

numTestPts = opt.numTestPts

sample_X = sample_X[opt.startPt:opt.startPt+numTestPts]
sample_X_torch = torch.from_numpy(sample_X)
sample_X_torch =sample_X_torch.permute(0,3,1,2)
sample_X_torch =sample_X_torch.float()

if data_use == 'natDigits':
    LL,LL_detail,LL_no_add,LL_detail_no_add = log_likelihood(encoder,decoder,transNet,sampler_c,Psi,sample_X_torch,sample_labels[opt.startPt:opt.startPt+numTestPts],anchors,opt.to_noise_std,num_anchor,M,numRestart,scale,opt,save_folder,num_samp)
    MSE,ELBO = test_metrics(encoder,decoder,transNet,sampler_c,Psi,sample_X_torch,sample_labels[opt.startPt:opt.startPt+numTestPts],anchors,opt.to_noise_std,num_anchor,M,1,opt,mse_loss_sum,latent_mse_loss,scale)
elif data_use == 'rotDigits':
    LL,LL_detail,LL_no_add,LL_detail_no_add = log_likelihood(encoder,decoder,transNet,sampler_c,Psi,sample_X_torch,sample_labels[0:numTestPts],opt.to_noise_std,num_anchor,M,numRestart,scale,opt,save_folder,num_samp)
    MSE,ELBO = test_metrics(encoder,decoder,transNet,sampler_c,Psi,sample_X,sample_labels,opt.to_noise_std,num_anchor,M,1,opt,mse_loss_sum,latent_mse_loss,scale)
    
sio.savemat(save_folder + 'LLMetrics_singleAnc_batch' + str(opt.batch_size) + '_' + str(num_samp) + 'samp_startPt' + str(opt.startPt) + '_step' + str(opt.stepUse) + '.mat',{'LL':LL,'LL_detail':LL_detail,'LL_no_add':LL_no_add,'LL_detail_no_add':LL_detail_no_add});
sio.savemat(save_folder + 'MSELEBOMetrics_singleAnc_batch' + str(opt.batch_size) + '_' + str(num_samp) + 'samp_step' + str(opt.stepUse) + '.mat',{'MSE':MSE,'ELBO':ELBO});

