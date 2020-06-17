#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import numpy as np
from six.moves import xrange
import scipy.io as sio


import argparse
import logging

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
#from test_metrics import *
from trans_opt_objectives import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./results/TOVAE', help='folder name')
parser.add_argument('--epoch', type=int, default=75, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--z_dim', type=int, default=6, help='Dimension of the latent space')
parser.add_argument('--x_dim', type=int, default=20, help='Dimension of the input space')
parser.add_argument('--c_samp', type=int, default=1, help='Number of samples from the coefficient distribution')
parser.add_argument('--num_anchor', type=int, default=8, help='Number of anchor points per class')
parser.add_argument('--M', type=int, default=4, help='Number of dictionary elements')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate for network weight updates')
parser.add_argument('--anchor_lr', type=float, default=0.0001, help='adam: learning rate for anthor point updates')
parser.add_argument('--lr_psi', type=float, default=0.00001, help='learning rate for Psi')
parser.add_argument('--b1', type=float, default=0.5, help='adam: momentum term')
parser.add_argument('--b2', type=float, default=0.999, help='adam: momentum term')
parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight of the reconstruction term of the loss function (zeta_1)')
parser.add_argument('--post_TO_weight', type=float, default=1.0, help='Weight of the posterior reconstruction term of the loss function (zeta_2)')
parser.add_argument('--post_l1_weight', type=float, default=1.0, help='Weight of the posterior l1  term of the loss function (zeta_3)')
parser.add_argument('--prior_weight', type=float, default=1.0, help='Weight of the prior reconstruction term of the loss function (zeta_4)')
parser.add_argument('--prior_l1_weight', type=float, default=0.01, help='Weight of the prior l1  term of the loss function (zeta_5)')
parser.add_argument('--post_cInfer_weight', type=float, default= 0.000001, help='Weight of the prior on the l1 term during inference in the posterior')
parser.add_argument('--prior_cInfer_weight', type=float, default=0.000001, help='Weight of the prior on the l1 term during inference in the prior')
parser.add_argument('--gamma', type=float, default=0.01, help='Weight on transport operator dictionary regularizer')
parser.add_argument('--img_size', type=int, default=28, help='Image dimension')
parser.add_argument('--num_net_steps', type=int, default=20, help='Number of steps on only the network and anchor weights')
parser.add_argument('--num_psi_steps', type=int, default=60, help='Number of steps on only the psi weights')
parser.add_argument('--priorWeight_nsteps', type=float, default= 0.0001, help='Weight of the prior term during the network weight update steps')
parser.add_argument('--netWeights_psteps', type=float, default=0.0001, help='Weight of the reconstruction loss during the transport operator weight update steps')
parser.add_argument('--to_noise_std', type=float, default=0.001, help='Noise for sampling gaussian noise in latent space')
parser.add_argument('--num_pretrain_steps', type=int, default=30000, help='Number of warm up steps for the network weights')
parser.add_argument('--data_use', type=str,default = 'natDigits',help='Specify which dataset to use [concen_circle,swiss2D,rotDigits,natDigits]')
parser.add_argument('--alternate_steps_flag', type=int, default=1, help='[0/1] to specify whether to alternate between steps updating net weights and psi weights ')
parser.add_argument('--closest_anchor_flag', type=int, default=1, help='[0/1] to to only add the error to the closest anchor point to the objective ')
parser.add_argument('--numRestart', type=int, default=1, help='number of restarts for coefficient inference')
parser.add_argument('--coeffRandStart', type=float, default=-2.5, help='sStarting point for selecting range of random restarts for coefficients')
parser.add_argument('--coeffRandAdd', type=float, default=5.0, help='Range of random restarts for coefficients')

opt = parser.parse_args()
print(opt)



def weights_init_normal(m):
    # Initialize the weights for the networks
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.05)
        torch.nn.init.constant_(m.bias.data, 0.0)

    
class Sample_c(nn.Module):
    # Sample c from a Laplace distribution with a specified spread
    def __init__(self):
        super(Sample_c, self).__init__()
       
    def forward(self, batch_size,M,zeta):
        """
        Convert uniform random variable to Laplace random variable
        
        Inputs:
            - batch_size:   batch size training run
            - M:            Number of transport operator dictionary elements
            - zeta:         Spread parameter for Laplace distribution
        
        Outputs:
            - c:            Vector of sampled Laplace random variables [batch_sizexM]
        """
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

# Parameter for scaling the latent space to accommodate coefficient inference
scale = 10.0 

# Define decay in psi learning rate with unsuccessful steps or after a certain number of steps
decay = 0.99                                    # Decay after psi steps that increase the objective
max_psi_lr = 0.008                              # Max psi learning rate allowed, after this learning rate is reached, the learning rate plateaus for every successful step
titrate_steps = num_pretrain_steps + 15000      # Number of steps after which titration decay occurs
titrate_decay = 0.9992                          # Decay rate after titrate_steps is reached, this encourages fine settling to a final state

# Specify the sampling spread and variance in the pretrained steps
post_l1_weight = 100.0
to_noise_std = 0.0

# Parameters for finding the closest anchor points during the computation of the VAELLS objective
closest_anchor_flag = opt.closest_anchor_flag
t_use = np.arange(0,1.0,0.01)

# Initialize the counts for alternating steps
net_count = 0
psi_count = num_psi_steps +1

# Specify which classes to train on
class_use = np.array([0,1,2,3,4,5,6,7,8,9])
class_use_str = np.array2string(class_use)    

test_size = batch_size
# flag to determine whether or not you want to load the file from a checkpoint
load_checkpoint = 0

np.seterr(all='ignore')

# Define save directories
if alternate_steps_flag == 0:
    save_folder = opt.model + '_vAN' + str(opt.priorWeight_nsteps) + '_vAT' + str(opt.netWeights_psteps) + '_' + str(numRestart) + 'start_' + data_use + '_pre' + str(num_pretrain_steps) + '_CA' + str(closest_anchor_flag) + '_M' + str(M) + '_z' + str(z_dim) + '_A' + str(opt.num_anchor)  + '_batch' + str(opt.batch_size) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '_lr' + str(opt.lr)  +'/'
    sample_dir = opt.model +  '_vAN' + str(opt.priorWeight_nsteps) + '_vAT' + str(opt.netWeights_psteps) + '_' + str(numRestart) + 'start_' + data_use + '_pre' + str(num_pretrain_steps) + '_CA' + str(closest_anchor_flag) + '_M' + str(M) + '_z' + str(z_dim) + '_A' + str(opt.num_anchor) + '_batch' + str(opt.batch_size) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '_lr' + str(opt.lr) +'_samples/'
else:
    save_folder = opt.model +  '_vAN' + str(opt.priorWeight_nsteps) + '_vAT' + str(opt.netWeights_psteps) + '_' + str(numRestart) + 'start_' + data_use + '_pre' + str(num_pretrain_steps) + '_CA' + str(closest_anchor_flag) + '_M' + str(M) + '_z' + str(z_dim) + '_A' + str(opt.num_anchor)  + '_batch' + str(opt.batch_size) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '_lr' + str(opt.lr) + '_nst' + str(num_net_steps) + 'pst' + str(num_psi_steps) +  '/'
    sample_dir= opt.model +  '_vAN' + str(opt.priorWeight_nsteps) + '_vAT' + str(opt.netWeights_psteps) + '_' + str(numRestart) + 'start_' + data_use + '_pre' + str(num_pretrain_steps) + '_CA' + str(closest_anchor_flag) + '_M' + str(M) + '_z' + str(z_dim) + '_A' + str(opt.num_anchor)  + '_batch' + str(opt.batch_size) + '_rw' + str(opt.recon_weight) + '_pol1' + str(opt.post_l1_weight) + '_poR' + str(opt.post_TO_weight)+ '_poC' + str(opt.post_cInfer_weight) + '_prl1' + str(opt.prior_l1_weight) + '_prR' + str(opt.prior_weight) +  '_prC' + str(opt.prior_cInfer_weight)  + '_g' + str(opt.gamma) + '_lr' + str(opt.lr) + '_nst' + str(num_net_steps) + 'pst' + str(num_psi_steps) + '_samples/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Start log file
logging.basicConfig(filename=save_folder + 'output.log',level = logging.DEBUG)

# Initialize the networks and datasets
if data_use == 'concen_circle':
    # Load data for concentric circle dataset
    nTrain = 400
    noise_std = 0.01
    learn_anchor_flag = 1
    mapMat = np.random.uniform(-1,1,(x_dim,z_dim))
    sio.savemat(save_folder + 'mapMat_circleHighDim_nonLinear_z' + str(z_dim) + '_x' + str(x_dim) +'.mat',{'mapMat':mapMat})
    from fullyConnectedModel import Encoder
    from fullyConnectedModel import Decoder
    sample_X,sample_orig,sample_labels = create_circle_data(nTrain//2,noise_std,mapMat,np.array([0.5,1]))
    anchor_X,anchor_orig = create_anchors_circle(num_anchor,0.0,mapMat,np.array([0.5,1]))
    batch_idxs = len(sample_X) // opt.batch_size
    sample_X_torch = torch.from_numpy(sample_X)
    sample_X_torch = sample_X_torch.float()
    anchor_X_torch = torch.from_numpy(anchor_X).float()
    
    test_inputs_X,test_orig,test_labels = create_circle_data(test_size//2,noise_std,mapMat,np.array([0.5,1]))
    test_inputs_torch = torch.from_numpy(test_inputs_X)
    test_inputs_torch = test_inputs_torch.float()
    encoder = Encoder(x_dim,z_dim,M)
    decoder = Decoder(x_dim,z_dim)
    num_class = 2

elif data_use == 'swiss2D':
    # Load data for swiss roll dataset
    nTrain = 1000
    noise_std = 0.01
    mapMat = np.random.uniform(-1,1,(x_dim,z_dim))
    sio.savemat(save_folder + 'mapMat_circleHighDim_nonLinear_z' + str(z_dim) + '_x' + str(x_dim) +'.mat',{'mapMat':mapMat})
    from fullyConnectedModel import Encoder
    from fullyConnectedModel import Decoder
    learn_anchor_flag = 1
    # Create points on a swiss roll that are mapped in a higher dimensional space
    sample_X,sample_orig,sample_labels = create_swissRoll_2D_data(nTrain,2.0,noise_std,mapMat)
    # Define anchor points
    anchor_X,anchor_orig,_ = create_anchors_swissRoll_2D(num_anchor,2.0,0.0,mapMat)
    batch_idxs = len(sample_X) // opt.batch_size
    sample_X_torch = torch.from_numpy(sample_X)
    sample_X_torch = sample_X_torch.float()
    anchor_X_torch = torch.from_numpy(anchor_X).float()
    
    test_inputs_X,test_orig,test_labels = create_swissRoll_2D_data(test_size,2.0,noise_std,mapMat)
    test_inputs_torch = torch.from_numpy(test_inputs_X)
    test_inputs_torch = test_inputs_torch.float()
    encoder = Encoder(x_dim,z_dim,M)
    decoder = Decoder(x_dim,z_dim)
    num_class = 1
elif data_use == 'rotDigits':
    # Load data for rotated digit dataset
    from covNetModel import Encoder
    from covNetModel import Decoder
    learn_anchor_flag = 0
    
    newClass = range(0,class_use.shape[0])
    
    data_X, data_y = load_mnist_classSelect('val',class_use,newClass)
    #data_X, data_y,_ = load_mnist('val')
    test_labels = data_y[0:test_size]
    test_inputs, test_angles = transform_image(data_X[0:test_size,:,:,:],test_labels,class_use,input_h,360.0,1)
    
    test_inputs_torch = torch.from_numpy(test_inputs)
    test_inputs_torch = test_inputs_torch.permute(0,3,1,2)
    test_inputs_torch = test_inputs_torch.float()
    
    data_X, data_y = load_mnist_classSelect('train',class_use,newClass)
    #data_X, data_y,_ = load_mnist('train')
    sample_X_torch = torch.from_numpy(data_X)
    sample_X_torch = sample_X_torch.permute(0,3,1,2)
    sample_X_torch = sample_X_torch.float()
    batch_idxs = len(data_X) // opt.batch_size
    encoder = Encoder(z_dim,opt.c_dim,opt.img_size)
    decoder = Decoder(z_dim,opt.c_dim,opt.img_size)
    num_class = class_use.shape[0]
elif data_use == 'natDigits':
    # Load data for natural MNIST variations
    from covNetModel import Encoder
    from covNetModel import Decoder
    learn_anchor_flag = 1
    class_transform = np.array([0,1,2,3,4,5,6,7,8,9])
    data_X, data_y,y_orig = load_mnist("val")
    
    test_labels = data_y[0:test_size]
    test_inputs = data_X[0:test_size,:,:,:]
    test_inputs_torch = torch.from_numpy(test_inputs)
    test_inputs_torch = test_inputs_torch.permute(0,3,1,2)
    test_inputs_torch = test_inputs_torch.float()

    data_X, data_y,y_orig = load_mnist("train")
    sample_labels = data_y
    sample_X_torch = torch.from_numpy(data_X)
    sample_X_torch = sample_X_torch.permute(0,3,1,2)
    sample_X_torch = sample_X_torch.float()


    batch_idxs = len(data_X) // opt.batch_size
    anchor_X,anchor_y = select_mnist_anchors(data_X,data_y,opt.num_anchor)
    anchor_X_torch = torch.from_numpy(anchor_X)
    anchor_X_torch= anchor_X_torch.permute(0,3,1,2)
    anchor_X_torch = anchor_X_torch.float()
    encoder = Encoder(z_dim,opt.c_dim,opt.img_size)
    decoder = Decoder(z_dim,opt.c_dim,opt.img_size)
    anchor_orig = 1
    num_class = 10
transNet = TransOpt()
sampler_c = Sample_c()

# Initialize weights
encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
    
## Initialize dictionary
Psi = Variable(torch.mul(torch.randn(N_use, M, dtype=torch.double),0.01), requires_grad=True)


# Load previously trained networks - replace checkpoint_folder with a checkpoint that is useful to you
if load_checkpoint == 1:
    checkpoint_folder = './results/checkpoint_folder/' 
    checkpoint = torch.load(checkpoint_folder + 'network_batch32_zdim10_step31000.pt') 
    encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
    decoder.load_state_dict(checkpoint['model_state_dict_decoder'])
    Psi = checkpoint['Psi']

Psi_use = Psi.detach().numpy()

# Define the loss functions
#mse_loss = torch.nn.MSELoss(reduction = 'mean')
mse_loss_sum = torch.nn.MSELoss(reduction = 'sum')
latent_mse_loss = torch.nn.MSELoss(reduction = 'sum')

# Initialize the anchor points if we're learning them
if learn_anchor_flag == 1:
    anchors = Variable(torch.randn(num_anchor*num_class,x_dim), requires_grad=True)
    anchors.data = anchor_X_torch.float()

# Specify optimizer
params_use = list(encoder.parameters()) + list(decoder.parameters()) + list(transNet.parameters()) 
optimizer_NN = torch.optim.Adam(params_use, lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler = StepLR(optimizer_NN, step_size=1, gamma=0.1)

# Initialize arrays for saving progress
start_time = time.time()
counter = 0
test_counter = 0
loss_save = np.zeros((opt.epoch*batch_idxs))
loss_recon = np.zeros((opt.epoch*batch_idxs))
loss_post_TO = np.zeros((opt.epoch*batch_idxs))
loss_post_l1 = np.zeros((opt.epoch*batch_idxs))
loss_prior_TO = np.zeros((opt.epoch*batch_idxs))
loss_frob = np.zeros((opt.epoch*batch_idxs))
lr_save = np.zeros((opt.epoch*batch_idxs))
time_save = np.zeros((opt.epoch*batch_idxs))
for epoch in xrange(opt.epoch):
    batch_counter = 1
    for idx in xrange(0, batch_idxs):
        if counter == num_pretrain_steps:
            # Adjust weights after warm up steps
            post_l1_weight = opt.post_l1_weight
            to_noise_std = opt.to_noise_std
            if num_pretrain_steps != 0:
                scheduler.step()
        epoch_time = time.time()
        
        
        # Load data batch
        if data_use == 'rotDigits':
            sample_labels_batch = data_y[idx*opt.batch_size:(idx+1)*opt.batch_size]
            batch_images_0,batch_angles = transform_image(data_X[idx*opt.batch_size:(idx+1)*opt.batch_size],sample_labels_batch,class_use,input_h,350.0,1)     
            batch_images_torch_0 = torch.from_numpy(batch_images_0)
            batch_images_torch_0 = batch_images_torch_0.permute(0,3,1,2)
            input_sample = batch_images_torch_0.float()
            
            angUse = list(np.arange(0,360,360/float(num_anchor)))
            anchors_np,anchor_angles = transform_image_specificAng(data_X[idx*opt.batch_size:(idx+1)*opt.batch_size],input_h,angUse)
            #anchors_np,anchor_angles = transform_image(data_X[idx*opt.batch_size:(idx+1)*opt.batch_size],sample_labels_batch,class_transform,input_h,350.0,num_anchor)     
            anchor_images_torch = torch.from_numpy(anchors_np)
            anchor_images_torch = anchor_images_torch.permute(0,3,1,2)
            anchors = anchor_images_torch.float()
            sample_labels_batch = range(0,batch_size)
        else:
            input_sample = sample_X_torch[idx*opt.batch_size:(idx+1)*opt.batch_size]
            sample_labels_batch = sample_labels[idx*opt.batch_size:(idx+1)*opt.batch_size]

        # Set NN gradients to zero
        optimizer_NN.zero_grad()
        
        # Encode the data into the latent space
        z_mu = encoder(input_sample)
        # Prepare latent vectors to be used in coefficient inference
        z_mu_scale = torch.div(z_mu,scale)
        z_mu_scale_np = z_mu_scale.detach().numpy()
    
        # Encode anchor points
        a_mu = encoder(anchors)  
        a_mu_scale = torch.div(a_mu,scale)

        # Loop for each sampled set of coefficient
        loss_total = 0.0
        loss_Psi = 0.0
        z_scale_save = np.zeros((num_c_samp,batch_size,z_dim))
        c_est_mu_samp_save = np.zeros((num_c_samp,batch_size,M))
        c_est_a_samp_save = np.zeros((num_c_samp,batch_size,num_anchor,M))
        for k in range(0,num_c_samp):
            # Sample the coefficients 
            z_coeff = sampler_c(batch_size,M,post_l1_weight)
            # Sample from the posterior
            z_scale = transNet(z_mu_scale.double(),z_coeff.double(),Psi,to_noise_std)
            z_scale_save[k,:,:] = z_scale.detach().numpy()
            z = torch.mul(z_scale,scale)
            z_scale_np = z_scale.detach().numpy()
            # Decode sampled latent vector
            x_est = decoder(z.float()).detach().numpy()
            
            # Compute the reconstruction loss
            recon_loss = 0.5*mse_loss_sum(decoder(z.float()).double(),input_sample.double())/batch_size
            
            # Incorporate the prior and posterior terms of the objective only after warm up
            if counter > num_pretrain_steps:
                # Compute the posterior loss function
                # Infer the coefficients between z_mu and z
                c_est_mu,E_mu,nit_mu,c_infer_time_post = compute_posterior_coeff(z_mu_scale_np,z_scale_np,Psi_use,opt.post_cInfer_weight,M)
                c_est_mu_samp_save[k,:,:] = c_est_mu
        
                # Transform mu with no noise using the inferred coefficients  
                z_est_mu_scale = transNet(z_mu_scale.double(),torch.from_numpy(c_est_mu),Psi,0.0)
                z_est_mu = torch.mul(z_est_mu_scale,scale)
                
                # Compute posterior terms of the objective
                post_TO_loss = (-0.5*opt.post_TO_weight*latent_mse_loss(z_scale.double(),z_est_mu_scale))/batch_size
                post_l1_loss = -torch.sum(torch.abs(torch.from_numpy(c_est_mu)))/batch_size
                
                prior_TO_sum, c_est_batch,E_anchor,nit_anchor,c_est_a_store,anchor_idx_use,num_anchor_use = compute_prior_obj(z_scale,Psi,a_mu_scale,sample_labels_batch,transNet,scale,prior_l1_weight,prior_weight,opt)  
                    
                c_est_a_samp_save[k,:,:,:] = c_est_batch
                prior_TO_sum = prior_TO_sum/batch_size
                
                # Compute final loss function
                loss_total = loss_total + recon_weight*recon_loss + post_TO_loss + opt.post_l1_weight*post_l1_loss + prior_TO_sum
                loss_Psi = loss_Psi +post_TO_loss + prior_TO_sum
            else:
                # Compute loss funciton during warm up
                loss_total = loss_total + recon_weight*recon_loss
                
        if counter > num_pretrain_steps:  
            # Add dictionary regularizer
            loss_total = loss_total/num_c_samp +  0.5*opt.gamma*torch.sum(torch.pow(Psi,2))
            loss_Psi = loss_Psi/num_c_samp +  0.5*opt.gamma*torch.sum(torch.pow(Psi,2))
            
            if learn_anchor_flag == 1:
                anchors.retain_grad()
            loss_val_comp = np.zeros((2))
            Psi_comp = np.zeros((2,N_use,M))  
            # Save the transport operator portion of the loss function to compare against the objective after the gradient update
            loss_val = loss_Psi.detach().numpy()
            loss_val_comp[0] = loss_val
        else:
            loss_total = loss_total/num_c_samp 
        
        # Take gradient step
        loss_total.backward()
        if counter <= num_pretrain_steps:
            optimizer_NN.step()
        else: 
            if net_count < num_net_steps or alternate_steps_flag == 0:
                optimizer_NN.step()
                net_count = net_count +1
            if psi_count < num_psi_steps or alternate_steps_flag == 0:
                
                Psi_comp[0,:,:] = Psi.detach().numpy()
                # Take gradient step on Psi
                Psi.data.sub_(lr_psi*Psi.grad.data)
                
                # Compute the transport operator portion of the loss function with the updated Psi values
                loss_Psi_new = 0.0
                loss_total_new = 0.0
                for k in range(0,num_c_samp):
                    z_scale_use = torch.from_numpy(z_scale_save[k,:,:])
                    z_est_mu_scale = transNet(z_mu_scale.double(),torch.from_numpy(c_est_mu_samp_save[k,:,:]),Psi,0.0)
                    post_TO_loss_new = (-0.5*latent_mse_loss(z_scale_use.double(),z_est_mu_scale))/batch_size
                    
                    prior_TO_sum_new = compute_prior_update(z_scale_use,Psi,c_est_a_samp_save[k,:,:,:],a_mu_scale,sample_labels_batch,transNet,scale,anchor_idx_use,prior_l1_weight,prior_weight,num_anchor_use,opt)
                    prior_TO_sum_new = prior_TO_sum_new/batch_size    
                    loss_Psi_new = loss_Psi_new + post_TO_loss_new + prior_TO_sum_new
                    loss_total_new = loss_total_new + recon_weight*recon_loss + post_TO_loss_new + opt.post_l1_weight*post_l1_loss + prior_TO_sum_new
                loss_Psi_new = loss_Psi_new/num_c_samp +  0.5*opt.gamma*torch.sum(torch.pow(Psi,2))
                loss_total_new = loss_total_new/num_c_samp +  0.5*opt.gamma*torch.sum(torch.pow(Psi,2))
                
                loss_val_new = loss_Psi_new.detach().numpy()
                loss_val_comp[1] = loss_val_new
                Psi_comp[1,:,:] = Psi.detach().numpy()
                      
                # Update anchor weights
                if learn_anchor_flag == 1:
                    anchor_grad = anchors.grad.detach().numpy()
                    anchors.data.sub_(lr_anchor*anchors.grad.data)
                
                # If the psi objective after the Psi update is higher than the objective before the update, don't accept the step and decrease lr
                # If the psi objective after the Psi update is lower than the objective before the update, accept the step and increase lr
                if loss_val_comp[1] > (loss_val_comp[0]) or np.isinf(loss_val_comp[1]):
                    Psi.data.add_(lr_psi*Psi.grad.data)
                    lr_psi = lr_psi*decay
                    print('Failed Step')
                    loss_val_use = loss_val_comp[0]
                else:
                    if counter < titrate_steps and lr_psi < max_psi_lr:
                         lr_psi = lr_psi/decay
                    loss_val_use = loss_val_comp[1]
            
                if counter >titrate_steps:
                    lr_psi = lr_psi*titrate_decay
                psi_count = psi_count +1
                
            # Update counts to determine how many network/psi updates have been taken before switching
            if net_count == num_net_steps and alternate_steps_flag == 1:
                net_count = net_count+1
                psi_count = 0
                prior_l1_weight = opt.prior_l1_weight
                prior_weight = opt.prior_weight
                recon_weight = opt.recon_weight*opt.netWeights_psteps
            if psi_count == num_psi_steps and alternate_steps_flag == 1:
                psi_count = psi_count +1
                net_count = 0
                prior_l1_weight = opt.prior_l1_weight*opt.priorWeight_nsteps
                prior_weight = opt.prior_weight*opt.priorWeight_nsteps
                recon_weight = opt.recon_weight
                
        # Zero out the gradients        
        if counter> num_pretrain_steps:   
            Psi.grad.data.zero_() 
            
            if learn_anchor_flag ==1:
                # Set anchor weights to 0
                anchors.grad.data.zero_()  
    
        # Save loss terms
        lr_save[counter] = lr_psi
        time_save[counter] = time.time() - epoch_time
        loss_save[counter] = loss_total.detach().numpy()
        loss_recon[counter] = recon_weight*recon_loss.detach().numpy()
        if counter > num_pretrain_steps: 
            loss_post_TO[counter] = post_TO_loss.detach().numpy()
            loss_post_l1[counter] = opt.post_l1_weight*post_l1_loss.detach().numpy()
            loss_prior_TO[counter] = prior_TO_sum.detach().numpy()
            loss_frob[counter] = (0.5*opt.gamma*torch.sum(torch.pow(Psi,2))).detach().numpy()
        print ("[Epoch %d/%d] [Batch %d/%d] time: %4.4f [loss: %f] [recon: %f] [post_TO: %f] [post_l1: %f] [prior_TO: %f]" % (epoch, opt.epoch,idx, batch_idxs,time.time() - start_time,
                                                             loss_save[counter],loss_recon[counter],loss_post_TO[counter],loss_post_l1[counter],loss_prior_TO[counter]))
        logging.info("[Epoch %d/%d] [Batch %d/%d] time: %4.4f [loss: %f] [recon: %f] [post_TO: %f] [post_l1: %f] [prior_TO: %f]" % (epoch, opt.epoch,idx, batch_idxs,time.time() - start_time,
                                                             loss_save[counter],loss_recon[counter],loss_post_TO[counter],loss_post_l1[counter],loss_prior_TO[counter]))
        if counter > num_pretrain_steps: 
            print ("lr: %f [loss 0: %f] [loss 1: %f]" % (lr_psi,loss_val_comp[0],loss_val_comp[1]))

        if np.mod(counter,100) == 0 and counter < num_pretrain_steps:
            a_mu_scale_np = a_mu_scale.detach().numpy()
            save_dict = {'a_mu_scale_np':a_mu_scale_np,'z_scale_np':z_scale_np,'z_coeff':z_coeff.detach().numpy(),
                         'loss_total':loss_save[:counter],'loss_recon':loss_recon[:counter],'loss_post_TO':loss_post_TO[:counter],'loss_post_l1':loss_post_l1[:counter],\
                         'loss_prior_TO':loss_prior_TO[:counter],'Psi_new':Psi.detach().numpy(),
                         'lr_save':lr_save[:counter],'time_save':time_save[:counter],'lr_main':opt.lr}
            if learn_anchor_flag == 1:
                save_dict['anchor_orig'] = anchor_orig
                save_dict['anchors'] = anchors.detach().numpy()
            sio.savemat(save_folder + 'lossVals.mat',save_dict)
            
        # Save test data    
        if np.mod(counter,5) == 0:
            # Plot sampled test outputs
            a_mu_scale_np = a_mu_scale.detach().numpy()
            sample_latent = encoder(test_inputs_torch)
            z_mu_scale = torch.div(sample_latent,scale)
            z_mu_scale_test_np = z_mu_scale.detach().numpy()
            z_coeff = sampler_c(test_size,M,post_l1_weight)
            z_scale = transNet(z_mu_scale.double(),z_coeff.double(),Psi,to_noise_std)
            z = torch.mul(z_scale,scale)
            sample_img = decoder(z.float())
            sample_orig_torch = decoder(sample_latent)
            if data_use == 'rotDigits' or data_use == 'natDigits':
                samples = sample_img[0:16,:,:,:]
                samples = samples.permute(0,2,3,1)
                samples = samples.detach().numpy()
                samples_orig = sample_orig_torch[0:16,:,:,:]
                samples_orig = samples_orig.permute(0,2,3,1)
                samples_orig = samples_orig.detach().numpy()
                save_images(samples, [4, 4],
                          '{}train_{:02d}_{:04d}_sample.png'.format(sample_dir, epoch, idx))
                sample_inputs_use = test_inputs[0:16,:,:,:]
                save_images(samples_orig, [4, 4],
                          '{}train_{:02d}_{:04d}_orig.png'.format(sample_dir, epoch, idx))
            if counter > num_pretrain_steps:
                # Save posterior samples
                numTestSamp = 20
                z_test_samp = np.zeros((numTestSamp,batch_size,z_dim))
                z_coeff_test_np = np.zeros((numTestSamp,batch_size,M))
                for s_idx in range(0,numTestSamp):
                    z_coeff_test = sampler_c(batch_size,M,post_l1_weight)
                    z_coeff_test_np[s_idx,:,:] = z_coeff_test.detach().numpy()
                    z_scale_test = transNet(z_mu_scale.double(),z_coeff_test.double(),Psi,to_noise_std)
                    z_test_samp[s_idx,:,:] = z_scale_test.detach().numpy()
                    
                save_dict= {'c_est_mu':c_est_mu,'E_mu':E_mu,'nit_mu':nit_mu,'c_est_batch':c_est_batch,'E_anchor':E_anchor,'nit_anchor':nit_anchor,
                            'a_mu_scale_np':a_mu_scale_np,'z_scale_np':z_scale_np,'z_test_samp':z_test_samp,'x_est':x_est,'c_est_a_store':c_est_a_store,
                            'z_coeff':z_coeff.detach().numpy(),'loss_total':loss_save[:counter],'loss_recon':loss_recon[:counter],'loss_post_TO':loss_post_TO[:counter],
                            'loss_post_l1':loss_post_l1[:counter],'loss_prior_TO':loss_prior_TO[:counter],'Psi_new':Psi.detach().numpy(),'lr_save':lr_save[:counter],
                            'time_save':time_save[:counter],'lr_main':opt.lr,'z_mu_scale_test_np':z_mu_scale_test_np,'z_coeff_test_np':z_coeff_test_np}
                
                if data_use == 'rotDigits' or data_use == 'natDigits':
                    save_dict['samples'] = samples
                    save_dict['samples_orig'] = samples_orig
                if learn_anchor_flag == 1:
                    save_dict['anchor_orig'] = anchor_orig
                    save_dict['anchors'] = anchors.detach().numpy()
                if np.mod(counter,500) == 0:
                    sio.savemat(save_folder + 'spreadInferenceTest_step' + str(counter) + '.mat',save_dict)
                else:
                    sio.savemat(save_folder + 'spreadInferenceTest_current.mat',save_dict)
                    
        # Save network weights            
        if np.mod(counter,5) == 0 and counter != 0: 
            net_save_dict = {'step': counter,'epoch': epoch,'model_state_dict_encoder': encoder.state_dict(),'model_state_dict_decoder': decoder.state_dict(),
                             'model_state_dict_transOpt': transNet.state_dict(),'optimizer_auto_state_dict': optimizer_NN.state_dict(),'loss': loss_total,'Psi':Psi}
            if np.mod(counter,500) == 0 and counter > num_pretrain_steps:

                if learn_anchor_flag == 1:
                    net_save_dict['anchors'] = anchors
                    save_name = save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim)  + '_step' + str(counter) + '.pt'
                else:
                    save_name = save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim)   + '_step' + str(counter) + '.pt'
            else:
                if counter < num_pretrain_steps:
                    save_name = save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim)  + '_pretrain.pt'
                else:
                    save_name = save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim)  + '_current.pt'
                if learn_anchor_flag == 1:
                    net_save_dict['anchors'] = anchors
            torch.save(net_save_dict,save_name)
                                                     
        counter += 1

net_save_dict = {'step': counter,'epoch': epoch,'model_state_dict_encoder': encoder.state_dict(),'model_state_dict_decoder': decoder.state_dict(),
                             'model_state_dict_transOpt': transNet.state_dict(),'optimizer_auto_state_dict': optimizer_NN.state_dict(),'loss': loss_total,'Psi':Psi}
if learn_anchor_flag == 1:
    net_save_dict['anchors'] = anchors
    save_name = save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim)  + '_step' + str(counter) + '.pt'
else:
    save_name = save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim)   + '_step' + str(counter) + '.pt'
