#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import os
import time
import numpy as np
import scipy.io as sio
import scipy as sp
from scipy.optimize import minimize 

import argparse

from torch.autograd import Variable


import torch.nn as nn
import torch.nn.functional as F
import torch

from transOptModel import TransOpt

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./results/TOVAE', help='folder name')
parser.add_argument('--epoch', type=int, default=75, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--z_dim', type=int, default=10, help='Dimension of the latent space')
parser.add_argument('--x_dim', type=int, default=20, help='Dimension of the input space')
parser.add_argument('--c_samp', type=int, default=1, help='Number of samples from the c distribution')
parser.add_argument('--num_anchor', type=int, default=10, help='Number of anchor points per class')
parser.add_argument('--M', type=int, default=1, help='Number of dictionary elements')
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
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--zeta', type=float, default=0.001, help='gd: weight on coefficient regularizer')
parser.add_argument('--binarize_flag',type = bool,default = False, help='flag of weather or not to binarize the image data')
parser.add_argument('--num_net_steps', type=int, default=20, help='Number of steps on only the network weights')
parser.add_argument('--num_psi_steps', type=int, default=60, help='Number of steps on only the psi weights')
parser.add_argument('--numRestart', type=int, default=1, help='number of restarts for coefficient inference')
parser.add_argument('--priorWeight_nsteps', type=float, default= 0.0001, help='Weight of the prior during the network weight update steps')
parser.add_argument('--netWeights_psteps', type=float, default=0.0001, help='Weight of the reconstruction loss during the transport operator weight update steps')
parser.add_argument('--to_noise_std', type=float, default=0.001, help='Noise for sampling gaussian noise in latent space')
parser.add_argument('--num_pretrain_steps', type=int, default=-1, help='Number of steps to train the network with no VAE component')
parser.add_argument('--data_use', type=str,default = 'rotDigits',help='Specify which dataset to use [swiss2D_identity,swiss2D,rotDigits,natDigits]')
parser.add_argument('--alternate_steps_flag', type=int, default=1, help='[0/1] to specify whether to alternate between steps updating net weights and psi weights ')

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

def transOptObj_c(c,Psi,x0,x1,zeta):
    N = np.int(np.sqrt(Psi.shape[0]))
    M = np.int(Psi.shape[1])
    coeff_use = np.expand_dims(c,axis=1)
    x0_use = np.expand_dims(x0,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    T = np.real(sp.linalg.expm(A))
    x1_est= np.dot(T,x0_use)[:,0]
    objFun = 0.5*np.linalg.norm(x1-x1_est)**2 + zeta*np.sum(np.abs(c))
    
    return objFun

def transOptDerv_c(c,Psi,x0,x1,zeta):
    N = np.int(np.sqrt(Psi.shape[0]))
    M = np.int(Psi.shape[1])
    coeff_use = np.expand_dims(c,axis=1)
    x0_use = np.expand_dims(x0,axis=1)
    x1_use = np.expand_dims(x1,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    T = np.real(sp.linalg.expm(A))
    
    eig_out = np.linalg.eig(A)
    U = eig_out[1]
    D = eig_out[0]
    V = np.linalg.inv(U)
    V = V.T

    innerVal = np.dot(-x1_use,x0_use.T) + np.dot(T,np.dot(x0_use,x0_use.T))
    P = np.dot(np.dot(U.T,innerVal),V)
    
    F_mat = np.zeros((D.shape[0],D.shape[0]),dtype=np.complex128)
    for alpha in range(0,D.shape[0]):
        for beta in range(0,D.shape[0]):
            if D[alpha] == D[beta]:
                F_mat[alpha,beta] = np.exp(D[alpha])
            else:
                F_mat[alpha,beta] = (np.exp(D[beta])-np.exp(D[alpha]))/(D[beta]-D[alpha])
    
    fp = np.multiply(F_mat,P)
    Q1 = np.dot(V,fp)
    Q = np.dot(Q1,U.T)
    c_grad = np.real(np.dot(np.reshape(Q,-1,order='F'),Psi) + zeta*np.sign(c))
    return c_grad

def infer_transOpt_coeff(x0,x1,Psi,zeta,randMin,randMax):
    c0 = np.random.uniform(randMin,randMax,M)
    opt_out = minimize(transOptObj_c,c0,args=(Psi_use,x0,x1,zeta),method = 'CG',jac=transOptDerv_c,options={'maxiter':50,'disp':False},tol = 10^-7)
    c_est = opt_out['x']
    E = opt_out['fun']
    nit = opt_out['nit']
    return c_est, E, nit


    
def compute_arc_length(Psi,coeff_infer,t,x0,N):
    A_mat = np.reshape(np.dot(Psi,coeff_infer),(N,N),order='F')
    arc_len = 0.0
    for t_use in t:
        T = np.real(sp.linalg.expm(A_mat*t_use))
        arc_len = arc_len + t_use*np.linalg.norm(A_mat*T*x0)
        
    return arc_len 
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


# Parameters for finding the closest anchor points
closest_anchor_flag = 1
t_use = np.arange(0,1.0,0.01)

# Initialize the counts for alternating steps
net_count = 0
psi_count = num_psi_steps +1

# Specify which classes we want to train on
class_use = np.array([0,1,2,3,4,5,6,7,8,9])
class_use_str = np.array2string(class_use)    
newClass = range(0,class_use.shape[0])
    

test_size = batch_size

y_dim = 10
t = np.arange(-0.625*80,0.65625*80,0.03125*80)
batch_orig = 32
stepUse =5000

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
start_time = time.time()
counter = 1

mse_loss = torch.nn.MSELoss(reduction = 'mean')

from covNetModel import Encoder
from covNetModel import Decoder
    
encoder = Encoder(z_dim,opt.c_dim,opt.img_size)
decoder = Decoder(z_dim,opt.c_dim,opt.img_size)
transNet = TransOpt()

# Load a pretrained file - make sure the model parameters fit the parameters of rht pre-trained models
checkpoint = torch.load('./pretrained_models/rotMNIST/network_batch32_zdim10.pt')
# Or specify a network that's been trained independently
#checkpoint = torch.load(save_folder + 'network_batch' + str(batch_orig) + '_zdim' + str(opt.z_dim) +  '_step' + str(stepUse) + '.pt')

encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
decoder.load_state_dict(checkpoint['model_state_dict_decoder'])

#data_X, data_y = load_mnist("val")
data_X, data_y = load_mnist_classSelect('test',class_use,newClass)

Psi = Variable(torch.mul(torch.randn(N_use, M, dtype=torch.double),0.01), requires_grad=True)
Psi = checkpoint['Psi']
Psi_use = Psi.detach().numpy()


imgChoice = np.zeros((y_dim,28,28,1))   
test_img = np.zeros((y_dim,input_h,input_w))
test_latent = np.zeros((y_dim,z_dim)) 
for k in range(0,10):
    class_num = k
    idxClass = np.where(data_y[:,class_num] == 1)[0]
    numEx = len(idxClass)
    #idxChoice = idxClass[np.random.randint(low = 0, high = numEx,size=1)]
    #idxChoice = idxClass[int(random.uniform(0, numEx))]
    idxChoice = idxClass[0]
    imgChoice[k,:,:,:] = data_X[idxChoice,:,:,:]
    imgInput = np.expand_dims(data_X[idxChoice,:,:,:],axis=0)
    #input_temp = imgInput
    input_temp,batch_angles = transform_image(imgInput,np.expand_dims(data_y[idxChoice,:],axis=0),class_use,input_h,0.0,1)     
    input_temp = torch.from_numpy(input_temp)
    input_temp =input_temp.permute(0,3,1,2)
    input_temp =input_temp.float()
    
    test_latent_temp = encoder(input_temp)
    test_latent_temp_scale = torch.div(test_latent_temp,scale)
    test_img_temp = decoder(test_latent_temp)
    test_img_temp = test_img_temp.permute(0,2,3,1)
    
    x0 = test_latent_temp.detach().numpy()
    z_seq = np.zeros((M,len(t),z_dim))
    img_seq = np.zeros((M,len(t),input_h,input_w,opt.c_dim))
    for m in range(0,M):
        coeff_use = np.zeros((M))
        t_count = 0
        for t_use in t:
            coeff_use[m] = t_use
            coeff_input = np.expand_dims(coeff_use,axis=0)
            z_est = transNet(test_latent_temp_scale.double(),torch.from_numpy(coeff_input),Psi,0.0)
            z_est = torch.mul(z_est,scale)
            transImgOut = decoder(z_est.float())
            z_est_np = z_est.detach().numpy()
            transImgOut = transImgOut.permute(0,2,3,1)
            transImg_np = transImgOut.detach().numpy()
            
            z_seq[m,t_count,:] = z_est_np
            img_seq[m,t_count,:,:,:] = transImg_np
            
            t_count = t_count+1
        print "Class " + str(k) + " Operator " + str(m)
    sio.savemat(sample_dir + '/transOptOrbitTest_rotDigit_startDigit_step' + str(stepUse) + '_' + str(k+1) + '.mat',{'latent_seq':z_seq,'imgOut':img_seq,'Psi_new':Psi_use,'t_vals':t,'imgChoice':imgChoice})
 
numEx = 100
imgLabel = np.zeros((numEx,10))
test_img = np.zeros((numEx,input_h,input_w))
test_latent = np.zeros((numEx,z_dim)) 
idxPoss = np.random.randint(low = 0, high = data_y.shape[0],size=numEx)
z_seq = np.zeros((numEx,M,len(t),z_dim))
img_seq = np.zeros((numEx,M,len(t),input_h,input_w,opt.c_dim))
for k in range(0,numEx):
    idxChoice = idxPoss[k]

    imgLabel[k,:] = data_y[idxChoice,:]
    imgInput = np.expand_dims(data_X[idxChoice,:,:,:],axis=0)
    #input_temp = imgInput
    input_temp,batch_angles = transform_image(imgInput,np.expand_dims(data_y[idxChoice,:],axis=0),class_use,input_h,0.0,1)     
    input_temp = torch.from_numpy(input_temp)
    input_temp =input_temp.permute(0,3,1,2)
    input_temp =input_temp.float()
    
    test_latent_temp = encoder(input_temp)
    test_latent_temp_scale = torch.div(test_latent_temp,scale)
    test_img_temp = decoder(test_latent_temp)
    test_img_temp = test_img_temp.permute(0,2,3,1)
    
    x0 = test_latent_temp.detach().numpy()
    for m in range(0,M):
        coeff_use = np.zeros((M))
        t_count = 0
        for t_use in t:
            coeff_use[m] = t_use
            coeff_input = np.expand_dims(coeff_use,axis=0)
            z_est = transNet(test_latent_temp_scale.double(),torch.from_numpy(coeff_input),Psi,0.0)
            z_est = torch.mul(z_est,scale)
            transImgOut = decoder(z_est.float())
            z_est_np = z_est.detach().numpy()
            transImgOut = transImgOut.permute(0,2,3,1)
            transImg_np = transImgOut.detach().numpy()
            
            z_seq[k,m,t_count,:] = z_est_np
            img_seq[k,m,t_count,:,:,:] = transImg_np
            
            t_count = t_count+1
        print "Ex " + str(k) + " Operator " + str(m)
        
sio.savemat(sample_dir + '/transOptOrbitTest_rotDigit_randDigit_step' + str(stepUse) + '.mat',{'latent_seq':z_seq,'imgOut':img_seq,'Psi_new':Psi_use,'t_vals':t,'imgLabel':imgLabel})



sampler_c = Sample_c()    
num_c_samp = 16    
imgOrig = np.zeros((y_dim,32,32,1))

z_pt_samp = np.zeros((y_dim,num_c_samp,z_dim))
img_samp = np.zeros((y_dim,num_c_samp,32,32,1))
for k in range(0,10):
    class_num = k
    idxClass = np.where(data_y[:,class_num] == 1)[0]
    numEx = len(idxClass)

    idxChoice = idxClass[2]
    imgInput = np.expand_dims(data_X[idxChoice,:,:,:],axis=0)
    input_temp,batch_angles = transform_image(imgInput,np.expand_dims(data_y[idxChoice,:],axis=0),class_use,input_h,0.0,1)     
    
    input_temp = torch.from_numpy(input_temp)
    input_temp =input_temp.permute(0,3,1,2)
    input_temp =input_temp.float()
    
    test_latent_temp = encoder(input_temp)
    test_latent_temp_scale = torch.div(test_latent_temp,scale)
    test_img_temp = decoder(test_latent_temp)
    test_img_temp = test_img_temp.permute(0,2,3,1)
    imgOrig[k,:,:,:] = np.expand_dims(test_img_temp .detach().numpy(),axis=0)

    
    
    for m in range(0,num_c_samp):
        z_coeff = sampler_c(1,M,opt.post_l1_weight*0.15)
        z_scale_samp = transNet(test_latent_temp_scale.double(),z_coeff.double(),Psi,opt.to_noise_std*3)
        z_pt_samp[k,m,:] = z_scale_samp.detach().numpy()
    
        z_samp = torch.mul(z_scale_samp,scale)
        transImgSamp = decoder(z_samp.float())
        transImgSamp = transImgSamp.permute(0,2,3,1)
        transSamp_np = transImgSamp.detach().numpy()
        img_samp[k,m,:,:,:] = transSamp_np
        
        
        

        print "Sampling: Class " + str(k) 
    sio.savemat(sample_dir + '/transOptSampleTest_rotDigit_startDigit_step' + str(stepUse) + '.mat',{'imgOrig':imgOrig,'z_pt_samp':z_pt_samp,'img_samp':img_samp})