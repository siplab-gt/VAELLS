#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import time
import numpy as np
import scipy as sp
from scipy.optimize import minimize 

import torch

from transOptModel import TransOpt

from utils import *
#from test_metrics import *

def transOptObj_c(c,Psi,x0,x1,zeta):
    """
    Define forward pass for transport operator objective with regularizer on coefficients
    
    Inputs:
        - c:    Vector of transpor toperator coefficients [M]
        - Psi:  Transport operator dictionarys [N^2 x M]
        - x0:   Starting point for transport operator path [N]
        - x1:   Ending point for transport operator path [N]
        - zeta: Weight on the l1 coefficient regularizer
        
    Outputs:
        - objFun: Computed transport operator objective
    """
    N = np.int(np.sqrt(Psi.shape[0]))
    coeff_use = np.expand_dims(c,axis=1)
    x0_use = np.expand_dims(x0,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    T = np.real(sp.linalg.expm(A))
    x1_est= np.dot(T,x0_use)[:,0]
    objFun = 0.5*np.linalg.norm(x1-x1_est)**2 + zeta*np.sum(np.abs(c))
    
    return objFun

def transOptDerv_c(c,Psi,x0,x1,zeta):
    """
    Compute the gradient for the transport operator objective with regularizer on coefficients
    
    Inputs:
        - c:    Vector of transpor toperator coefficients [M]
        - Psi:  Transport operator dictionarys [N^2 x M]
        - x0:   Starting point for transport operator path [N]
        - x1:   Ending point for transport operator path [N]
        - zeta: Weight on the l1 coefficient regularizer
        
    Outputs:
        - c_grad: Gradient of the transport operator objective with repsect to the coefficients
    """
    N = np.int(np.sqrt(Psi.shape[0]))
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
    """
    Infer the transport operator coefficients
    
    Inputs:
        - x0:       Starting point for transport operator path [N]
        - x1:       Ending point for transport operator path [N]
        - Psi:      Transport operator dictionarys [N^2 x M]
        - zeta:     Weight on the l1 coefficient regularizer
        - randMin:  Minimium value for the uniform distribution used to intialize coefficeints
        - randMax:  Maximum value for the uniform distribition used to initializer the coefficeints
        
    Outputs:
        - c_est:    Final inferred coefficients
        - E:        Final objective function value
        - nit:      Number of inference steps
    """
    M = Psi.shape[1]
    c0 = np.random.uniform(randMin,randMax,M)
    opt_out = minimize(transOptObj_c,c0,args=(Psi,x0,x1,zeta),method = 'CG',jac=transOptDerv_c,options={'maxiter':50,'disp':False},tol = 10^-7)
    c_est = opt_out['x']
    E = opt_out['fun']
    nit = opt_out['nit']
    return c_est, E, nit

def compute_posterior_coeff(z0,z1,Psi_use,post_cInfer_weight,M):
    batch_size = z0.shape[0]
    c_est_mu = np.zeros((batch_size,M))
    E_mu = np.zeros((batch_size,1))
    nit_mu = np.zeros((batch_size,1))
    c_infer_time_post = np.zeros((batch_size,1))
    
    for b in range(0,batch_size):
        c_infer_time_start = time.time()
        x0 = z0[b,:].astype('double')
        x1 = z1[b,:].astype('double')
        c_est_mu[b,:],E_mu[b],nit_mu[b] = infer_transOpt_coeff(x0,x1,Psi_use.astype('double'),post_cInfer_weight,0.0,1.0)
        c_infer_time_post[b] = time.time()-c_infer_time_start
        
    return c_est_mu, E_mu, nit_mu, c_infer_time_post
    
def compute_prior_obj(z_scale,Psi,a_mu_scale,sample_labels_batch,transNet,scale,prior_l1_weight,prior_weight,opt):
    # Detach variables
    z1 = z_scale.detach().numpy()
    a_mu_scale_np = a_mu_scale.detach().numpy()
    Psi_use = Psi.detach().numpy()
    # Initialize arrays for saving inference details
    prior_TO_sum = 0.0
    c_est_batch= np.zeros((opt.batch_size,opt.num_anchor,opt.M))
    E_anchor= np.zeros((opt.batch_size,opt.num_anchor,opt.numRestart))
    nit_anchor = np.zeros((opt.batch_size,opt.num_anchor,opt.numRestart))
    c_est_a_store = np.zeros((opt.batch_size,opt.num_anchor,opt.numRestart,opt.M))
    anchor_idx_use = np.zeros((opt.batch_size))
    for b in range(0,opt.batch_size):
        x1 = z1[b,:].astype('double')
        prior_TO_anchor_sum = 0.0
        c_est_a = np.zeros((opt.num_anchor,opt.M))
        
        # Specify the anchors that are compared to each sample
        if opt.data_use == 'natDigits':
            label_use = np.where(sample_labels_batch[b,:]==1)[0]
        else:
            label_use = sample_labels_batch[b]
        anchors_use_np = a_mu_scale_np[int(opt.num_anchor*label_use):int(opt.num_anchor*(label_use+1)),:]
        a_mu_scale_use = a_mu_scale[int(opt.num_anchor*label_use):int(opt.num_anchor*(label_use+1)),:]

        # Infer coefficients between 
        man_dist_min = 1000000.0
        for a_idx in range(0,opt.num_anchor):
            # Infer the coefficients between anchors and z with random restarts
            x0 = anchors_use_np[a_idx,:].astype('double')
            E_single = np.zeros((opt.numRestart))
            for r_idx in range(0,opt.numRestart):
                rangeMin = opt.coeffRandStart + r_idx*opt.coeffRandAdd
                rangeMax = rangeMin + opt.coeffRandAdd
                c_est_a_store[b,a_idx,r_idx,:],E_anchor[b,a_idx,r_idx],nit_anchor[b,a_idx,r_idx] = infer_transOpt_coeff(x0,x1,Psi_use.astype('double'),opt.prior_cInfer_weight,rangeMin,rangeMax)
                E_single[r_idx] = E_anchor[b,a_idx,r_idx]
            # Select the coefficients from the random restart that resulted in the lowest objective function
            minIdx = np.argmin(E_single)
            c_est_a_ind = c_est_a_store[b,a_idx,minIdx,:]
            c_est_a[a_idx,:] = c_est_a_store[b,a_idx,minIdx,:]
            c_est_a_ind = c_est_a[a_idx,:]
            man_dist = E_anchor[b,a_idx,minIdx]
            # If using only the closest anchor point, only add to the objective function if the anchor has the smallest manifold distance
            if opt.closest_anchor_flag == 0:
                z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_scale_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                z_est_a_ind = torch.mul(z_est_a_scale_ind,scale) 
                prior_TO_temp = torch.exp(-0.5*prior_weight*torch.sum(torch.pow(z_scale[b,:].double()-z_est_a_scale_ind,2))-prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                prior_TO_anchor_sum = prior_TO_anchor_sum+prior_TO_temp
            elif opt.closest_anchor_flag == 1:
                if man_dist < man_dist_min:
                    z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_scale_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                    z_est_a_ind = torch.mul(z_est_a_scale_ind,scale) 
                    prior_TO_anchor_sum = torch.exp(-0.5*prior_weight*torch.sum(torch.pow(z_scale[b,:].double()-z_est_a_scale_ind,2))-prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                    man_dist_min = man_dist
                    anchor_idx_use[b] = a_idx
        if opt.closest_anchor_flag == 1:
            num_anchor_use = 1
        else:
            num_anchor_use = opt.num_anchor
        prior_TO_sum = prior_TO_sum - torch.log(prior_TO_anchor_sum/num_anchor_use)
        
        c_est_batch[b,:,:] = c_est_a
        
    return prior_TO_sum, c_est_batch,E_anchor,nit_anchor,c_est_a_store,anchor_idx_use,num_anchor_use

def compute_prior_update(z_scale_use,Psi,c_est_a_samp_save,a_mu_scale,sample_labels_batch,transNet,scale,anchor_idx_use,prior_l1_weight,prior_weight,num_anchor_use,opt):
    #a_mu_scale_np = a_mu_scale.detach().numpy()
    prior_TO_sum_new = 0.0
    for b in range(0,opt.batch_size):

        prior_TO_anchor_sum_new = 0.0

        if opt.data_use == 'natDigits':
            label_use = np.where(sample_labels_batch[b,:]==1)[0]
        else:
            label_use = sample_labels_batch[b]
        #anchors_use_np = a_mu_scale_np[int(opt.num_anchor*label_use):int(opt.num_anchor*(label_use+1)),:]
        a_mu_scale_use = a_mu_scale[int(opt.num_anchor*label_use):int(opt.num_anchor*(label_use+1)),:]
        for a_idx in range(0,opt.num_anchor):
            # Infer the coefficients between anchors and z
            if opt.closest_anchor_flag == 0 or anchor_idx_use[b] == a_idx:
                c_est_a_ind = c_est_a_samp_save[b,a_idx,:]
                z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_scale_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                #z_est_a_ind = torch.mul(z_est_a_scale_ind,scale) 
                prior_TO_temp_new = torch.exp(-0.5*prior_weight*torch.sum(torch.pow(z_scale_use[b,:].double()-z_est_a_scale_ind,2))-prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                prior_TO_anchor_sum_new = prior_TO_anchor_sum_new+prior_TO_temp_new

        #print(torch.log(prior_TO_anchor_sum/num_anchor).detach().numpy())
        
        prior_TO_sum_new = prior_TO_sum_new - torch.log(prior_TO_anchor_sum_new/num_anchor_use)
        
    return prior_TO_sum_new
    
