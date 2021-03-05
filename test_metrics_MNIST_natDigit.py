# -*- coding: utf-8 -*-

from __future__ import division
import time
import numpy as np
from six.moves import xrange
import scipy.io as sio



import torch

from utils import *
from trans_opt_objectives import *



def log_likelihood(encoder,decoder,transNet,sampler_c,Psi,x,sample_labels,anchors,to_noise_std,num_anchor,M,numRestart,scale,opt,save_folder,k =10):
    '''
    Compute log_likelihood on rotated MNIST digits
    
    Inputs:
        - encoder:      Encoder network
        - decoder:      Decoder network
        - transNet:     Transport operator layer
        - sampler_c:    Layer that samples the transport operator coefficients
        - Psi:          Current transport operator dictionary elements
        - x:            Batch of data [batch_size,H,W,C]
        - labels:       Batch of labels [batch_size,y_dim]
        - to_noise_std: Sampling noise standard deviation for Gaussian prior distribution
        - num_anchor:   Number of anchors per class
        - M:            Number of transport operator dictionary elements
        - numRestart:   Number of restarts for coefficient inference
        - scale:        Value to scale the latent vectors by to get them in a range that is suitable for coefficient inference
        - save_folder:  Directory for saving data
        - k:            Number of samples of each latent vector for computing the LL
        
        
    Outputs:
        - LL_total:     Average log-likelihood with all constants added
        - LL_inner:     Array of log-likelihood with all constants added for each latent vector
        - LL_total_no_add: Average log-likelihood with no constants added
        - LL_inner_no_add: Array of log-likelihood with no constants added
        
    '''
    # Compute the test log likelihood
    batch_size_use = x.size(0)
    input_h = opt.img_size
    D = np.prod(x.size())/batch_size_use
     
    #num_anchor = anchors.size(0)
    a_mu= encoder(anchors)
    a_mu_scale = torch.div(a_mu,scale)
    a_mu_scale_np = a_mu_scale.detach().numpy()

    d = a_mu.size(1)
    sigma_recon = np.sqrt(1.0/(opt.recon_weight))
    p_x_add = -D/2*np.log(2*np.pi)-D*np.log(sigma_recon)
    
    #gamma_post = 1/np.sqrt(opt.post_TO_weight)
    gamma_post = to_noise_std
    post_TO_weight = 1.0/(gamma_post**2)
    b_post = 1.0/opt.post_l1_weight
    q_z_x_add = -d/2.0*np.log(2.0*np.pi)-d*np.log(gamma_post) - M*np.log(2.0*b_post) 
    
    gamma_prior = to_noise_std
    #gamma_prior = 1/np.sqrt(opt.prior_weight)
    prior_weight =1.0/(gamma_prior**2)
    #b_prior = 1/opt.prior_l1_weight
    b_prior = 1.0/opt.post_l1_weight
    prior_l1_weight = opt.post_l1_weight
    p_z_add = -d/2.0*np.log(2*np.pi)-d*np.log(gamma_prior) - M*np.log(2.0*b_prior) 
    
    LL_inner = np.zeros((batch_size_use,1))
    LL_inner_no_add = np.zeros((batch_size_use,1))
    time_save = np.zeros((batch_size_use))
    #log_q_z_x_no_add_store = np.zeros((batch_size_use))
    #log_p_x_z_no_add_store = np.zeros((batch_size_use))
    for n in range(0,batch_size_use):
        batch_time_start = time.time()
        
        x_ind_torch = torch.unsqueeze(x[n,:,:,:],0)        
        
        
        
        z_mu  = encoder(x_ind_torch)
        z_mu_scale = torch.div(z_mu,scale)
        z_mu_scale_np = z_mu_scale.detach().numpy()

        x_repeat = x_ind_torch.repeat(k,1,1,1) # This may need to be expanded for images
        z_mu_scale_repeat = z_mu_scale.repeat(k,1)
        z_coeff = sampler_c(k,M,opt.post_l1_weight)
        z_scale = transNet(z_mu_scale_repeat.double(),z_coeff.double(),Psi,to_noise_std)
        z = torch.mul(z_scale,scale)
        z_scale_np = z_scale.detach().numpy()
        
        
        log_p_x_z_no_add = -0.5*opt.recon_weight*torch.sum((decoder(z.float()).double().reshape(k,-1)-x_repeat.double().reshape(k,-1))**2,1)
        log_p_x_z_no_add_store = log_p_x_z_no_add.detach().numpy()
        log_p_x_z = log_p_x_z_no_add +p_x_add

        x0 = z_mu_scale_np[0,:].astype('double')
        c_est_mu = np.zeros((k,M))
        for b in range(0,k):
            x1 = z_scale_np[b,:].astype('double')
            c_est_mu[b,:],E,nit = infer_transOpt_coeff(x0,x1,Psi.detach().numpy().astype('double'),opt.post_cInfer_weight,0.0,1.0)
        z_est_mu_scale = transNet(z_mu_scale_repeat.double(),torch.from_numpy(c_est_mu),Psi,0.0)

        
        log_q_z_x_no_add = -post_TO_weight*0.5*torch.sum((scale*(z_scale.double()-z_est_mu_scale))**2,1) -opt.post_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_mu)),1) 
        log_q_z_x_no_add_store = log_q_z_x_no_add.detach().numpy()
        log_q_z_x = log_q_z_x_no_add + q_z_x_add
        # Compute the prior loss function
        if sample_labels.ndim == 1:
            label_use = sample_labels[n] # Change this depending on the application
        else:
            label_use = np.where(sample_labels[n,:]==1)[0]

        anchors_use_np = a_mu_scale_np[int(num_anchor*label_use):int(num_anchor*(label_use+1)),:]
        a_mu_use = a_mu_scale[int(num_anchor*label_use):int(num_anchor*(label_use+1)),:]
        log_p_z_no_add = torch.zeros(k)
        log_p_z = torch.zeros(k)
        
        anchor_idx_use = np.zeros((opt.batch_size))
        for b in range(0,k):
            x1 = z_scale_np[b,:].astype('double')

            prior_TO_anchor_sum = 0.0
            c_est_a = np.zeros((num_anchor,M))
            E_anchor= np.zeros((num_anchor,numRestart))
            arc_len_min = 1000000.0
            for a_idx in range(0,num_anchor):
                # Infer the coefficients between anchors and z
                
                x0 = anchors_use_np[a_idx,:].astype('double')
                E_single = np.zeros((numRestart))
                c_est_a_store = np.zeros((numRestart,M))
                for r_idx in range(0,numRestart):
                    #rangeMin = 0.0
                    #rangeMax = 1.0
                    #rangeMin = -600+r_idx*300
                    #rangeMax = rangeMin + 300
                    rangeMin = -2.5 + r_idx*5
                    rangeMax = rangeMin + 2.5
                    c_est_a_store[r_idx,:],E_anchor[a_idx,r_idx],nit_anchor = infer_transOpt_coeff(x0,x1,Psi.detach().numpy().astype('double'),opt.prior_cInfer_weight,rangeMin,rangeMax)
                    E_single[r_idx] = E_anchor[a_idx,r_idx]
                minIdx = np.argmin(E_single)
                c_est_a[a_idx,:] = c_est_a_store[minIdx,:]
                #c_est_a[a_idx,:],E_anchor[b,a_idx],nit_anchor[b,a_idx] = infer_transOpt_coeff(x0,x1,Psi_use.astype('double'),opt.prior_cInfer_weight,-100.0,100.0)
                c_est_a_ind = c_est_a[a_idx,:]
                test = 0
                arc_len = E_anchor[a_idx,minIdx]
                if opt.closest_anchor_flag == 0:
                    z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                    z_est_a_ind = torch.mul(z_est_a_scale_ind,scale) 
                    prior_TO_temp = torch.exp(-0.5*prior_weight*torch.sum(torch.pow(scale*(z_scale[b,:].double()-z_est_a_scale_ind),2))-prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                    prior_TO_anchor_sum = prior_TO_anchor_sum+prior_TO_temp
                elif opt.closest_anchor_flag == 1:
                    if arc_len < arc_len_min:
                        z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                        z_est_a_ind = torch.mul(z_est_a_scale_ind,scale) 
                
                        prior_TO_anchor_sum = torch.exp(-0.5*prior_weight*torch.sum(torch.pow(scale*(z_scale[b,:].double()-z_est_a_scale_ind),2))-prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                        #print('Change: arc orig: ' + str(arc_len_min) + ' arc new: ' + str(arc_len) + ' prior: ' + str(prior_TO_anchor_sum.detach().numpy()))
                        test = 1
                        arc_len_min = arc_len
                        anchor_idx_use[b] = a_idx
                #z_est_a_ind = transNet(torch.unsqueeze(a_mu_scale[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)

                #prior_TO_temp = torch.exp(-0.5*opt.prior_weight*torch.sum(torch.pow(z[b,:].double()-z_est_a_ind,2))-opt.prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                #prior_TO_anchor_sum = prior_TO_anchor_sum+prior_TO_temp
            if opt.closest_anchor_flag == 1:
                num_anchor_use = 1
            else:
                num_anchor_use = num_anchor
            log_p_z_no_add[b] = torch.log(prior_TO_anchor_sum/num_anchor_use)
            log_p_z[b] = log_p_z_no_add[b] + p_z_add
       
        
        LL_inner_no_add[n] = ((log_p_x_z_no_add + log_p_z_no_add.double() - log_q_z_x_no_add).logsumexp(-1) - np.log(k)).detach().numpy()
        LL_inner[n] = ((log_p_x_z + log_p_z.double() - log_q_z_x).logsumexp(-1) - np.log(k)).detach().numpy()
        time_save[n] = time.time()-batch_time_start
        print ("LL: [Test Sample %d/%d] time: %4.4f" % (n, batch_size_use,time.time()-batch_time_start))
        sio.savemat(save_folder + 'testMetrics_batch' + str(opt.batch_size) + '_' + str(k) + 'samp_startPt' + str(opt.startPt) + '_progress.mat',{'LL_inner':LL_inner,'LL_inner_no_add':LL_inner_no_add,'step':n,'time_save':time_save});
    LL_total = np.mean(LL_inner)
    LL_total_no_add = np.mean(LL_inner_no_add)                                          
    return LL_total, LL_inner,LL_total_no_add, LL_inner_no_add

def test_metrics(encoder,decoder,transNet,sampler_c,Psi,x,sample_labels,anchors,to_noise_std,num_anchor,M,numRestart,opt,mse_loss_sum,latent_mse_loss,scale):
    '''
    Compute log_likelihood 
    
    Inputs:
        - encoder:          Encoder network
        - decoder:          Decoder network
        - transNet:         Transport operator layer
        - sampler_c:        Layer that samples the transport operator coefficients
        - Psi:              Current transport operator dictionary elements
        - x:                Batch of data [batch_size,H,W,C]
        - labels:           Batch of labels [batch_size,y_dim]
        - to_noise_std:     Sampling noise standard deviation for Gaussian prior distribution
        - num_anchor:       Number of anchors per class
        - M:                Number of transport operator dictionary elements
        - numRestart:       Number of restarts for coefficient inference
        - opt:              Set of parameters
        - mse_loss_sum:     MSE loss function definition for output data
        - latent_mse_loss:  MSE loss function used for comparing the latent vectors transformed by transport operators
        - scale:            Value to scale the latent vectors by to get them in a range that is suitable for coefficient inference

        
        
    Outputs:
        - ELBO:             ELBO Computed without added constants
        - MSE:              Mean squared error between the input data and reconstructed data outputs
        
    '''
    test_size = 50
    
    a_mu= encoder(anchors)
    a_mu_scale = torch.div(a_mu,scale)
    a_mu_scale_np = a_mu_scale.detach().numpy()
    
    batch_size = x.shape[0]
    input_h = opt.img_size
    batch_idxs = batch_size // test_size
    MSE_total = np.zeros((batch_idxs))
    ELBO_total = np.zeros((batch_idxs))
    for idx in xrange(0, batch_idxs):
        batch_time_start = time.time()
        X_use = x[idx*test_size:(idx+1)*test_size]
        X_use = X_use.float()
        
        z_mu = encoder(X_use)
        z_mu_scale = torch.div(z_mu,scale)
        z_mu_scale_np = z_mu_scale.detach().numpy()

        # Sample the coefficients 
        z_coeff = sampler_c(test_size,M,opt.post_l1_weight)
        z_scale = transNet(z_mu_scale.double(),z_coeff.double(),Psi,to_noise_std)
        z = torch.mul(z_scale,scale)
        z_scale_np = z_scale.detach().numpy()
        
        # Compute the reconstruction loss
        MSE = 0.5*mse_loss_sum(decoder(z.float()).double(),X_use.double())/test_size
        MSE_new = torch.sum(torch.mean((decoder(z.float()).double().reshape(test_size,-1)-X_use.double().reshape(test_size,-1))**2,1),0)/test_size
        MSE_total[idx] = MSE_new.detach().numpy()
        # Compute the posterior loss function
        # Infer the coefficients between z_mu and z
        c_est_mu = np.zeros((test_size,M))
        for b in range(0,test_size):
            x0 = z_mu_scale_np[b,:].astype('double')
            x1 = z_scale_np[b,:].astype('double')
            c_est_mu[b,:],E,nit = infer_transOpt_coeff(x0,x1,Psi.detach().numpy().astype('double'),opt.post_cInfer_weight,0.0,1.0)
        # Transform mu with no noise     
        z_est_mu_scale = transNet(z_mu_scale.double(),torch.from_numpy(c_est_mu),Psi,0.0)
    
         
        post_TO_loss = (-0.5*opt.post_TO_weight*latent_mse_loss(scale*z_scale.double(),scale*z_est_mu_scale))/test_size
        post_l1_loss = -torch.sum(torch.abs(torch.from_numpy(c_est_mu)))/test_size
        
        
        
        # Compute the prior loss function
        prior_TO_sum = 0.0
        for b in range(0,test_size):
            
            x1 = z_scale_np[b,:].astype('double')
            prior_TO_anchor_sum = 0.0
            c_est_a = np.zeros((num_anchor,M))
            if sample_labels.ndim == 1:
                label_use = sample_labels[b] # Change this depending on the application
            else:
                label_use = np.where(sample_labels[b,:]==1)[0]
    
            anchors_use_np = a_mu_scale_np[int(num_anchor*label_use):int(num_anchor*(label_use+1)),:]
            a_mu_use = a_mu_scale[int(num_anchor*label_use):int(num_anchor*(label_use+1)),:]
            E_anchor= np.zeros((num_anchor,numRestart))
            arc_len_min = 1000000.0
            for a_idx in range(0,num_anchor):
                # Infer the coefficients between anchors and z
                
                x0 = anchors_use_np[a_idx,:].astype('double')
                E_single = np.zeros((numRestart))
                c_est_a_store = np.zeros((numRestart,M))
                for r_idx in range(0,numRestart):

                    rangeMin = -2.5 + r_idx*5
                    rangeMax = rangeMin + 2.5
                    c_est_a_store[r_idx,:],E_anchor[a_idx,r_idx],nit_anchor = infer_transOpt_coeff(x0,x1,Psi.detach().numpy().astype('double'),opt.prior_cInfer_weight,rangeMin,rangeMax)
                    E_single[r_idx] = E_anchor[a_idx,r_idx]
                minIdx = np.argmin(E_single)
                c_est_a[a_idx,:] = c_est_a_store[minIdx,:]
                c_est_a_ind = c_est_a[a_idx,:]
                arc_len = E_anchor[a_idx,minIdx]
                if opt.closest_anchor_flag == 0:
                    z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                    z_est_a_ind = torch.mul(z_est_a_scale_ind,scale) 
                    prior_TO_temp = torch.exp(-0.5*opt.prior_weight*torch.sum(torch.pow(scale*(z_scale[b,:].double()-z_est_a_scale_ind),2))-opt.prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                    prior_TO_anchor_sum = prior_TO_anchor_sum+prior_TO_temp
                elif opt.closest_anchor_flag == 1:
                    if arc_len < arc_len_min:
                        z_est_a_scale_ind = transNet(torch.unsqueeze(a_mu_use[a_idx,:].double(),0),torch.from_numpy(np.expand_dims(c_est_a_ind,axis =0)),Psi,0.0)
                        prior_TO_anchor_sum = torch.exp(-0.5*opt.prior_weight*torch.sum(torch.pow(scale*(z_scale[b,:].double()-z_est_a_scale_ind),2))-opt.prior_l1_weight*torch.sum(torch.abs(torch.from_numpy(c_est_a_ind))))
                        #print('Change: arc orig: ' + str(arc_len_min) + ' arc new: ' + str(arc_len) + ' prior: ' + str(prior_TO_anchor_sum.detach().numpy()))
                        arc_len_min = arc_len
            if opt.closest_anchor_flag == 1:
                num_anchor_use = 1
            else:
                num_anchor_use = num_anchor                
            #print(torch.log(prior_TO_anchor_sum/num_anchor).detach().numpy())
            prior_TO_sum = prior_TO_sum - torch.log(prior_TO_anchor_sum/num_anchor_use)
        print ("Metrics: [Test Sample %d/%d] time: %4.4f" % (idx, batch_idxs,time.time()-batch_time_start))
        prior_TO_sum = prior_TO_sum/batch_size
        ELBO = -1*(opt.recon_weight*MSE + post_TO_loss + opt.post_l1_weight*post_l1_loss + prior_TO_sum)
        ELBO_total[idx] = ELBO.detach().numpy()
    return MSE_total, ELBO_total
