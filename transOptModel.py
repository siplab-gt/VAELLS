# -*- coding: utf-8 -*-


from __future__ import division

import numpy as np
import scipy as sp
from torch.autograd import Function
from torch.nn.modules.module import Module

import torch

class TransOptFunction(Function):
    """ 
    Class that defines the transport operator layer forward and backward passes
    """
    @staticmethod
    def forward(ctx, input, Psi):
        """
        Apply the transformation matrix defined by coeff_use and Psi to the 
        input latent vector
        """
        N = np.int(np.sqrt(Psi.shape[0]))
        M = np.int(Psi.shape[1])
        batch_size = input.shape[0]
        ctx.save_for_backward(input,Psi)
        input_coeff,Psi= input.detach().numpy(), Psi.detach().numpy()
        input_np = input_coeff[:,0:N]
        coeff = input_coeff[:,N:]
        x1_est = np.zeros((batch_size,N))
        for b_idx in range(0,batch_size):
            x0_use = np.expand_dims(input_np[b_idx,:],axis=1)
            coeff_use = np.expand_dims(coeff[b_idx,:],axis=1)

            A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
            T = np.real(sp.linalg.expm(A))
            x1_est[b_idx,:] = np.dot(T,x0_use)[:,0]
            
        
        result = x1_est
        
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient on the transport operator dictionaries
        """
        input,Psi = ctx.saved_tensors
        input_coeff,Psi= input.detach().numpy(), Psi.detach().numpy()
        N = np.int(np.sqrt(Psi.shape[0]))
        M = np.int(Psi.shape[1])
        batch_size = input.shape[0]
        input_np = input_coeff[:,0:N]
        coeff = input_coeff[:,N:]
        grad_output = grad_output.detach().numpy()
        c_grad_total = np.zeros((batch_size,M))
        Psi_grad_total = np.zeros((np.int(N*N),M))
        grad_z0_total = np.zeros((batch_size,N))
        

        for b_idx in range(0,batch_size):

            x0_use = np.expand_dims(input_np[b_idx,:],axis = 1)
            coeff_use = np.expand_dims(coeff[b_idx,:],axis=1)
            A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
            T = np.real(sp.linalg.expm(A))
            grad_z1_use = np.expand_dims(grad_output[b_idx,:],axis =1)
            
            grad_z0_total[b_idx,:] = np.dot(np.transpose(T),grad_z1_use)[:,0];

            eig_out = np.linalg.eig(A)
            U = eig_out[1]
            D = eig_out[0]
            V = np.linalg.inv(U)
            V = V.T
            P = np.dot(np.dot(U.T,grad_z1_use),np.dot(x0_use.T,V))
            
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
            c_grad = np.dot(np.reshape(Q,-1,order='F'),Psi)
            c_grad_total[b_idx,:] = np.real(c_grad)
            
            Psi_grad_single = np.zeros((N*N,M))   
            for m in range(0,M):
                PsiGradTemp = np.zeros((N,N))
                for k in range(0,N):
                    for ii in range(0,N):
                        PsiGradTemp[k,ii] = np.real(Q[k,ii]*coeff[b_idx,m])
                Psi_grad_single[:,m] = np.reshape(PsiGradTemp,(N*N),order='F')
            Psi_grad_total = Psi_grad_total+Psi_grad_single
            
        grad_z_coeff = np.concatenate((grad_z0_total,c_grad_total),axis=1)
        return torch.from_numpy(grad_z_coeff).to(torch.double),torch.from_numpy(Psi_grad_total).to(torch.double)

class TransOpt(Module):
    def __init__(self):

        super(TransOpt, self).__init__()

    def forward(self, input_z,coeff,Psi,std):
        """
        Define forward pass of Transport Operator layer
        
        Input:
            - input_z:  Input latent vector
            - coeff:    Transport operator coefficients defining the transformation matrix
            - Psi:      Current transport operator dictoinary
            - std:      Noise std for posterior sampling (if needed)
        """
        input_z_coeff = torch.cat((input_z,coeff),dim = 1)
        z_noNoise = TransOptFunction.apply(input_z_coeff,Psi)
        eps = torch.randn_like(z_noNoise)*std
        z_out = z_noNoise + eps
        return z_out
