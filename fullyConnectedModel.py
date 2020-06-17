# -*- coding: utf-8 -*-

from __future__ import division
import torch.nn as nn


class Encoder(nn.Module):
 
    def __init__(self, x_dim, z_dim, M):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        :param M: number of transport operators
        """
        super(Encoder, self).__init__()


        self.model_enc = nn.Sequential(
            nn.Linear(int(x_dim),512),
            nn.ReLU(),
        )
        
        # compute mean and Laplacian weights
        self.fc_mean = nn.Linear(512, z_dim)
  
        

    def forward(self, x):
         # 2 hidden layers encoder
        x = self.model_enc(x)
        # compute mean and Laplacian weights
        z_mean = self.fc_mean(x)
        # The addition of 1e-10 prevents collapse

     
        return z_mean
    
class Decoder(nn.Module):

    def __init__(self, x_dim, z_dim):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        """
        super(Decoder, self).__init__()
       
        
        self.model = nn.Sequential(
            nn.Linear(z_dim,512),
            nn.ReLU(),
            nn.Linear(512,x_dim),
        )

    def forward(self, z):
        img= self.model(z)
        return img     