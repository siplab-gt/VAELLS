# -*- coding: utf-8 -*-


from __future__ import division

import torch.nn as nn


class Encoder(nn.Module):
 
    def __init__(self, z_dim, c_dim,img_size):
        """
        Encoder initializer
        :param z_dim: dimension of the latent representation
        :param c_dim: channels in input images
        :param img_size: size of input image
        """
        super(Encoder, self).__init__()
        
        self.model_enc = nn.Sequential(
            nn.Conv2d(int(c_dim), 64, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ZeroPad2d((1,2,1,2)),
            nn.Conv2d(64, 64, 4, stride=1, padding=0),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(int(64*img_size*img_size/16),z_dim)
        

    def forward(self, x):
         # 2 hidden layers encoder
        x = self.model_enc(x)
        x = x.view(x.size(0),-1)
        # compute mean and Laplacian weights
        z_mean = self.fc_mean(x)
     
        return z_mean
    
class Decoder(nn.Module):

    def __init__(self,z_dim,c_dim,img_size):
        """
        Decoder initializer
        :param z_dim: dimension of the latent representation
        :param c_dim: channels in input images
        :param img_size: size of input image
        """
        super(Decoder, self).__init__()
        self.img_4 = img_size/4
        self.fc = nn.Sequential(
                nn.Linear(z_dim,int(self.img_4*self.img_4*64)),
                nn.ReLU(),
                )
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d( 64, 64, 4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d( 64, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d( 64, int(c_dim), 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size()[0]
        temp_var = self.fc(z)
        temp_var = temp_var.view(batch_size,64,int(self.img_4),int(self.img_4))
        img= self.model(temp_var)
        return img       
