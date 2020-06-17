#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Code for running batch models of dictionary VAE
'''
import numpy as np
import os
c_samp_vals = [1]
batch_size = [32]
recon_weight = [1.0]
post_l1_weight = [1.0]
post_TO_weight = [1.0]
prior_weight = [1.0]
prior_l1_weight = [0.1]
post_cInfer_weight = [0.000001]
prior_cInfer_weight = [0.000001,0.000005]
gamma = [0.0]

for c_num in c_samp_vals:
    for batch_num in batch_size:
        for z_idx  in zeta:
            for l1_idx in post_l1_weight:
                for TO_idx  in post_TO_weight:
                    for prior_idx in prior_weight:
                        for prior_l1_idx in prior_l1_weight:
                            for post_cInfer_idx in post_cInfer_weight:
                                for prior_cInfer_idx in prior_cInfer_weight:
                                    for gamma_idx in gamma:    
                                        for recon_idx in recon_weight:
                                            os.system("python VAELLS.py --batch_size " + str(batch_num) + " --c_samp " + str(c_num) + " --recon_weight " + str(recon_idx) + " --prior_weight " + str(prior_idx) + " --post_TO_weight " + str(TO_idx) + " --post_l1_weight " + str(l1_idx) + " --prior_l1_weight " + str(prior_l1_idx) +  " --post_cInfer_weight " + str(post_cInfer_idx) + " --prior_cInfer_weight " + str(prior_cInfer_idx) +" --gamma " + str(gamma_idx))
