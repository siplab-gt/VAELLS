from __future__ import division

import scipy.misc
import numpy as np
import cv2
import scipy as sp
import os
from sklearn.datasets import make_swiss_roll
from scipy.optimize import minimize 



def create_circle_data(numPoints,noise_std,mapMat,r = 1):
    numCircle = r.shape[0]
    for k in range(0,numCircle):
      
        angles_use = np.random.uniform(0,2*np.pi,numPoints)
    
        x = np.expand_dims(r[k]*np.cos(angles_use),axis = 1)
        y = np.expand_dims(r[k]*np.sin(angles_use),axis = 1)
        
        if k == 0:
            samples_orig = np.concatenate((x,y),axis=1)
            labels = np.ones((numPoints))*k
        else:
            samples_orig = np.concatenate((samples_orig,np.concatenate((x,y),axis=1)),axis=0)
            labels = np.concatenate((labels,np.ones((numPoints))*k),axis=0)
        
        
    samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig))) 
    samples = samples + np.random.randn(samples.shape[0],samples.shape[1])*noise_std
    randIdx = np.random.permutation(numPoints*numCircle)
    samples = samples[randIdx,:]
    samples_orig = samples_orig[randIdx,:]
    labels = labels[randIdx]
    
    return samples, samples_orig,labels

def create_anchors_circle(numPoints,noise_std,mapMat,r = 1,rand_flag= 0):
    numCircle = r.shape[0]

    for k in range(0,numCircle):
        if rand_flag == 0:
            angles_use = np.arange(0,360,360/numPoints)*np.pi/180.0
        else:
            angles_use = np.random.uniform(0,2*np.pi,numPoints)
        
        x = np.expand_dims(r[k]*np.cos(angles_use),axis = 1)
        y = np.expand_dims(r[k]*np.sin(angles_use),axis = 1)

        if k == 0:
            samples_orig = np.concatenate((x,y),axis=1)
        else:
            samples_orig = np.concatenate((samples_orig,np.concatenate((x,y),axis=1)),axis=0)
        
        
        samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig)))
        samples = samples + np.random.randn(samples.shape[0],samples.shape[1])*noise_std
    
    return samples, samples_orig

def create_sphere_data(numPoints,noise_std,mapMat,r = 1):
    numCircle = r.shape[0]
    for k in range(0,numCircle):
      
        theta_use = np.random.uniform(0,2*np.pi,numPoints)
        phi_use = np.random.uniform(0,2*np.pi,numPoints)
        
        
        x = np.expand_dims(r[k]*np.sin(theta_use)*np.cos(phi_use),axis = 1)
        y = np.expand_dims(r[k]*np.sin(theta_use)*np.sin(phi_use),axis = 1)
        z = np.expand_dims(r[k]*np.cos(theta_use),axis=1)
        
        if k == 0:
            samples_orig = np.concatenate((x,y,z),axis=1)
            labels = np.ones((numPoints))*k
        else:
            samples_orig = np.concatenate((samples_orig,np.concatenate((x,y,z),axis=1)),axis=0)
            labels = np.concatenate((labels,np.ones((numPoints))*k),axis=0)
        
        
    samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig))) 
    samples = samples + np.random.randn(samples.shape[0],samples.shape[1])*noise_std
    randIdx = np.random.permutation(numPoints*numCircle)
    samples = samples[randIdx,:]
    samples_orig = samples_orig[randIdx,:]
    labels = labels[randIdx]
    
    return samples, samples_orig,labels

def create_anchors_sphere(numPoints,noise_std,mapMat,r = 1):
    numCircle = r.shape[0]
    
    for k in range(0,numCircle):
        theta_use = np.random.uniform(0,2*np.pi,numPoints)
        phi_use = np.random.uniform(0,2*np.pi,numPoints)
    
        x = np.expand_dims(r[k]*np.sin(theta_use)*np.cos(phi_use),axis = 1)
        y = np.expand_dims(r[k]*np.sin(theta_use)*np.sin(phi_use),axis = 1)
        z = np.expand_dims(r[k]*np.cos(theta_use),axis=1)

        if k == 0:
            samples_orig = np.concatenate((x,y,z),axis=1)
        else:
            samples_orig = np.concatenate((samples_orig,np.concatenate((x,y,z),axis=1)),axis=0)
        
        
        samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig))) 
        samples = samples + np.random.randn(samples.shape[0],samples.shape[1])*noise_std

    
    return samples, samples_orig

def create_swissRoll_data(numPoints,noise_std,mapMat):
    samples_orig, _ = make_swiss_roll(numPoints, noise_std)
                
    samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig))) + noise_std
    #samples = np.log(1+np.exp(samples))  
    randIdx = np.random.permutation(numPoints)
    samples = samples[randIdx,:]/20.0
    samples_orig = samples_orig[randIdx,:]/20.0
    labels = np.zeros((numPoints))
                                        
    return samples, samples_orig,labels

def create_swissRoll_2D_data(numPoints,tStd,noise_std,mapMat):
    tt = (3*np.pi/2)*(1+tStd*np.random.uniform(0.0,1.0,numPoints))
    x = np.expand_dims(np.multiply(tt,np.cos(tt)),axis = 1)
    y = np.expand_dims(np.multiply(tt,np.sin(tt)),axis = 1)
    samples_orig = np.concatenate((x,y),axis=1)            
    samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig))) 
    samples = samples + np.random.randn(samples.shape[0],samples.shape[1])*noise_std
    #samples = np.log(1+np.exp(samples))  
    randIdx = np.random.permutation(numPoints)
    samples = samples[randIdx,:]/10.0
    samples_orig = samples_orig[randIdx,:]/10.0
    labels = np.zeros((numPoints))
                                        
    return samples, samples_orig,labels

def create_anchors_swissRoll_2D(numPoints,tStd,noise_std,mapMat,rand_flag = 0):
    if rand_flag == 0:
        tt = (3*np.pi/2)*(1+tStd*np.linspace(0.1,0.9,numPoints))
    else:
        tt = (3*np.pi/2)*(1+tStd*np.random.uniform(0.0,1.0,numPoints))

    x = np.expand_dims(np.multiply(tt,np.cos(tt)),axis = 1)
    y = np.expand_dims(np.multiply(tt,np.sin(tt)),axis = 1)
    samples_orig = np.concatenate((x,y),axis=1)            
    samples = np.transpose(np.matmul(mapMat,np.transpose(samples_orig))) 
    samples = samples + np.random.randn(samples.shape[0],samples.shape[1])*noise_std
    #samples = np.log(1+np.exp(samples))  
    randIdx = np.random.permutation(numPoints)
    samples = samples[randIdx,:]/10.0
    samples_orig = samples_orig[randIdx,:]/10.0
    labels = np.zeros((numPoints))

    return samples, samples_orig,labels

def load_mnist(data_type,y_dim=10):
        data_dir = os.path.join("/storage/home/hcoda1/6/mnorko3/p-crozell3-0/rich_project_pf1/", 'mnist')

        print(os.path.join(data_dir,'train-images-idx3-ubyte'))
        fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
    
        fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
    
        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        if data_type == "train":
            X = trX[0:50000,:,:,:]
            y = trY[0:50000].astype(np.int)
        elif data_type == "test":
            X = teX
            y = teY.astype(np.int)
        elif data_type == "val":
            X = trX[50000:60000,:,:,:]
            y = trY[50000:60000].astype(np.int)
            
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        y_orig = y 
        y_vec = np.zeros((len(y), y_dim), dtype=np.float)
        for i, label in enumerate(y):
          y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec,y_orig 
    
def load_mnist_classSelect(data_type,class_use,newClass):
    
    X,Y_vec, Y  = load_mnist(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    Y_out = Y_vec[class_idx_total,:]
    return X,Y_out
    

    
def select_mnist_anchors(data_X,data_y,num_anchor):
    data_shape = data_X.shape
    anchor_images = np.zeros((num_anchor*10,)+data_shape[1:])
    anchor_labels = np.zeros((num_anchor*10,10))
    for k in range(0,10):
        idxClass = np.where(data_y[:,k] == 1)[0]
        numEx = len(idxClass)
        idxChoice = idxClass[np.random.randint(low = 0, high = numEx,size=num_anchor)]
        anchor_images[k*num_anchor:(k+1)*num_anchor,:,:,:] = data_X[idxChoice,:,:,:]
        anchor_labels[k*num_anchor:(k+1)*num_anchor,:] = data_y[idxChoice,:]
    return anchor_images,anchor_labels
        
        
        
    
def transform_image(input_data,labels,class_transform,input_size,maxAng,numCopy):
    batch_size = input_data.shape[0]
    input_h = input_size
    input_w = input_size
    c_dim = input_data.shape[3]
    imgOut = np.zeros((batch_size*numCopy,input_h,input_w,c_dim))
    angOut = np.zeros((batch_size*numCopy))
    counter = 0
    for k in range(0,batch_size):
        for m in range(0,numCopy):
            imgTemp = np.pad(input_data[k,:,:,0],((2,2),(2,2)),'constant',constant_values=((0, 0),(0, 0)))
            classUse = np.where(labels[k,:] != 0)[0]
            img_h = imgTemp.shape[0]
            img_w = imgTemp.shape[1]
            
            class_check = np.in1d(classUse,class_transform)
            if class_check: 
                angle_use = np.random.rand(1)*maxAng
            else:
                angle_use = 0
            angOut[counter] = angle_use
            M = cv2.getRotationMatrix2D((img_h/2,img_w/2),angle_use,1)
            imgOut[counter,:,:,:] = np.expand_dims(cv2.warpAffine(imgTemp,M,(img_h,img_w)),axis=2)
            counter += 1
        
    return imgOut,angOut

def transform_image_specificAng(input_data,input_size,angUse):
    batch_size = input_data.shape[0]
    input_h = input_size
    input_w = input_size
    c_dim = input_data.shape[3]
    imgOut = np.zeros((batch_size*len(angUse),input_h,input_w,c_dim))
    angOut = np.zeros((batch_size*len(angUse)))
    counter = 0
    for k in range(0,batch_size):
        for ang_idx in angUse:
            imgTemp = np.pad(input_data[k,:,:,0],((2,2),(2,2)),'constant',constant_values=((0, 0),(0, 0)))
            img_h = imgTemp.shape[0]
            img_w = imgTemp.shape[1]
            angle_use = ang_idx
            angOut[counter] = angle_use
            M = cv2.getRotationMatrix2D((img_h/2,img_w/2),angle_use,1)
            imgOut[counter,:,:,:] = np.expand_dims(cv2.warpAffine(imgTemp,M,(img_h,img_w)),axis=3)
            counter += 1
    return imgOut,angOut
        
        
def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  #return images
  #return np.add(images,1.)
  return (images+1.)/2.

def imsave(images, size, path):
  return sp.misc.imsave(path, merge(images, size))

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

        
    
    
