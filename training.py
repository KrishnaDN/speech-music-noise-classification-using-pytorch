#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:45:06 2019

@author: Krishna
"""

import torch
import numpy as np


from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os 
import numpy as np
from torch import optim
from model.multi_scale_1d_CNN import MSResNet
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
#### Dataset info
num_epochs=2000
data_path_train='meta/training.txt'
data_path_test = 'meta/testing.txt'
#### Params



def speech_collate(batch):
    targets = []
    specs = []
    for sample in batch:
        specs.append(sample['audio_wav'])
        targets.append((sample['labels']))
    return specs, targets

### Data related
dataset_train = SpeechDataGenerator(manifest=data_path_train)
dataloader_train = DataLoader(dataset_train, batch_size=150,num_workers=12, shuffle=True,collate_fn=speech_collate) 

dataset_test = SpeechDataGenerator(manifest=data_path_test)
dataloader_test = DataLoader(dataset_test, batch_size=150,num_workers=12,shuffle=True,collate_fn=speech_collate)
## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = MSResNet(input_channel=1,num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    train_loss_list = []
    train_acc_list =[]
    total = 0.
    correct = 0.
    for i in range(10):
        for i_batch, sample_batched in enumerate(dataloader_train):
            #print(sample_batched)
            
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
            labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
            
            features, labels = features.to(device), labels.to(device)
            features.requires_grad = True
            optimizer.zero_grad()
            preds = model(features)
            total_loss = loss(preds, labels.squeeze())
            total_loss.backward()
            
            optimizer.step()
            #prediction = np.argmax(preds.detach().cpu().numpy(),axis=1)
            
            accuracy = accuracy_score(labels.detach().cpu().numpy(),np.argmax(preds.detach().cpu().numpy(),axis=1))
            train_loss_list.append(total_loss.item())
            train_acc_list.append(accuracy)
    mean_loss = np.mean(np.asarray(train_loss_list))
    mean_acc = np.mean(np.asarray(train_acc_list))
    
    print('********* Loss {} and Accuracy {} after {} epoch '.format(mean_loss,mean_acc,epoch))
    
    model_save_path = os.path.join('trained_models_new', 'check_point_'+str(epoch))
    state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
    torch.save(state_dict, model_save_path)
    
    
    cum_acc=0.0
    test_acc_list=[]
    for i in range(10):
        for i_batch, sample_batched in enumerate(dataloader_test):
            #print(sample_batched)
            
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
            labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
            
            features, labels = features.to(device), labels.to(device)
            features.requires_grad = True
            optimizer.zero_grad()
            preds = model(features)
            total_loss = loss(preds, labels.squeeze())
            total_loss.backward()
            
            optimizer.step()
            #prediction = np.argmax(preds.detach().cpu().numpy(),axis=1)
            accuracy = accuracy_score(labels.detach().cpu().numpy(),np.argmax(preds.detach().cpu().numpy(),axis=1))
            test_acc_list.append(accuracy)
    mean_test_acc = np.mean(np.asarray(test_acc_list))   
    print('********* Final test accuracy {} after {} '.format(mean_test_acc,epoch))
    
    
    
    
    
