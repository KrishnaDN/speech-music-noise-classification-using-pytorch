#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:34:16 2019

@author: krishna
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
import librosa
#### Dataset info
num_epochs=2000
data_path_train='meta/training.txt'
data_path_test = 'meta/testing.txt'
#### Params



#video_path = 'temp.mp4'
#extract_audio = 'ffmpeg -i temp.mp4 -f wav temp.wav'
#os.system(extract_audio)


downsample = 'sox temp.wav -r 16k -c 1 temp_16k.wav'
os.system(downsample)
audio_file = 'temp_16k.wav'
######
read_audio,fs = librosa.load(audio_file,sr=16000)

def create_batch(read_audio,win_length=8000):
    total_num_seg = int(len(read_audio)/win_length)
    all_chunks = []
    for i in range(total_num_seg):
        seg_start = i*win_length
        seg_end = (i*win_length)+8000
        cut_audio = read_audio[seg_start:seg_end]
        all_chunks.append(torch.from_numpy(cut_audio).reshape([1,8000]))
    return all_chunks

segments = create_batch(read_audio, 8000)


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

checkpoint = torch.load('trained_models/check_point_1999')
model.load_state_dict(checkpoint['model'])


features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in segments]))
features = features.to(device)

preds = model(features)
np.argmax(preds[0].detach().cpu().numpy(),axis=1)
