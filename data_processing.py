#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:27:49 2019

@author: Krishna
"""

import glob
import os
import soundfile as sf
import random


class_id = {'music':0,'speech':1,'noise':2}

test_data=[]
train_data=[]
###
root_data = '/Users/apple/Downloads/musan/'
all_folders = glob.glob(root_data+'/*/')
for class_folder in all_folders:
    class_name = class_folder.split('/')[-2]
    sub_folders = sorted(glob.glob(class_folder+'/*/'))
    per_class_files = []
    for sub_folder in sub_folders:
        all_files = sorted(glob.glob(sub_folder+'/*.wav'))
        for file_path in all_files:
            per_class_files.append(file_path)
    test_samples = random.sample(range(len(per_class_files)),int(len(per_class_files)*0.1))
    for i in test_samples:
        test_data.append(per_class_files[i])
    for i in range(len(per_class_files)):
        if i in test_samples:
            continue
        else:
            train_data.append(per_class_files[i])

#######
fid_train = open('training.txt','w')
fid_test = open('testing.txt','w')

for filepath in train_data:
    audio_data,fs = sf.read(filepath)
    dur = len(audio_data)/float(fs)
    if dur <0.5:
        continue
    class_name = filepath.split('/')[-3]
    to_write = filepath+' '+str(class_id[filepath.split('/')[-3]])
    fid_train.write(to_write+'\n')
fid_train.close()


for filepath in test_data:
    audio_data,fs = sf.read(filepath)
    dur = len(audio_data)/float(fs)
    if dur <0.5:
        continue
    class_name = filepath.split('/')[-3]
    to_write = filepath+' '+str(class_id[filepath.split('/')[-3]])
    fid_test.write(to_write+'\n')
fid_test.close()
