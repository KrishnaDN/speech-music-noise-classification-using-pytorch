# Third Party
import librosa
import numpy as np
import torch
# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(audio_filepath, sr, win_length=8000,mode='train'):
    audio_data,fs  = librosa.load(audio_filepath,sr=16000)
    if mode == 'train':
        extended_wav = np.append(audio_data, audio_data)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
       
        return extended_wav
    else:
        extended_wav = np.append(audio_data, audio_data[::-1])
        return extended_wav



def load_data(filepath, win_length=8000, sr=16000, mode='train'):
    wav = load_wav(filepath, sr=sr, mode=mode)
    time = len(wav)
    #rint(time)
    randtime = np.random.randint(0, time-win_length)
    if randtime+win_length>=time:
        seg = wav[:win_length]
    else:
        seg = wav[randtime:randtime+win_length]
    #print(seg.shape)
    # preprocessing, subtract mean, divided by time-wise var
    ret_seg = seg.reshape([1,len(seg)])
    return ret_seg
    

