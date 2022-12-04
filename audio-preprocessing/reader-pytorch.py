import datetime
import h5py
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
import soundfile as sf

import tensorflow as tf 
import tensorflow.keras.layers as layers
import tensorflow_io as tfio


sampleSize = 16000
sample_rate = 16000  # the length of audio for one second
# sample_rate = 8000 # parameter for piano
normalized = False # parameter for piano

quantization_channels=256 #discretize the value to 256 numbers

def mu_law_encode(audio, quantization_channels=256):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1)*1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(int) #discretize to 0~255


def mu_law_decode(output, quantization_channels=256):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * ((output*1.0) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

def onehot(a,mu=quantization_channels):
    b = np.zeros((mu,a.shape[0]))
    b[a,np.arange(a.shape[0])] = 1
    return b

def cateToSignal(output, quantization_channels=256,stage=0):
    mu = quantization_channels - 1
    if stage == 0:
        # Map values back to [-1, 1].
        signal = 2 * ((output*1.0) / mu) - 1
        return signal
    else:
        magnitude = (1 / mu) * ((1 + mu)**np.abs(output) - 1)
        return np.sign(output) * magnitude


class Dataset(data.Dataset):
    def __init__(self, listx, rootx,pad, transform=None):
        self.rootx = rootx
        self.listx = listx
        self.pad=int(pad)
        #self.device=device
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        np.random.seed()
        namex = self.listx[index]

        h5f = h5py.File(self.rootx + str(namex) + '.h5', 'r')
        x = h5f['x'][:]

        # TODO: piano has these two lines
        # x, _ = sf.read('piano/piano{}.wav'.format(namex))
        # print('train piano{}.wav,train audio shape{},rate{}'.format(namex,x.shape,_))

        # TODO: y has these two lines commented out
        factor1 = np.random.uniform(low=0.83, high=1.0)
        x = x*factor1

        if normalized:
            xmean = x.mean()
            xstd = x.std()
            x = (x - xmean) / xstd

        x = mu_law_encode(x)

        x = torch.from_numpy(x.reshape(-1)).type(torch.LongTensor)
        #x = F.pad(y, (self.pad, self.pad), mode='constant', value=127)

        
        return namex,x.type(torch.LongTensor)


class RandomCrop(object):
    def __init__(self, pad,output_size=sample_rate):
        self.output_size = output_size
        self.pad=pad

    def __call__(self, sample):
        #print('randomcrop',np.random.get_state()[1][0])
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        x, y = sample['x'], sample['y']
        shrink = 0
        #startx = np.random.randint(self.pad + shrink * sampleSize, x.shape[-1] - sampleSize - self.pad - shrink * sampleSize)
        #print(startx)
        #x = x[startx - pad:startx + sampleSize + pad]
        #y = y[startx:startx + sampleSize]
        l = np.random.uniform(0.25, 0.5)
        sp = np.random.uniform(0, 1 - l)
        step = np.random.uniform(-0.5, 0.5)
        ux = int(sp * sample_rate)
        lx = int(l * sample_rate)
        # x[ux:ux + lx] = librosa.effects.pitch_shift(x[ux:ux + lx], sample_rate, n_steps=step)

        return {'x': x, 'y': y}


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']
        return {'x': torch.from_numpy(x.reshape(1, -1)).type(torch.float32),
                'y': torch.from_numpy(y.reshape(-1)).type(torch.LongTensor)}


class Testset(data.Dataset):
    def __init__(self, listx, rootx,pad,dilations1,device):
        self.rootx = rootx
        self.listx = listx
        self.pad = int(pad)
        self.device=device
        self.dilations1=dilations1
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]

        h5f = h5py.File(self.rootx + str(namex) + '.h5', 'r')
        x = h5f['x'][:]

        queue = []
        for i in self.dilations1:
            queue.append(torch.normal(torch.zeros(64,i),std=1).to(self.device))
            #queue.append(torch.zeros((64,i), dtype=torch.float32).to(self.device))

        x = mu_law_encode(x)

        x = torch.from_numpy(x.reshape(-1)).type(torch.LongTensor)
        #y = (torch.randint(0, 255, (self.field)).long())

        return namex,x,queue


        # TODO: the following code is for piano
        y, _ = sf.read('piano/piano{}.wav'.format(namex))
        #factor1 = np.random.uniform(low=0.83, high=1.0)
        #y = y*factor1

        if normalized:
            ymean = y.mean()
            ystd = y.std()
            y = (y - ymean) / ystd

        y = mu_law_encode(y)

        #y = torch.from_numpy(y.reshape(-1)).type(torch.LongTensor)
        print('test piano{}.wav,train audio shape{},rate{}'.format(namex, y.shape, _))
        y = torch.from_numpy(y.reshape(-1)[int(16000*1):]).type(torch.LongTensor)
        print("first second as seed")
        #y = torch.randint(0, 256, (100000,)).type(torch.LongTensor)
        #print("random init")
        #y = F.pad(y, (self.pad, self.pad), mode='constant', value=127)

        return namex,y