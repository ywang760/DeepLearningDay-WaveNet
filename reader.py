import datetime
import h5py
import numpy as np

import torch
from torch.utils import data
import torch.nn.functional as F
import soundfile as sf
import librosa
import tensorflow as tf 
import tensorflow.keras.layers as layers

sampleSize = 16000
sample_rate = 16000  # the length of audio for one second
# sample_rate = 8000 # parameter for piano
normalized = False # parameter for piano

quantization_channels = 256 #discretize the value to 256 numbers

def mu_law_encode(signal, quantization_channels):
    # Manual mu-law companding and mu-bits quantization
    mu = (quantization_channels - 1).asType(np.float32)
    # signal should be in [-1, +1]
    # minimum operation to deal with rare large amplitudes caused by resampling
    signal = np.minimum(np.abs(signal), 1.0)
    # According to algorithm: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm 
    magnitude = np.log1p(mu * signal) / np.log1p(mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    quantized_signal = ((signal + 1) / 2 * mu + 0.5).astype(np.int32)

    return quantized_signal
    
def mu_law_decode(signal, quantization_channels):
    # Calculate inverse mu-law companding and dequantization
    mu = quantization_channels - 1
    # Map signal from [0, mu-1] to [-1, +1]
    signal = 2 * (signal.astype(np.float32) / mu) - 1
    signal = np.sign(signal) * (1.0 / mu) * ((1.0 + mu)**abs(signal) - 1.0)
    return signal

def onehot(a, mu=quantization_channels):
    # TODO: not sure why need transpose here
    return tf.transpose(tf.one_hot(a,mu))



# TODO: copied directly from reader-pytorch.py, need to check
def cateToSignal(output, quantization_channels=256,stage=0):
    mu = quantization_channels - 1
    if stage == 0:
        # Map values back to [-1, 1].
        signal = 2 * ((output*1.0) / mu) - 1
        return signal
    else:
        magnitude = (1 / mu) * ((1 + mu)**np.abs(output) - 1)
        return np.sign(output) * magnitude




