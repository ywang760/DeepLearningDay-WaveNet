import numpy as np
# import tensorflow as tf 
import librosa

quantization_channels = 10

sample_rate = 16000
sampleSize = 16000

def randomcrop(sample, pad=0, output_size=sample_rate):
    print('randomcrop', np.random.get_state()[1][0])
    np.random.seed()
    x, y = sample['x'], sample['y']

    shrink = 0
    low = pad + shrink * sampleSize # 16000
    high = x.shape[-1] - sampleSize - pad - shrink * sampleSize
    print(f'low: {low}, high: {high}')
    startx = np.random.randint(low = low, high = high)
    print(f'startx: {startx}')
    x = x[startx - pad : startx + sampleSize + pad]
    print(f'x shape: {x.shape}')
    print(f'x: {x}')
    y = y[startx : startx + sampleSize]
    print(f'y shape: {y.shape}')
    print(f'y: {y}')
    l = np.random.uniform(0.25, 0.5)
    sp = np.random.uniform(0, 1 - l)
    step = np.random.uniform(-0.5, 0.5)
    ux = int(sp * sample_rate)
    lx = int(l * sample_rate)
    x[ux:ux + lx] = librosa.effects.pitch_shift(x[ux:ux + lx], sample_rate, n_steps=step)

    return {'x': x, 'y': y}

input = {'x': np.ones((2, 32000)), 'y': np.ones((32000,))}