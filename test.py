import numpy as np
import tensorflow as tf 

def mu_law_encode(signal, quantization_channels):
    # Manual mu-law companding and mu-bits quantization
    mu = (quantization_channels - 1)
    # signal should be in [-1, +1]
    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    quantized_signal = ((signal + 1) / 2 * mu + 0.5)

    return list(quantized_signal)


signal = np.random.normal(0, 1, 1000000)
std = np.std(mu_law_encode(signal, 256))
print(std)