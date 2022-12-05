import numpy as np
import tensorflow as tf
import soundfile as sf

class Generate:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    # predicted_output is a 1-d tensor
    def generate(self, predicted_output):
        sf.write("generated.wav", predicted_output, self.sampling_rate)