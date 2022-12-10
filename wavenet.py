import numpy as np
import tensorflow as tf 
import tensorflow.keras.layers as layers
import soundfile as sf

def mu_law_decode(signal, quantization_channels):
    # Calculate inverse mu-law companding and dequantization
    mu = quantization_channels - 1
    # Map signal from [0, mu-1] to [-1, +1]
    signal = 2 * (signal.astype(np.float32) / mu) - 1
    signal = np.sign(signal) * (1.0 / mu) * ((1.0 + mu)**abs(signal) - 1.0)
    return signal

class Wavenet(tf.keras.Model):
    def __init__(self, timesteps = 32, output_dims = 256, quantization_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps
        self.output_dims = output_dims
        self.quantization_channels = quantization_channels
        self.model = tf.keras.Sequential([
            layers.Embedding(self.quantization_channels, 100, input_length=32, trainable=True),
            layers.Conv1D(64, 3, padding='causal', activation='relu'),
            layers.Dropout(0.2),
            layers.MaxPool1D(2),
            layers.Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'),
            layers.Dropout(0.2),
            layers.MaxPool1D(2),
            layers.Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'),
            layers.Dropout(0.2),
            layers.MaxPool1D(2),
            layers.GlobalMaxPool1D(),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.output_dims, activation='softmax'),
        ])

    def call(self, inputs):
        return self.model(inputs)

    def generate(self, generate_time, sampling_rate):
        mean = self.quantization_channels / 2
        std = mean * 0.909
        no_samples = generate_time * sampling_rate
        inputs = tf.random.normal((no_samples, self.timesteps), mean=mean, stddev=std, dtype=tf.float32)

        # forward pass:
        predicted_output = self.model.predict(inputs)
        print(f"Model prediction has shape {predicted_output.shape}")

        # generate predictions
        labels = np.argmax(predicted_output, axis=-1)
        print(f"Labels has shape {labels.shape}")

        # decode the predictions
        self.out = mu_law_decode(labels, self.quantization_channels)
        sf.write("generated.wav", self.out, sampling_rate)
        print("Finished generating audio")