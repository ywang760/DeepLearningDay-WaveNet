import numpy as np
import librosa
import tensorflow as tf 
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from pydub import AudioSegment

sampleSize = 16000

quantization_channels = 256 #discretize the value to 256 numbers

def mu_law_encode(signal, quantization_channels=256):
    # Manual mu-law companding and mu-bits quantization
    mu = (quantization_channels - 1).asType(np.float32)
    # signal should be in [-1, +1]
    # minimum operation to deal with rare large amplitudes caused by resampling
    signal = np.minimum(np.abs(signal), 1.0)
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

# def onehot(a, mu=quantization_channels):
#     # TODO: not sure why need transpose here
#     return tf.transpose(tf.one_hot(a,mu))

# # TODO: copied directly from reader-pytorch.py, need to check
# def cateToSignal(output, quantization_channels=256,stage=0):
#     mu = quantization_channels - 1
#     if stage == 0:
#         # Map values back to [-1, 1].
#         signal = 2 * ((output*1.0) / mu) - 1
#         return signal
#     else:
#         magnitude = (1 / mu) * ((1 + mu)**np.abs(output) - 1)
#         return np.sign(output) * magnitude

class Preprocess:

    def __init__(self, time_steps, sampling_rate, datapath):
        self.time_steps = time_steps
        self.sampling_rate = sampling_rate
        self.datapath = datapath # datapath is the directory path that contains the .wav files
        self.inputs = []
        self.normalized = False

    def load_data(self):
        # transform mp3 to wav
        mp3_files = librosa.util.find_files(self.datapath, ext=['mp3'])
        for mp3_file in mp3_files:
            wav_file = mp3_file[:-4] + '.wav'
            sound = AudioSegment.from_mp3(mp3_file)
            sound.export(wav_file, format="wav")

        wav_files = librosa.util.find_files(self.datapath, ext=['wav'])

        if len(wav_files) == 0:
            raise FileNotFoundError("No .wav files found in the directory")

        for file in wav_files:
            # load the audio file, range from -1 to 1
            audio, sr = librosa.load(file, sr=self.sampling_rate, mono=True)
            self.normalized = True
            # convert the audio file to mono
            # audio = librosa.to_mono(audio)
            # normalize the audio file
            if not self.normalized:
                audio = audio / np.max(np.abs(audio))
            # discretize the audio file
            # audio = mu_law_encode(audio, quantization_channels)
            self.inputs.append(audio)

    # takes in a list of inputs, each is a long array
    def create_dataset(self):

        self.load_data()

        x = []
        y = []

        for input in self.inputs:
            for i in range(0, len(input) - self.time_steps):
                # preparing input and output sequences
                input_ = input[i:i + self.time_steps]
                output = input[i + self.time_steps]
                x.append(input_)
                y.append(output)
        
        x = np.array(x)
        y = np.array(y)
        
        x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_training, x_testing, y_training, y_testing


if __name__ == "__main__":
    preprocess = Preprocess(time_steps=32, sampling_rate=44100, datapath='audiotest/')
    x_training, x_testing, y_training, y_testing = preprocess.create_dataset()

    print(x_training.shape)
    print(x_testing.shape)
    print(y_training.shape)
    print(y_testing.shape)
