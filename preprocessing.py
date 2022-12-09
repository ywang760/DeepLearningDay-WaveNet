import numpy as np
import librosa
import tensorflow as tf 
from pydub import AudioSegment

def mu_law_encode(signal, quantization_channels):
    # Manual mu-law companding and mu-bits quantization
    mu = (quantization_channels - 1)
    # signal should be in [-1, +1]
    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    quantized_signal = ((signal + 1) / 2 * mu + 0.5)

    return list(quantized_signal)

class Preprocess:

    def __init__(self, time_steps, sampling_rate, datapath, quantization_channels=256):
        self.time_steps = time_steps
        self.sampling_rate = sampling_rate
        self.datapath = datapath # datapath is the directory path that contains the .wav files
        self.inputs = []
        self.quantization_channels = quantization_channels
        self.normalized = False

    def load_data(self):
        # transform mp3 to wav
        mp3_files = librosa.util.find_files(self.datapath, ext=['mp3'])
        print("Found {} mp3 files".format(len(mp3_files)))
        i = 0
        for mp3_file in mp3_files:
            wav_file = mp3_file[:-4] + '.wav'
            sound = AudioSegment.from_file(mp3_file, format="mp3")
            sound.export(wav_file, format="wav")
            i += 1
            print(f"Created {i} .wav files")

        wav_files = librosa.util.find_files(self.datapath, ext=['wav'])

        if len(wav_files) == 0:
            raise FileNotFoundError("No .wav files found in the directory")
        
        print("Found {} wav files".format(len(wav_files)))

        for file in wav_files:
            # load the audio file, range from -1 to 1
            audio, sr = librosa.load(file, sr=self.sampling_rate, mono=True)
            self.normalized = True

            # trim the audio file
            audio, _ = librosa.effects.trim(audio)
            
            # convert the audio file to mono
            audio = librosa.to_mono(audio)

            # normalize the audio file
            if not self.normalized:
                audio = audio / np.max(np.abs(audio))
                
            # discretize the audio file
            audio = mu_law_encode(audio, self.quantization_channels)
            self.inputs.append(audio)
    
        print("Finished loading data")

    # takes in a list of inputs, each is a long array
    def create_dataset(self):

        self.load_data()

        x = []
        y = []

        cnt = 0

        for input in self.inputs:
            for i in range(0, len(input) - self.time_steps):
                # preparing input and output sequences
                input_ = input[i:i + self.time_steps]
                output = input[i + self.time_steps]
                x.append(input_)
                y.append(output)
            cnt += 1
            print(f"Loaded {cnt} input data")
        
        x = np.array(x)
        y = np.array(y)
        y = tf.one_hot(y, self.quantization_channels)

        test_size = 0.2
        i = int(len(x) * test_size)
        x_tr = x[i:]
        x_test = x[:i]
        y_tr = y[i:]
        y_test = y[:i]
        
        print("Finished creating dataset")
        return x_tr, x_test, y_tr, y_test