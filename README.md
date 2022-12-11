# Music Generation with WaveNet

This is an adaptation of [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499.pdf) for music generation using MIDI and WAV audio files. This is a final project for CSCI 1470 Deep Learning at Brown University, created by Yutong Wang, Zhijun Liu, Rosanna Zhao, and Shuyang Song.
A detailed description of the project can be found [here](https://devpost.com/software/music-generation-with-wavenet).

This model depends on TensorFlow 2, which can be installed with ```pip install tensorflow```.

# Usage

## Audio files in MIDI format
This section is adapted from [this repository](https://github.com/soumya997/Music-Generation-Using-Deep-Learning).
We modified it to become an easy-to-use object-oriented API. 

To run: ```python3 wavenet.py piano/ 128 100```
```piano/``` is the directory containing the MIDI files. 
```128``` is the batch size for training the model. 
```100``` is the number of epochs to train the model.

You might need to install the following packages:
```pip install music21```

Example MIDI files can be found in the ```piano/``` directory.
```final_project.ipynb``` is a Jupyter notebook that demonstrates how to use the API.
```example output.mid``` is an example of the output MIDI file.


## Audio files in WAV format
To run: use the ```final_project.ipynb``` notebook.

Possible parameters to adjust:
time_steps, number of samples per input
sampling_rate, number of samples per second
quantization_channels, number of possible values for each sample
batch_size, number of samples per batch
epochs, number of epochs to train the model

You might need to install the following packages:
```pip install ffmpeg```
```pip install pydub```
```pip install librosa```
```pip install soundfile```
```pip install matplotlib```

Example mp3 files can be found in the ```guzheng/``` directory. The code can convert mp3 files to WAV files.