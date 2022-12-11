import sys
import numpy as np
import os
from music21 import *
from collections import Counter
from sklearn.model_selection import train_test_split


class Preprocess():
    def __init__(self):
        self.unique_x = None
        self.unique_y = None

    def read_midi(self, file):
        print("Loading Music File:", file)
        notes = []
        notes_to_parse = None
        # parsing a midi file
        midi = converter.parse(file)
        # grouping based on different instruments
        s2 = instrument.partitionByInstrument(midi)
        # Looping over all the instruments
        for part in s2.parts:
            # select elements of only piano
            if 'Piano' in str(part):
                notes_to_parse = part.recurse()
                # finding whether a particular element is note or a chord
                for element in notes_to_parse:
                    # note
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    # chord
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
        return np.array(notes)


    def read_all_midi(self, path):
        files = [i for i in os.listdir(path) if i.endswith(".mid")]
        notes_array = np.array([self.read_midi(path + i) for i in files])
        return notes_array


    def prepare(self, notes_array):
        notes_ = [element for note_ in notes_array for element in note_]
        # computing frequency of each note
        freq = dict(Counter(notes_))
        frequent_notes = [note_ for note_, count in freq.items() if count >= 50]

        new_music = []

        for notes in notes_array:
            temp = []
            for note_ in notes:
                if note_ in frequent_notes:
                    temp.append(note_)
            new_music.append(temp)

        new_music = np.array(new_music)
        no_of_timesteps = 32
        x = []
        y = []

        for note_ in new_music:
            for i in range(0, len(note_) - no_of_timesteps, 1):
                # preparing input and output sequences
                input_ = note_[i:i + no_of_timesteps]
                output = note_[i + no_of_timesteps]

                x.append(input_)
                y.append(output)

        x = np.array(x)
        y = np.array(y)
        unique_x = list(set(x.ravel()))
        self.unique_x = unique_x
        x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
        x_seq = []
        for i in x:
            temp = []
            for j in i:
                # assigning unique integer to every note
                temp.append(x_note_to_int[j])
            x_seq.append(temp)

        x_seq = np.array(x_seq)
        unique_y = list(set(y))
        self.unique_y = unique_y
        y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
        y_seq = np.array([y_note_to_int[i] for i in y])

        x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

        return x_tr, x_val, y_tr, y_val

    def get_unique(self):
        return self.unique_x, self.unique_y



from keras.layers import (Dense,Conv1D,Embedding,MaxPool1D,Dropout,GlobalMaxPool1D)
from keras.models import Sequential

class Wavenet():
    def __init__(self):
        self.no_of_timesteps = 32
        self.model = Sequential()

    def construct(self, len_x, len_y):
        self.model.add(Embedding(len_x, 100, input_length=32, trainable=True))
        self.model.add(Conv1D(64, 3, padding='causal', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPool1D(2))
        self.model.add(Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPool1D(2))
        self.model.add(Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPool1D(2))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(len_y, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.model.summary()

    def fit(self, x_tr, x_val, y_tr, y_val, batch_size=128, epochs=100):
        self.model.fit(np.array(x_tr), np.array(y_tr), batch_size=batch_size, epochs=epochs,
                  validation_data=(np.array(x_val), np.array(y_val)), verbose=1)


    def predict(self, x_val, unique_x):
        ind = np.random.randint(0, len(x_val) - 1)

        random_music = x_val[ind]

        predictions = []
        for i in range(10):
            random_music = random_music.reshape(1, self.no_of_timesteps)

            prob = self.model.predict(random_music)[0]
            y_pred = np.argmax(prob, axis=0)
            predictions.append(y_pred)

            random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
            random_music = random_music[1:]

        x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
        predicted_notes = [x_int_to_note[i] for i in predictions]

        return predicted_notes


class ToMidi():
    def __init__(self):
        pass

    def to_midi(self, prediction_output):
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:

            # pattern is a chord
            print(pattern)
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    cn = int(current_note)
                    new_note = note.Note(cn)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)

                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)

            # pattern is a note
            else:

                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 1
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='music.mid')



if __name__ == '__main__':
    """This is the main function"""
    """
    Run the line in the command window below:
    python3 wavenet.py piano/ 128 100
    """
    path = sys.argv[1]
    batch_size = int(sys.argv[2])
    epoch = int(sys.argv[3])
    pre = Preprocess()
    notes_array = pre.read_all_midi(path=path)
    x_tr, x_val, y_tr, y_val = pre.prepare(notes_array)
    unique_x, unique_y = pre.get_unique()
    model = Wavenet()
    model.construct(len(unique_x), len(unique_y))
    model.fit(x_tr, x_val, y_tr, y_val, batch_size, epoch)
    predicted_output = model.predict(x_val, unique_x)
    midi = ToMidi()
    midi.to_midi(predicted_output)
