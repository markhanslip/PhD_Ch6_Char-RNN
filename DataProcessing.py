import numpy as np
from numpy import inf
import parselmouth
import os
import time

class DataProcessing:

    def __init__(self, audio_file, output_file):

        self.audio_file = audio_file
        self.output_file = output_file
        self.audio_data = None
        self.pitches = None
        self.onsets = None
        self.output_data = None

    def load_audio(self):

        self.audio_data = parselmouth.Sound(self.audio_file)
        print('audio loaded')

    def get_freqs(self):

        self.pitches = self.audio_data.to_pitch_ac(time_step=0.01, pitch_floor=50.0, pitch_ceiling=1400.0) # check this doesn't need a sr arg
        self.pitches = self.pitches.selected_array['frequency']
        self.pitches[self.pitches==0] = np.nan
        self.pitches = list(self.pitches)
        self.pitches = np.nan_to_num(self.pitches)
        print('extracted freqs')
        # print(self.pitches)
        # print(max(self.pitches))

    def freqs_to_MIDI(self):

        self.pitches = 12*np.log2(self.pitches/440)+69
        self.pitches[self.pitches == -inf] = 0
        self.pitches = np.around(self.pitches, decimals=1)
        print('converted freqs to MIDI')
        print(self.pitches)
        # print(max(self.pitches))
        # count zero values and replace with a single duration value:

    def get_onsets(self):

        # work out which note values represent an onset and multiply the two vectors
        temp_onsets = np.ediff1d(self.pitches) #or d = diff(midi)

        temp_onsets = (temp_onsets <= -0.8) & (temp_onsets >= -44) | (temp_onsets >= 0.8)
        temp_onsets = temp_onsets.astype(int)

        # replace consecutive onsets with 0:
        temp_onsets = list(temp_onsets)
        #print('temp onsets:', temp_onsets)
        self.onsets=[]
        for i, n in enumerate(temp_onsets):
            if n == 0:
                self.onsets.append(n)
            if n == 1 and temp_onsets[i+1] == 0:
                self.onsets.append(temp_onsets[i])
            if n == 1 and temp_onsets[i+1] == 1:
                self.onsets.append(0)

        self.onsets = np.insert(self.onsets, 0, 0)
        #print('onsets', self.onsets)
        self.pitches = self.onsets * self.pitches
        print(self.pitches)
        print(max(self.pitches))
        nz = np.flatnonzero(self.pitches)
        if max(self.pitches) > 44:
            self.pitches= self.pitches[nz[0]:] # this threw error
            print('extracted onsets')
        else:
            pass
        return self.pitches
        print(self.pitches)

    def remove_zeros_for_pitches_only(self):

        self.output_data = self.pitches[self.pitches!=0]
        print("after zeros removed:", self.output_data)

    def add_durations(self):

        self.output_data=[]
        count_0 = 0
        count_1 = 0

        if max(self.pitches) > 0:

            for i in range(len(self.pitches)-1):

                if self.pitches[i] > 1:
                    self.output_data.append(self.pitches[i])

                if self.pitches[i] == 1 and self.pitches[i+1] == 1:
                    count_1 += 1

                if self.pitches[i] == 1 and self.pitches[i+1] > 1:
                    count_1 += 1
                    self.output_data.append(count_1)
                    count_1 = 0
                    self.output_data.append(0)

                if self.pitches[i] == 1 and self.pitches[i+1] == 0:
                    count_1 += 1
                    self.output_data.append(count_1)
                    count_1 = 0

                if self.pitches[i] == 0 and self.pitches[i+1] == 0:
                    count_0 += 1

                if self.pitches[i] == 0 and self.pitches[i+1] > 1:
                    count_0 += 1
                    self.output_data.append(count_0)
                    count_0 = 0

            self.output_data = self.output_data[:-1] # remove last val to preserve format
            self.output_data = np.clip(self.output_data, 10, 99)
            self.output_data = list(self.output_data)
            return self.output_data
            print('done parsing audio data')

        else:
            pass

    def augment_data(self):

        aug_array = self.output_data.copy()
        orig_array = self.output_data.copy()

        transpositions = [1, 2]

        for transposition in transpositions:
            for i, n in enumerate(aug_array):
                if i % 2 == 0:
                    n = n+transposition
            self.output_data=self.output_data+aug_array
            # print(len(p))
            aug_array = orig_array.copy()

        # self.output_data = np.clip(self.output_data, 10, 99) #
        self.output_data = list(self.output_data)
        print('augmented data')

    def float2int(self):

        print(type(self.output_data))

        for index, num in enumerate(self.output_data):
            self.output_data[index] = int(num)

        print("converted floats to integers for SeqGAN")

    def reshape_data_for_seqGAN(self):

        self.output_data = self.output_data.tolist()

        while len(self.output_data) % 12 != 0:
            self.output_data = self.output_data[:-1]

    def write2file(self):

        self.output_data = np.array(self.output_data)

        # self.output_data = self.output_data.reshape((int(len(self.output_data)/12), 12))
        np.savetxt(self.output_file, self.output_data, fmt='%2u', delimiter=' ', newline='\n')
        print('saved output data to', self.output_file)
        # else:
            # pas

    def pitches2file(self):

        np.savetxt(self.output_file, [self.output_data], fmt='%2u', delimiter=' ')
        print('saved output data to', self.output_file)
