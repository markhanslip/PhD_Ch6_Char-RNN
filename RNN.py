import torch
import torch.nn as nn
from torch.autograd import Variable
import unidecode
import random
from tqdm import tqdm
import time
import numpy as np
import string
import os
import threading
from NotesDict import notes_dict
import subprocess

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):

        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):

        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

class Trainer:

    def __init__(self, training_data:str, epochs=500, layers=3):

        self.data_file = training_data
        self.data = None
        self.data_len = None
        self.inp = None
        self.target = None
        self.chunk_len = int(96*1.5)
        self.hidden_size = 100
        self.batch_size = 10
        self.learning_rate = 0.01
        self.model = 'gru'
        self.n_epochs = epochs
        self.loss = None
        self.temp_loss = None
        self.n_layers = layers
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        self.decoder = None
        self.model_path = None

    def load_data(self):

        self.data = unidecode.unidecode(open(self.data_file).read())
        self.data_len = len(self.data)
        print('loaded data file')

    def convert_to_tensor(self):

        all_characters = string.digits + string.whitespace

        self.inp = torch.LongTensor(self.batch_size, self.chunk_len)
        self.target = torch.LongTensor(self.batch_size, self.chunk_len)
        for bi in range(self.batch_size):
            start_index = random.randint(0, self.data_len - self.chunk_len)
            end_index = start_index + self.chunk_len + 1
            chunk = self.data[start_index:end_index]

            inp_tensor = torch.zeros(len(chunk[:-1])).long()
            for c in range(len(chunk[:-1])):
                try:
                    inp_tensor[c] = all_characters.index(chunk[:-1][c])
                except:
                    continue
            self.inp[bi] = inp_tensor

            target_tensor = torch.zeros(len(chunk[1:])).long()
            for c in range(len(chunk[1:])):
                try:
                    target_tensor[c] = all_characters.index(chunk[1:][c])
                except:
                    continue
            self.target[bi] = target_tensor

        self.inp = Variable(self.inp)
        self.target = Variable(self.target)

        if self.cuda:
            self.inp = self.inp.cuda()
            self.target = self.target.cuda()

        print('loaded training set')

    def build_model(self):

        self.decoder = CharRNN(
            len(string.printable),
            self.hidden_size,
            len(string.printable),
            # model=self.model,
            n_layers=self.n_layers,
        )

        if self.cuda:
            self.decoder.cuda()

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        print('built RNN architecture')

    def train(self):

        start = time.time()
        all_losses = []
        loss_avg = 0

        print("Training for %d epochs..." % self.n_epochs)

        # enumerate might speed this up slightly, also don't really need the progress bar:

        for epoch in tqdm(range(1, self.n_epochs + 1)):

            hidden = self.decoder.init_hidden(self.batch_size)
            if self.cuda:
                hidden = hidden.cuda()
            self.decoder.zero_grad()
            self.loss = 0
            # enumerate might speed this up slightly:
            for c in range(self.chunk_len):
                output, hidden = self.decoder(self.inp[:,c], hidden)
                self.loss += self.loss_fn(output.view(self.batch_size, -1), self.target[:,c])

            self.loss.backward()
            self.optimizer.step()

            self.loss = self.loss.data / self.chunk_len
            loss_avg += self.loss

            if epoch % 10 == 0:
                print('[%s (%d %d%%) %.4f]' % ((time.time()-start), epoch, epoch / self.n_epochs * 100, self.loss))

    def save_model(self):

        self.model_path = os.path.splitext(os.path.basename(self.data_file))[0] + '.pth'
        torch.save(self.decoder, self.model_path)
        print('Saved as %s' % self.model_path)

class Generator:

    def __init__(self, model_path):
        self.model_path = model_path
        self.predicted = ''
        self.decoder = None
        self.samplelib = {}
        self.notes = []
        self.durs = []
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

    def load_model(self):
        self.decoder = torch.load(self.model_path)
        print('loaded trained model')

    def generate(self, prompt, predict_len, temperature):

        all_characters = string.digits + string.whitespace

        hidden = self.decoder.init_hidden(1)
        prompt_tensor = torch.zeros(len(prompt)).long()
        for c in range(len(prompt)):
            try:
                prompt_tensor[c] = all_characters.index(prompt[c])
            except:
                continue
        prompt_tensor = Variable(prompt_tensor.unsqueeze(0))

        if self.cuda:
            hidden = hidden.cuda()
            prompt_tensor = prompt_tensor.cuda()
        self.predicted = prompt

        # Use priming string to "build up" hidden state
        for p in range(len(prompt) - 1):
            _, hidden = self.decoder(prompt_tensor[:,p], hidden)

        next_input_tensor = prompt_tensor[:,-1]

        for p in range(predict_len):
            output, hidden = self.decoder(next_input_tensor, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = all_characters[top_i]
            self.predicted += predicted_char

            next_input_tensor = torch.zeros(len(predicted_char)).long()
            for c in range(len(predicted_char)):
                try:
                    next_input_tensor[c] = all_characters.index(predicted_char[c])
                except:
                    continue
            next_input_tensor = Variable(next_input_tensor.unsqueeze(0))

            if self.cuda:
                next_input_tensor = next_input_tensor.cuda()

        print(self.predicted)
        return self.predicted

    def parse_output(self):

        self.notes=[]
        self.durs=[]

        # trim end of predicted string if necessary

        if self.predicted[-1] == ' ':
            self.predicted = self.predicted[:-1]
        if self.predicted[-1] == ',':
            self.predicted = self.predicted[:-1]
        if self.predicted[-2:] == ', ':
            self.predicted = self.predicted[:-2]
        self.predicted = [int(float(idx)) for idx in self.predicted.split(', ')]

        # separate notes and durs
        for i in range(0, len(self.predicted), 2):
            self.notes.append(self.predicted[i])

        for i in range(1, len(self.predicted), 2):
            self.durs.append(self.predicted[i])

        # replace outliers
        for index, note in enumerate(self.notes):
            if note < 44 or note > 87:
                self.notes[index] = random.choice(range(44, 77))

        print('notes:', self.notes)
        print('durs:', self.durs)
        return self.notes, self.durs

    def parse_output_pitch_only(self):

        self.notes=[]

        # trim end of predicted string if necessary

        if self.predicted[-1] == ' ':
            self.predicted = self.predicted[:-1]
        if self.predicted[-1] == ' ':
            self.predicted = self.predicted[:-1]
        if self.predicted[-2:] == ' ':
            self.predicted = self.predicted[:-2]
        self.predicted = [int(float(idx)) for idx in self.predicted.split(' ')]

        # separate notes and durs

        self.notes = self.predicted

        # replace outliers
        for index, note in enumerate(self.notes):
            if note < 44 or note > 87:
                self.notes[index] = random.choice(range(44, 77))

        print('notes:', self.notes)
        return self.notes

    def pipe_to_lilypond(self, ly_file='CharRNN_output.ly', tempo_factor=1.5):

        rounds = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32])
        self.durs = np.array(self.durs)
        self.durs = (tempo_factor*100)/self.durs

        print(self.durs)

        x = np.subtract.outer(self.durs, rounds)
        y = np.argmin(abs(x), axis=1)
        # y = y.astype(int)
        self.durs = list(rounds[y])
        self.durs = [int(i) for i in self.durs]

        print(self.durs)

        for i, dur in enumerate(self.durs):

            if dur == 1.5:
                self.durs[i] = '2'
            elif dur == 3:
                self.durs[i] = '4'
            elif dur == 6:
                self.durs[i] = '8'
            elif dur == 12:
                self.durs[i] = '16'
            elif dur == 24:
                self.durs[i] = '32'
            else:
                self.durs[i] = str(int(self.durs[i]))

        print(self.durs)

        if not os.path.exists(ly_file):

            lilypond = open(ly_file, 'w')
            lilypond.write("\\" + "version " + '"2.22.1"')
            lilypond.write('\n')
            lilypond.write('\n')

            lilypond.write("\\" + "header {")
            lilypond.write('\n')
            lilypond.write('  title = ' + '"Char-RNN outputs"')
            lilypond.write('\n')
            lilypond.write("}")
            lilypond.write('\n')
            lilypond.write('\n')

            lilypond.write("global = {")
            lilypond.write('\n')
            lilypond.write('  ' + "\\" + "hide Staff.TimeSignature")
            lilypond.write('\n')
            # lilypond.write("  " + "\\" + "cadenzaOn")
            # lilypond.write('\n')
            lilypond.write("  " + "\\" + "omit Staff.BarLine")
            lilypond.write('\n')
            # lilypond.write("  " + "\\" + "hide Stem")
            # lilypond.write('\n')
            lilypond.write("}")
            lilypond.write('\n')
            lilypond.write('\n')

        elif os.path.exists(ly_file):
            lilypond = open(ly_file, "a+") # open in append + read mode
            lilypond.seek(0)
            data = lilypond.read(10000) # is this lines or chars?
            if len(data) > 0 :
                lilypond.write("\n")
                lilypond.write("\n")

        lilypond.write("\\" + 'new Staff {')
        lilypond.write("\n")
        lilypond.write("  <<")
        lilypond.write("\n")
        lilypond.write("    \global { ")

        for note, dur in zip(self.notes, self.durs):
            lilypond.write(notes_dict[note] + dur + ' ') # add new phrase

        lilypond.write('}')
        lilypond.write('\n')
        lilypond.write("  >>")
        lilypond.write('\n')
        lilypond.write('}')

        lilypond.close()

        time.sleep(0.1)

    def notate(self, ly_file='CharRNN_output.ly'):

        subprocess.Popen(['lilypond -f png {}'.format(ly_file)], shell=True).wait()

    def pipe_to_lilypond_pitch_only(self, ly_file='CharRNN_output.ly'):

        # self.notes = self.predicted[9:-1]
        # print(self.notes)
        #
        # for note in self.notes:
        #     if note < 44.0:
        #         note = 44.0

        if not os.path.exists(ly_file):

            lilypond = open(ly_file, 'w')
            lilypond.write("\\" + "version " + '"2.22.2"')
            lilypond.write('\n')
            lilypond.write('\n')

            lilypond.write("\\" + "header {")
            lilypond.write('\n')
            lilypond.write('  title = ' + '"Char-RNN outputs"')
            lilypond.write('\n')
            lilypond.write("}")
            lilypond.write('\n')
            lilypond.write('\n')

            lilypond.write("global = {")
            lilypond.write('\n')
            lilypond.write('  ' + "\\" + "hide Staff.TimeSignature")
            lilypond.write('\n')
            lilypond.write("  " + "\\" + "cadenzaOn")
            lilypond.write('\n')
            lilypond.write("  " + "\\" + "hide Stem")
            lilypond.write('\n')
            lilypond.write("  " + "\\" + "omit Staff.BarLine")
            lilypond.write('\n')
            # lilypond.write("  " + "\\" + "hide Stem")
            # lilypond.write('\n')
            lilypond.write("}")
            lilypond.write('\n')
            lilypond.write('\n')

        elif os.path.exists(ly_file):
            lilypond = open(ly_file, "a+") # open in append + read mode
            lilypond.seek(0)
            data = lilypond.read(10000) # is this lines or chars?
            if len(data) > 0 :
                lilypond.write("\n")
                lilypond.write("\n")

        lilypond.write("\\" + 'new Staff {')
        lilypond.write("\n")
        lilypond.write("  <<")
        lilypond.write("\n")
        lilypond.write("    \global { ")

        for note in self.notes:
            lilypond.write(notes_dict[note] + ' ') # add new phrase

        lilypond.write('}')
        lilypond.write('\n')
        lilypond.write("  >>")
        lilypond.write('\n')
        lilypond.write('}')

        lilypond.close()

        time.sleep(1.0)
