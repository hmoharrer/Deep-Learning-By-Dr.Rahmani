# -*- coding: utf-8 -*-
"""Farsi_Book.ipynb
Written by Hossein.Moharrer.Derakhshandeh
DeepLearning Cource by Dr.Rahmani .
Date : 2022-06-30
RNN Text Generation Method .
"""

# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
import requests

# load ascii text and covert to lowercase
filename = "donyayenaaram.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
print (raw_text)
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: "), n_patterns

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
#tf.random.set_seed(42)
batch_size = 100
hidden_units = 500
n_epoch= 50
#dropout = 0.4
model = Sequential()
model.add(LSTM(hidden_units, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(hidden_units))
model.add(Dropout(0.4))
model.add(Dense(y.shape[1], activation='softmax'))
optimizer = optimizers.RMSprop(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)
#model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.summary())

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=1, mode='min')

# fit the model

#result_checker = ResultChecker(model, 1, 500)

'''model.fit(X, y, batch_size=batch_size, verbose=1, epochs=n_epoch,
                 callbacks=[result_checker, checkpoint, early_stop])'''
model.fit(X, y, epochs=n_epoch, batch_size=batch_size, callbacks=callbacks_list)

# Generate Text with Seed 

filename = "weights-improvement-50-0.6326-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1) 
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
for i in range(10000):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = x / float(n_vocab)
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = int_to_char[index]
  seq_in = [int_to_char[value] for value in pattern]
  with open('Output.txt', 'a', encoding='utf-8') as f:
    f.write(result)
  sys.stdout.write(result)
  #output=result
  #print(result)
  pattern.append(index)
  pattern = pattern[1:len(pattern)]
  #with open('quotes.txt', 'a', encoding='utf-8') as f:
  #  f.write(result)

print ("\nDone.")