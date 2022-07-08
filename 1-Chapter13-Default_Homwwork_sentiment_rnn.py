# -*- coding: utf-8 -*-
"""Chapter13_Sentiment_RNN.ipynb
    
- Written by Hossein.Moharrer.Derakhshandeh.
- DeepLearning Cource by Dr.Rahmani .
- Date : 2022-07-06 
    
"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print('Number of training samples:', len(y_train))
print('Number of positive samples', sum(y_train))

print('Number of test samples:', len(y_test))

print(X_train[0])

word_index = imdb.get_word_index()
index_word = {index: word for word, index in word_index.items()}

print([index_word.get(i, ' ') for i in X_train[0]])

review_lengths = [len(x) for x in X_train]

import matplotlib.pyplot as plt
plt.hist(review_lengths, bins=10)
plt.show()

maxlen =200 
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print('X_train shape after padding:', X_train.shape)
print('X_test shape after padding:', X_test.shape)

#Building a simple LSTM network
tf.random.set_seed(42)
model = models.Sequential()

embedding_size = 32
model.add(layers.Embedding(vocab_size, embedding_size))

model.add(layers.LSTM(50))

model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 64
n_epoch = 3

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epoch,
          validation_data=(X_test, y_test))

acc = model.evaluate(X_test, y_test, verbose = 0)[1]
print('Test accuracy:', acc)

#Stacking multiple LSTM layers
model = models.Sequential()
model.add(layers.Embedding(vocab_size, embedding_size))
model.add(layers.LSTM(50, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(50, dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

optimizer = optimizers.Adam(learning_rate=0.003)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 7
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epoch,
          validation_data=(X_test, y_test))

acc = model.evaluate(X_test, y_test, verbose=0)[1]

print('Test accuracy with stacked LSTM:', acc)