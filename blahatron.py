#this code is heavily based on the code by NeuralNine so I'm keeping the header intact
'''

Code by NeuralNine (c) 2020
Visit https://www.neuralnine.com/

Instagram: @neuralnine
YouTube: NeuralNine

'''

import random
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

text = open("model.txt", 'rb').read().decode(encoding='utf-8')

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=bool)
y = np.zeros((len(sentences),
              len(characters)), dtype=bool)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

model = load_model('blahatron')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#model = Sequential()
#model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
#model.add(Dense(len(characters)))
#model.add(Activation('softmax'))
#
#model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
print(model.summary())

model.fit(x, y, batch_size=128, epochs=50, verbose=1, callbacks=[tensorboard_callback])

model.save('blahatron')
