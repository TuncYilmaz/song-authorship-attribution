#Inspired by https://github.com/Jverma/cnn-text-classification-keras/blob/master/text_cnn.py


# pickle file function
import pickle
def writePickle(Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname, tname):
    filename = "pickle_vars/"+tname+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

# import necessary packages
print("..loading packages")

import numpy as np
import pandas as pd
import json
import tensorflow as tf

# import keras modules

print("..loading keras modules")
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, Concatenate
from keras.layers import Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Embedding
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers

# desing your parameters

labeltype = "artist"
nb_of_outputs = 120
typename = "sub_word/"
early_stopping = 10
batch_size = 30
nb_epochs = 40

# decide which sets will be used and the number of output units, depending on the label type
print('Loading x_label datasets...')

if label_type == 'artist':
  y_tr = y_tr_artist
  y_te = y_te_artist
  y_val = y_val_artist
  nb_of_outputs = 120
  x_tr = readPickle("x_tr_artist",typename)
  x_val = readPickle("x_val_artist",typename)
  x_te = readPickle("x_te_artist",typename)

elif label_type == 'genre':
  y_tr = y_tr_genre
  y_te = y_te_genre
  y_val = y_val_genre
  nb_of_outputs = 12
  x_tr = readPickle("x_tr_genre",typename)
  x_val = readPickle("x_val_genre",typename)
  x_te = readPickle("x_te_genre",typename)


# read label sets
print('Loading y_label datasets...')

y_tr_genre = readPickle("y_tr_genre",typename)
y_tr_artist = readPickle("y_tr_artist",typename)
y_val_genre = readPickle("y_val_genre",typename)
y_val_artist = readPickle("y_val_artist",typename)
y_te_genre = readPickle("y_te_genre",typename)
y_te_artist = readPickle("y_te_artist",typename)

# read the pretrained embeddings
embedding_weights = readPickle("embedding_weights",typename)

# create the model framework
def CreateModel(max_embedding_vector_dim, max_input_length, vocab_size, kernel_sizes, nb_of_kernels, nb_of_outputs):
  
  # Input Layer
  input_layer = Input(shape=(max_input_length,), name='input_layer', dtype='int64')
  
  # Embedding Layer
  # the size will be (h+1,k), where h is the unique number of words in out vocab (+1 for padding), and k is the embedding vector dimension (i.e 300)
  embedding_layer = Embedding(vocab_size + 1,
                            max_embedding_vector_dim,
                            input_length=max_input_length,
                            weights=[embedding_weights])
  embedded = embedding_layer(input_layer, name='embedding_layer')
  
  # Convolutional Layers with Max Pooling
  # following the Kim(2014) paper, there will be 100 kernels for each of the 3,4 and 5 window size variants; a total of 300 kernels
  
  embedded = Reshape((max_input_length, max_embedding_vector_dim, 1))(embedded) # ???? CHECK WHETHER WE REALLY NEED TO DO THAT
  convs = list() # an empty list to accumulate all the convolution outputs
  for k_size in kernel_sizes: # for i in [5,4,3]
    c = Conv2D(nb_of_kernels, (k_size, max_embedding_vector_dim), activation='relu', name='conv1d_layer_with_'+str(k_size)+'window_size')(embedded)
    c = MaxPooling2D((max_input_length - k_size + 1, 1), name = 'pool1d_layer_with_'+str(k_size)+'window_size')(c)
    convs.append(c)
  
  # concatenate all the convolution outputs
  conc = concatenate([c for c in convs])
  
  # flatten
  conc_f = Flatten()(conc)

  # dropout layer
  drop = Dropout(0.5, name = 'dropout_layer')(conc_f)

  # predictions
  preds = Dense(len(nb_of_outputs), activation='softmax', name = 'dense_layer')(drop)

  # build model
  model = Model(inputs = input_layer, outputs = preds)
  adadelta = optimizers.Adadelta()
        
  model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

  name = "max_length:_"+str(max_input_length)+"vocab_size:_"+vocab_size+"kernel_sizes:_"+str(kernel_sizes)+"kernelseach:_"+str(nb_of_kernels)

  return model, name

model, name = CreateModel(50, 2441, 10000, [5,4,3], 100, 120):


model.summary()
print('Fit model...')

if not early_stopping:
    callback = None
else:
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

history = model.fit(x_tr, y_tr,
          validation_data=(x_val, y_val), batch_size=batch_size, epochs=nb_epochs, shuffle=True, callbacks = [callback])

name = typename+"_"+labeltype+"_"+"B"+str(batch_size)+"E"+str(nb_epochs)+"ES"+str(early_stopping)+str(name)
    
print('Predicting...')
predictions = model.predict(x_te)
print(np.argmax(predictions, axis=-1))
print(np.argmax(y_te, axis=-1))

writePickle(predictions, "predictions/Predictions_"+str(name))

# show the accuracy of the trained model on test set
score = model.evaluate(x_te, y_te, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("Saving the model and its history...")
model.save("saved_models/"+str(name)+".keras")
with open('pickle_vars/history/'+str(name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
  

