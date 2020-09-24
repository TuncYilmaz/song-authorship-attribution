
'''!!! GO BELOW TO FUNCTION CALL FOR CHANGING VARIABLES AND OUTPUTS !!!'''

# import necessary print packages
from __future__ import print_function
from __future__ import division

# pickle file function
import pickle
def writePickle(Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "pickle_vars/character/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

# necessary packages
print("..loading packages")
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
      
# keras modules
print("..loading keras modules")
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding, Concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping




# read important variables
print('Loading data and other variables...')
x_tr = readPickle("x_tr")
y_tr_genre = readPickle("y_tr_genre")
y_tr_artist = readPickle("y_tr_artist")
x_val = readPickle("x_val")
y_val_genre = readPickle("y_val_genre")
y_val_artist = readPickle("y_val_artist")
x_te = readPickle("x_te")
y_te_genre = readPickle("y_te_genre")
y_te_artist = readPickle("y_te_artist")
      
# Embedding layer Initialization
embedding_weights = readPickle("embedding_weights")

# np.random.seed(123)  # for reproducibility

# Whether to save model parameters
save = False
model_name_path = 'params/model_name.json'
model_weights_path = 'params/model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 11111 # this value can be changed in different versions

# Model params

# add parameters
vocab_size = 160
embedding_dim = 300
# for conv layers
nb_filter = 56
# Number of units in the dense layer
dense_outputs = 1024
# Conv layer kernel size
filter_kernels = [5, 5, 3, 3, 3, 3] #!!!! first pool size changed from 3 to 7


# Compile/fit params
batch_size = 120
nb_epoch = 40

'''!!!!!'''
# artist or genre? # if the model aims at artist labeling, the below value gets true
artist = True
if artist:
    y_tr = y_tr_artist
    y_te = y_te_artist
    y_val = y_val_artist
    output_size = 120
    name = "artist_model_"+str(filter_kernels[0])+"filter_size_"+str(nb_filter)+"filters_"+str(batch_size)+"batch_"+str(nb_epoch)+"epoch"
else:
    y_tr = y_tr_genre
    y_te = y_te_genre
    y_val = y_val_genre
    output_size = 12
    name = "genre_model_"+str(filter_kernels[0])+"filter_size_"+str(nb_filter)+"filters_"+str(batch_size)+"batch_"+str(nb_epoch)+"epoch"

# a helper function that sets up parameters and builds layers of the architecture
def create_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, output_size):
    
    # standart initialization
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

        
    # Input
    input_shape = (maxlen,)
    input_layer = Input(shape=input_shape, name='input_layer', dtype='int64')

    # Embedding
    embedding_layer = Embedding(vocab_size + 1,
                            embedding_dim,
                            input_length=11111,
                            weights=[embedding_weights])
    embedded = embedding_layer(input_layer)



    # All the convolutional layers...
    conv = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0], kernel_initializer=initializer,
                         padding='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(embedded)
    conv = MaxPooling1D(pool_size=7)(conv)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=3)(conv1)

    conv2 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[2], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv1)

    conv3 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[3], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv2)

    conv4 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[4], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[5], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    # Output dense layer with softmax activation
    pred = Dense(output_size, activation='softmax', name='output')(z)

    model = Model(inputs=input_layer, outputs=pred)

    #sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.001)  # Feel free to use SGD above. I found Adam with lr=0.001 is faster than SGD with lr=0.01
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

print('Building the model...')
model = create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                              nb_filter, output_size)


model.summary()
print('Fit model...')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
model.fit(x_tr, y_tr,
          validation_data=(x_val, y_val), batch_size=batch_size, epochs=nb_epoch, shuffle=True, callbacks = [callback])

# save the predictions on the test set
predictions = model.predict(x_te)
print(np.argmax(predictions, axis=-1))
print(np.argmax(y_te, axis=-1))

writePickle(predictions, "predictions/predictions_"+str(name))

# show the accuracy of the trained model on test set
score = model.evaluate(x_te, y_te, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("saved_models/characters_"+str(name))
