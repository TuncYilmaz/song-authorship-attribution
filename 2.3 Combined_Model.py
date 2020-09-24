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
#from __future__ import print_function
#from __future__ import division

import numpy as np
import pandas as pd
import json
import tensorflow as tf

# import keras modules
print("..loading keras modules")
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding, Concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping

'''---------------------------------------------------------------'''
''' read it from the character folder since for all model types y_label datasets are the same '''

# read important variables (datasets)
print('Loading y_label datasets...')
typename = "character/"
y_tr_genre = readPickle("y_tr_genre",typename)
y_tr_artist = readPickle("y_tr_artist",typename)
y_val_genre = readPickle("y_val_genre",typename)
y_val_artist = readPickle("y_val_artist",typename)
y_te_genre = readPickle("y_te_genre",typename)
y_te_artist = readPickle("y_te_artist",typename)


'''---------------------------------------------------------------'''


# a helper function that sets up parameters and builds layers of the architecture
def create_model(input_type, label_type, embedding_type, nb_filters, nb_dense_outputs, filters, batch_size, nb_epochs, early_stopping, pools, maxlen, vocab_size, embedding_dim):
    
    if input_type not in ["char", "sub_word", "word"]: # depending on the input structure of the model, we need either 'char', 'sub_word' or 'word' for this argument
        raise ValueError('argument "input_type" must be either "char", "sub_word" or "word"')
        
    if label_type not in ["genre", "artist"]: # depending on the output type of the model, we need either 'genre' or 'artist' for this argument
        raise ValueError('argument "label_type" must be either "genre" or "artist"')

    if embedding_type not in ["no_embedding", "pre_trained", "one_hot"]: # we use three types of embeddings
        raise ValueError('argument "embedding_type" must be either "no_embedding", "pre_trained" or "one_hot"')

    # we need to give a filter size list, which is a list of 6 consecutive integers
    if type(filters) != list:  
        raise ValueError('argument "filters" must be of type "list"; e.g. [7,5,3,3,3,3]')
    elif len(filters) != 6:
        raise ValueError('"filters" list must consist of 6 integers')
        
    if early_stopping != False and type(early_stopping) != int: # there is either early stopping applied with an integer value, or there is not early stopping used in the model
        raise ValueError('argument "early_stopping" must be either False (indicating the model does not use early stopping) or an integer value that is less than the "nb_epochs" argument')

    # we need to give a pooling size list, which is a list of 3 consecutive integers
    if type(pools) != list:  
        raise ValueError('argument "pools" must be of type "list"; e.g. [3,3,3]')
    elif len(pools) != 3:
        raise ValueError('"filter_kernels" list must consist of 3 integers')
        
    if input_type == "char":
        typename = "character/"
    elif input_type == "sub_word":
        typename = "sub_word/"
    elif input_type == "word":
        typename = "word/"
    else:
        raise ValueError('Model input type is given incorrectly!')
        
    # STANDARD INITIALIZATION
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
    
    # INPUT LAYER
    input_shape = (maxlen,)
    input_layer = Input(shape=input_shape, name='input_layer', dtype='int64')
    
    # EMBEDDING LAYER
    
    if embedding_type != 'no_embedding':
        embedding_weights = readPickle("embedding_weights",typename)
    
        embedding_layer = Embedding(vocab_size + 1,
                            embedding_dim,
                            input_length=maxlen,
                            weights=[embedding_weights])
        embedded = embedding_layer(input_layer)
    else: # in case we don't have an embedding layer
        embedded = input_layer
        
    # CONVOLUTIONAL LAYERS
    
    conv = Convolution1D(filters=nb_filters, kernel_size=filters[0], kernel_initializer=initializer,
                         padding='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(embedded)
    conv = MaxPooling1D(pool_size=pools[0])(conv)

    conv1 = Convolution1D(filters=nb_filters, kernel_size=filters[1], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=pools[1])(conv1)

    conv2 = Convolution1D(filters=nb_filters, kernel_size=filters[2], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv1)

    conv3 = Convolution1D(filters=nb_filters, kernel_size=filters[3], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv2)

    conv4 = Convolution1D(filters=nb_filters, kernel_size=filters[4], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=nb_filters, kernel_size=filters[5], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=pools[2])(conv5)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(nb_dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(nb_dense_outputs, activation='relu')(z))
    
    # decide which sets will be used and the number of output units, depending on the label type
    if label_type == 'artist':
        y_tr = y_tr_artist
        y_te = y_te_artist
        y_val = y_val_artist
        output_size = 120

    elif label_type == 'genre':
        y_tr = y_tr_genre
        y_te = y_te_genre
        y_val = y_val_genre
        output_size = 12


    # Output dense layer with softmax activation
    pred = Dense(output_size, activation='softmax', name='output')(z)

    model = Model(inputs=input_layer, outputs=pred)

    
    adam = Adam(lr=0.001)  # SGD below can also be tried
    #sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    name = str(input_type)+"_"+str(embedding_type)+"_"+str(label_type)+"_"+str(batch_size)+"batch_"+str(nb_epochs)+"epoch_"+str(early_stopping)+"Stop_"+"filters="+str(filters)+"_pools="+str(pools)+"_"+str(nb_filters)+"filters_"+str(nb_dense_outputs)+"dense_outputs_"+str(maxlen)+"length"
    
    return model, name, early_stopping, y_tr, y_val, y_te, batch_size, nb_epochs, typename

    
# create the model, model name and the early stopping option
print('Building the model...')

model, name, early_stopping, y_tr, y_val, y_te, batch_size, nb_epochs, typename = create_model(input_type = "sub_word", label_type = "artist", embedding_type = "pre_trained", nb_filters = 112, nb_dense_outputs = 2048, filters = [3, 3, 3, 3, 3, 3], batch_size = 30, nb_epochs = 40, early_stopping = 4, pools = [3,3,3], maxlen = 3674, vocab_size = 10000, embedding_dim = 50)


print('Loading training, validation and test inputs...') 
x_tr = readPickle("x_tr",typename)
x_val = readPickle("x_val",typename)
x_te = readPickle("x_te",typename)

model.summary()
print('Fit model...')

if early_stopping == False:
    callback = None
else:
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

history = model.fit(x_tr, y_tr,
          validation_data=(x_val, y_val), batch_size=batch_size, epochs=nb_epochs, shuffle=True, callbacks = [callback])
    
    
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
model.save("saved_models/"+str(name))
with open('pickle_vars/history/'+str(name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
#writePickle(history,"history/History_"+str(name))
