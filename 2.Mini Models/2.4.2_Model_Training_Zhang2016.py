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
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding, Concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
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
def create_model(seed, input_type, label_type, nb_filters, nb_dense_outputs, filters, batch_size, nb_epochs, early_stopping, pools, maxlen, vocab_size, embedding_dim):        
    if input_type == "char":
        typename = "character/"
    elif input_type == "sub_word":
        typename = "sub_word/"
    elif input_type == "word":
        typename = "word/"
    else:
        raise ValueError('Model input type is given incorrectly!')
        
    # STANDARD INITIALIZATION
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    
    # INPUT LAYER
    input_shape = (maxlen,)
    input_layer = Input(shape=input_shape, name='input_layer', dtype='int64')
    
    # EMBEDDING LAYER
    

    embedding_weights = readPickle("embedding_weights",typename)
    
    embedding_layer = Embedding(vocab_size + 1,
                            embedding_dim,
                            input_length=maxlen,
                            weights=[embedding_weights])
    embedded = embedding_layer(input_layer)

        
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
    conv5 = AveragePooling1D(pool_size=pools[2])(conv5)
    conv5 = Flatten(name='out_before_dropout')(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(nb_dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(nb_dense_outputs, activation='relu',name='out_before_softmax')(z))
    
    # decide which sets will be used and the number of output units, depending on the label type
    if label_type == 'artist':
        y_tr = y_tr_artist
        y_te = y_te_artist
        y_val = y_val_artist
        output_size = 120
        x_tr = readPickle("x_tr_artist",typename)
        x_val = readPickle("x_val_artist",typename)
        x_te = readPickle("x_te_artist",typename)

    elif label_type == 'genre':
        y_tr = y_tr_genre
        y_te = y_te_genre
        y_val = y_val_genre
        x_tr = readPickle("x_tr_genre",typename)
        x_val = readPickle("x_val_genre",typename)
        x_te = readPickle("x_te_genre",typename)
        output_size = 12
        

    # Output dense layer with softmax activation
    pred = Dense(output_size, activation='softmax', name='output')(z)

    model = Model(inputs=input_layer, outputs=pred)

    learning_rate = 0.051
    decay_rate = 0.00015
    momentum = 0.8
    adam = Adam(lr=learning_rate, decay=decay_rate)
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    
    #adam = Adam(lr=0.05)  # SGD below can also be tried
    #sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    name = str(input_type)+"_"+str(label_type)+"_nbf"+str(nb_filters)+"_"+str(filters)+"_"+str(pools)+"_nbd"+str(nb_dense_outputs)+"_BS"+str(batch_size)+"_NBE"+str(nb_epochs)+"_ES"+str(early_stopping)+"_SEED"+str(seed)+"_lastpoolavg_morelayers"
    
    return model, name, early_stopping, x_tr, x_val, x_te, y_tr, y_val, y_te, batch_size, nb_epochs, typename

    
# create the model, model name and the early stopping option
print('Building the model...')



'''choose your model type here: 'char' or 'sub_word' '''

m = 'sub_word'
#m = 'char' 

if m == 'char':
    maxlen = 8303
    vocab_size = 116
    embedding_dim = 300
elif m == 'sub_word':
    maxlen = 2441
    vocab_size = 10000
    embedding_dim = 50

    
model, name, early_stopping, x_tr, x_val, x_te, y_tr, y_val, y_te, batch_size, nb_epochs, typename = create_model(seed = 30, input_type = m, label_type = "artist", nb_filters = 112, nb_dense_outputs = 3072, filters = [3, 3, 3, 3, 3, 3], batch_size = 30, nb_epochs = 80, early_stopping = 40, pools = [9,9,9], maxlen = maxlen, vocab_size = vocab_size, embedding_dim = embedding_dim)


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
model.save("saved_models/"+str(name)+".keras")
with open('pickle_vars/history/'+str(name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
