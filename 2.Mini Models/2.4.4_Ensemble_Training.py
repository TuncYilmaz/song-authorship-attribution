import numpy as np
import pandas as pd
import json
import tensorflow as tf
import keras

# import keras modules
print("..loading keras modules")
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding, Concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping

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


label_type = 'artist'
#typename = "sub_word/"
typename = "character/"
label_dictionary = readPickle(label_type, "id2")
test_labels = readPickle(label_type,typename+"y_te_")

y_tr_genre = readPickle("y_tr_genre",typename)
y_tr_artist = readPickle("y_tr_artist",typename)
y_val_genre = readPickle("y_val_genre",typename)
y_val_artist = readPickle("y_val_artist",typename)
y_te_genre = readPickle("y_te_genre",typename)
y_te_artist = readPickle("y_te_artist",typename)


# decide which sets will be used and the number of output units, depending on the label type
if label_type == 'artist':
    y_tr = y_tr_artist
    y_te = y_te_artist
    y_val = y_val_artist
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

    
    

# start by loading the models that you prefer to stack together

model1 = keras.models.load_model('saved_models/char_artist_nbf112_[3, 3, 3, 3, 3, 3]_[9, 9, 9]_nbd1024_BS30_NBE80_ES20_SEED10.keras')
model2 = keras.models.load_model('saved_models/char_artist_nbf112_[3, 3, 3, 3, 3, 3]_[9, 9, 9]_nbd1024_BS30_NBE80_ES20_SEED20.keras')
model3 = keras.models.load_model('saved_models/char_artist_nbf112_[3, 3, 3, 3, 3, 3]_[9, 9, 9]_nbd1024_BS30_NBE80_ES20_SEED30.keras')

models = [model1, model2, model3]


# define stacked model from multiple member input models
def define_stacked_model(models):
    # update all layers in all models to not be trainable
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in models]
    # concatenate merge output from each model

    ensemble_outputs = [model.get_layer('ensemble_' + str(i+1) + '_' +'out_before_dropout').output for i,model in enumerate(models)]
    merge = Concatenate()(ensemble_outputs)
    z = Dropout(0.5)(Dense(1024, activation='relu')(merge))
    z = Dropout(0.5)(Dense(1024, activation='relu')(z))
    pred = Dense(120, activation='softmax', name='z')(merge)

    model = Model(inputs=ensemble_visible, outputs=pred)
    # compile
    learning_rate = 0.1
    decay_rate = 0.00015
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
    return model        

stacked_model = define_stacked_model(models)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

stacked_model.fit([x_tr for _ in range(len(stacked_model.input))], y_tr,
          validation_data=([x_val for _ in range(len(stacked_model.input))], y_val), batch_size=480, epochs=5, shuffle=True, callbacks = [callback])

print('Predicting...')
predictions = stacked_model.predict([x_te for _ in range(len(stacked_model.input))])
print(np.argmax(predictions, axis=-1))
print(np.argmax(y_te, axis=-1))


# show the accuracy of the trained model on test set
score = stacked_model.evaluate([x_te for _ in range(len(stacked_model.input))], y_te, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# evaluate component models one by one
for i, model in enumerate(models):
    _, acc = model.evaluate(x_te, y_te, verbose=0)
    print('Model', i+1,'Accuracy: %.3f' % acc)
    
    
