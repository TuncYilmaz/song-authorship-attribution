import numpy as np

import pickle
def writePickle(Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "pickle_vars/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import string

def load_data():
    
    data = pd.read_csv('sub_dataset.csv', header=None)
    data = data.dropna()

    x = data[2]
    x = np.array(x)

    y_artist = data[0] - 1
    y_artist = to_categorical(y_artist)
    
    y_genre = data[1] - 1
    y_genre = to_categorical(y_genre)
    
    return (x, y_artist, y_genre)

x, y_artist, y_genre = load_data()

# Shuffle data
np.random.seed(23) # !!!!!! use the same seed value in all dataset split versions !!!!!!
shuffle_indices = np.random.permutation(np.arange(len(y_artist))) 
# shuffle all inputs with the same indices
x_shuffled = x[shuffle_indices]
y_artist_shuffled = y_artist[shuffle_indices]

# form count dictionaries for both artist and genre labels
# first, artist labels
artist_label_count_dict = dict()
for i in range(120):
    artist_label_count_dict[i] = 0
# then also for genre labels
genre_label_count_dict = dict()
for i in range(12):
    genre_label_count_dict[i] = 0
    
'''Now, we'll go through the data examples one by one. If each artist label occurs less than 80 times, the sample 
will belong to the training set; occurances between 81th and 90th times will go to the validation set and the 
occurances between 91th and 100th times (last 10 occurances) will go to the test set.
For genre labels, any genre label occuring less than 800 times will go to the training set; occurances between 801th and 
900th times will go to the validation set and the occurances between 901th and 1000th times (last 100 occurances) will 
go to the test set.'''

# create training, validation and test sets with equal distributions of artists and genres
x_tr_artist, x_val_artist, x_te_artist = list(), list(), list()
y_tr_artist, y_val_artist, y_te_artist = list(), list(), list()

for sample, art_label in zip(x_shuffled, y_artist_shuffled):
    artist_label_index = np.argmax(art_label, axis=-1)

    # for artist labels
    if artist_label_count_dict[artist_label_index] < 80:
        x_tr_artist.append(sample)
        y_tr_artist.append(art_label)
    elif 80 <= artist_label_count_dict[artist_label_index] < 90:
        x_val_artist.append(sample)
        y_val_artist.append(art_label)
    elif 90 <= artist_label_count_dict[artist_label_index] < 100:
        x_te_artist.append(sample)
        y_te_artist.append(art_label)
    else:
        print("There is an error with artist counts!")
    artist_label_count_dict[artist_label_index] += 1
             
# turn the output datasets in np arrays
x_tr_pho_artist = np.array(x_tr_artist)
x_val_pho_artist = np.array(x_val_artist)
x_te_pho_artist = np.array(x_te_artist)

y_tr_pho_artist = np.array(y_tr_artist)
y_val_pho_artist = np.array(y_val_artist)
y_te_pho_artist = np.array(y_te_artist)


# get your rhyme model predictions file with the best predictions
p_file_name = "Predictions_rhyme_artist_B100E50ES10max_length:_5913vocab_size:_89kernel_sizes:_[11, 9, 7, 5]kernelseach:_10029.12.2020.15:03:22"

# get test set labels, your predicted labels, the dictionary to decode label ids, and training and test sets for occluded predictions
test_labels = readPickle("rhyme/y_te_artist") # artist labels for the rhyme model (test)
train_labels = readPickle("rhyme/y_tr_artist") # artist labels for the rhyme model (train)
predictions = readPickle("predictions/"+p_file_name) # just use the file name
artist_label_dictionary = readPickle("id2artist")
test_inputs = readPickle("rhyme/x_te_artist")
train_inputs = readPickle("rhyme/x_tr_artist")

# load the equivalent model stored
from tensorflow import keras
model = keras.models.load_model("saved_models/rhyme_artist_B100E50ES10max_length:_5913vocab_size:_89kernel_sizes:_[11, 9, 7, 5]kernelseach:_10029.12.2020.15:03:22.keras")


# finally, download your vocabulary. with this, we will be able to convert phoneme ids to actual phonemes
vocabulary = readPickle("rhyme/phoneme_vocabulary_Embeddings")
phoneme_encoder = {i+1 : pho for i, pho in enumerate(vocabulary)}
phoneme_decoder = {pho : i+1 for i, pho in enumerate(vocabulary)}
phoneme_encoder[0] = "PAD" # add also one for padding
phoneme_decoder["PAD"] = 0

print(phoneme_encoder)

# get all the artist labels for lyrics correctly predicted
corrects = [artist_label_dictionary[a+1] for a,b in zip(np.argmax(predictions, axis=-1), np.argmax(test_labels, axis=-1)) if a == b]
# check whether your accuracy holds with the model accuracy
print("Overall accuracy of the model:",len(corrects)/len(test_labels))
# now, sort the list by the number of occurances. see who was predicted with extreme accuracy
from collections import Counter
sorted_corrects = Counter(corrects)
sorted_corrects.most_common(10) # show the best 10
overall_results = dict()
for artist in artist_label_dictionary.values():
    try:
        overall_results[artist] = sorted_corrects[artist]
    except:
        overall_results[artist] = 0
print(len(overall_results), overall_results)

import math
import copy

def occluder(test_labels, predictions, label_dictionary, test_inputs, model, input_index, occlusion_size):
    
    if occlusion_size % 2 == 0 or type(occlusion_size) != int:
        raise ValueError('Occlusion size must be an odd integer')      
    
    # 1. select the desired singleton test input. It might be a correctly or incorrectly labeled sample
    input_selection = test_inputs[input_index]
    
    # 2. use the model parameters to make a prediction out of your single example. store the prediction and its confidence value
    input_selection_copy1 = copy.deepcopy(input_selection) # use a deep copy to keep the original undistorted
    inp = input_selection_copy1.reshape(1, len(input_selection))
    pred = model.predict(inp)[0]
    #print("Best guess is {} with {}".format(label_dictionary[np.argsort(pred)[-1]+1], pred[np.argsort(pred)[-1]]))
    #print("S. Best guess is {} with {}".format(label_dictionary[np.argsort(pred)[-2]+1], pred[np.argsort(pred)[-2]]))
    best = label_dictionary[np.argsort(pred)[-1]+1]
    base_prob = pred[np.argsort(pred)[-1]]
    second = label_dictionary[np.argsort(pred)[-2]+1]
    second_prob = pred[np.argsort(pred)[-2]]
    target = label_dictionary[np.argmax(test_labels[0], axis=-1)+1]
    
    
    # 3. depending on the occlusion_size, change input and make a new prediction with the occluded input
    # then, compare the new results with the original base probability, and record the difference in a list
    
    prob_change_list = list()

    for i in range(len(input_selection)):
        input_selection_copy2 = copy.deepcopy(input_selection) # use a copy to keep the original undistorted
        size = int((occlusion_size - 1)/2)
        begin = max(i-size,0)
        end = min(i+size,len(input_selection)-1)
        
        input_selection_copy2[begin:end] = 0  # !!! THIS CONSTANT CAN BE CHANGED !!!
        inp = input_selection_copy2.reshape(1,len(input_selection))
        occluded_pred = model.predict(inp)[0]
        best_guess = label_dictionary[np.argsort(pred)[-1]+1]
        second_guess = label_dictionary[np.argsort(pred)[-2]+1]
        best_guess_prob = occluded_pred[np.argsort(occluded_pred)[-1]]
        second_guess_prob = occluded_pred[np.argsort(occluded_pred)[-2]]
        
        
        # check whether the new probabilities changed the guessing order
        if best_guess == best:
            prob_change_list.append(best_guess_prob - base_prob)
        elif second_guess == best:
            prob_change_list.append(second_guess_prob - base_prob)
        else:
            raise ValueError('The probabilities changed dramatically!!!')
    
    return prob_change_list, input_selection, input_index, best, base_prob, second, second_prob

output = dict()
for i in range(1200):
    change_list, input_selection, index, best, base_prob, second, second_prob = occluder(test_labels, predictions, artist_label_dictionary, test_inputs, model, i, 3)
    output[i] = [change_list, max(change_list), min(change_list), sum(change_list)/len(change_list), len(change_list), best, base_prob, second, second_prob]
    if i % 10 == 0:
        print(i, "completed. Writing occlusions...")
        print(output)
        writePickle(output, "occlusion_outputs")
        