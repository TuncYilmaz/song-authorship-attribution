'''Read the datasets and import indexing dictionaries'''

import pickle
def writePickle(Variable, fname):
    filename = fname +".pkl"
    f = open("cosine_model_pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "pickle_vars/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj



import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical

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

print("loading datasets...")
x, y_artist, y_genre = load_data()




'''Split data using the same split for all models. Genre labels and artists are equally distributed'''

print("splitting datasets...")

# Shuffle data
np.random.seed(23) # !!!!!! use the same seed value in all dataset split versions !!!!!!
shuffle_indices = np.random.permutation(np.arange(len(y_artist))) 
# shuffle all inputs with the same indices
x_shuffled = x[shuffle_indices]
y_artist_shuffled = y_artist[shuffle_indices]
y_genre_shuffled = y_genre[shuffle_indices]

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
x_tr_artist_equal, x_val_artist_equal, x_te_artist_equal = list(), list(), list()
x_tr_genre_equal, x_val_genre_equal, x_te_genre_equal = list(), list(), list()
y_tr_artist_equal, y_val_artist_equal, y_te_artist_equal = list(), list(), list()
y_tr_genre_equal, y_val_genre_equal, y_te_genre_equal = list(), list(), list()

for sample, art_label, gen_label in zip(x_shuffled, y_artist_shuffled, y_genre_shuffled):
    artist_label_index = np.argmax(art_label, axis=-1)
    genre_label_index = np.argmax(gen_label, axis=-1)
    # for artist labels
    if artist_label_count_dict[artist_label_index] < 80:
        x_tr_artist_equal.append(sample)
        y_tr_artist_equal.append(art_label)
    elif 80 <= artist_label_count_dict[artist_label_index] < 90:
        x_val_artist_equal.append(sample)
        y_val_artist_equal.append(art_label)
    elif 90 <= artist_label_count_dict[artist_label_index] < 100:
        x_te_artist_equal.append(sample)
        y_te_artist_equal.append(art_label)
    else:
        print("There is an error with artist counts!")
    artist_label_count_dict[artist_label_index] += 1
        
    # for genre labels
    if genre_label_count_dict[genre_label_index] < 800:
        x_tr_genre_equal.append(sample)
        y_tr_genre_equal.append(gen_label)
    elif 800 <= genre_label_count_dict[genre_label_index] < 900:
        x_val_genre_equal.append(sample)
        y_val_genre_equal.append(gen_label)
    elif 900 <= genre_label_count_dict[genre_label_index] < 1000:
        x_te_genre_equal.append(sample)
        y_te_genre_equal.append(gen_label)
    else:
        print("There is an error with genre counts!")
    genre_label_count_dict[genre_label_index] += 1


'''Here we need only the training and test sets. Also the x_tr versions for artist and genre are actually the same, so we'll just select either of those.'''

x_train = x_tr_artist_equal + x_val_artist_equal
y_train_artist = y_tr_artist_equal + y_val_artist_equal
y_train_genre = y_tr_genre_equal + y_val_genre_equal
x_te = x_te_artist_equal
y_te_genre = y_te_genre_equal
y_te_artist = y_te_artist_equal 


'''Get the similarity dictionary'''
def readPickleCosine(fname):
    filename = "cosine_model_pickle_vars/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

similarity_dict = readPickleCosine("similarity_dict")


'''EVALUATION'''

import numpy as np


def evaluate_similarity(label_type):
    if label_type not in ["genre", "artist"]: # depending on the output type of the model, we need either 'genre' or 'artist' for this argument
        raise ValueError('argument "label_type" must be either "genre" or "artist"')
        
    if label_type == "genre":
        y = y_te_genre
        label_dictionary = readPickle(str("id2"+"genre"))
        y_train = y_train_genre
    else:
        y = y_te_artist
        label_dictionary = readPickle(str("id2"+"artist"))
        y_train = y_train_artist
    risky = list()   
    comparison_dict = dict()
    values = list()
    crazy_values = list()
    truth = 0
    all_examples = 0
    for test_index, similarities in similarity_dict.items():
        all_examples += 1
        max_index = np.argmax(similarities)
        closest_label = label_dictionary[np.argmax(y_train[max_index], axis=-1)+1]
        actual_label = label_dictionary[np.argmax(y[test_index], axis=-1)+1]
        if closest_label == actual_label:
            truth += 1
            values.append(max(similarities))

            crazy_values.append([x_train[max_index],x_te[test_index]])
            if 0.8 < max(similarities) < 0.9:
                risky.append([x_train[max_index],x_te[test_index]])
        comparison_dict[test_index] = [actual_label, closest_label]
        print("Example {} processed: Actual label is {} while the closest predicted label is {}".format(all_examples, actual_label, closest_label))
        
    print("Out of {} test examples, {} are identified with the correct label. Therefore the overall accuracy of this similarity model is: {}".format(all_examples, truth, (truth/all_examples)))
    
    print(values)

    writePickle(risky, "risky")
    writePickle(comparison_dict, "comparison_dict")    
evaluate_similarity("artist")
        
    