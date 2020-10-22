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

print("We'll work with a training set of {} samples and a test set of {} samples".format(len(x_train), len(x_te)))

# to get the corpus and the dictionary, we'll use all data samples
x_all = x_train + x_te



'''Convert lyrics to simple bags of words'''
import gensim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess

lyrics_tokens = [[token for token in simple_preprocess(lyric)] for lyric in x_all]
tokens_dictionary = corpora.Dictionary(lyrics_tokens)
print("The token dictionary created by all examples (including training and test) is:",tokens_dictionary.token2id)

all_tokenized = [simple_preprocess(lyric) for lyric in x_all]
corpus = [tokens_dictionary.doc2bow(lyric, allow_update=True) for lyric in all_tokenized]
corpus_word_counts = [[(tokens_dictionary[id], count) for id, count in line] for line in corpus]
print("Some examples of samples written in word counts:\n", corpus_word_counts[0], '\n', corpus_word_counts[1])



'''Download necessary gensim packages and a word vector model. Find the most similar training example for all test examples one by one'''

print("downloading gensim packages and fasttext model...")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api

# download fast text model
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

print("calculating similarity matrix... this may take a while!")
similarity_matrix = fasttext_model300.similarity_matrix(tokens_dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

# create a similarity dictionary where keys are text examples, and values are lists of cosine similarity values that show the similarity of each test sample with all training examples, one by one
similarity_dict = dict()
for i, lyric in enumerate(x_te):
    print("{}th example in progress".format(i))
    lyric_tokenized = tokens_dictionary.doc2bow(simple_preprocess(lyric), allow_update=True)
    similarity_dict[i] = list()
    for sample in x_train:
        sample_tokenized = tokens_dictionary.doc2bow(simple_preprocess(sample), allow_update=True)
        similarity_dict[i].append(softcossim(lyric_tokenized, sample_tokenized, similarity_matrix))
        
# save the dictionary
print("saving the similarity dictionary...")
writePickle(similarity_dict, "similarity_dict")

# print an example
print("An example of similarity: Similarity percentages of the first test example with 10800 training examples:", similarity_dict[0])

        
    
