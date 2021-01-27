import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)



'''First, load the datasets and calculate the n-grams in line with your context size'''

# load all the lyrics in the mini dataset (12000 song lyrics)

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


# convert the dataset to phonemes
# also get the n-grams
import pickle

def writePickle(Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/rhyme/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "pickle_vars/rhyme/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


PHO2id = readPickle("PHO2id_Embeddings")
pronounciation_dict = readPickle("pronounciation_dict_Embeddings")
x = readPickle("lyrics")

print(len(x))

pho_x = list()
context_grams = list()
for song in x:
    pho_song = list()
    for line in song.split('\n'):
        pho_line = list()
        if line == '' or len(line) < 2:
            pass
        else:
            for token in line.split():
                try:
                    for pho in pronounciation_dict[token].split():
                        pho_song.append(PHO2id[pho]+1)
                        pho_line.append(pho)
                except:
                    pass
            try:
                line_grams = [([pho_line[i], pho_line[i + 1], pho_line[i + 2]], pho_line[i + 3]) for i in range(len(pho_line) - 3)]
                context_grams.extend(line_grams)
            except:
                pass                    
    pho_x.append(pho_song)
    
print(len(context_grams))
import random
# limit the size
context_grams = random.sample(context_grams, 500000)
print(len(context_grams))

# convert the list of lists to an np matrix
pho_x = np.array([np.array(song) for song in pho_x])


CONTEXT_SIZE = 3
EMBEDDING_DIM = 88
nb_epochs = 10


class NGramLanguageModeler(nn.Module):

    def __init__(self, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(embedding_dim, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 256)
        #self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(256, embedding_dim)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        #out = F.relu(self.linear2(out))
        out = self.linear3(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)


# get the embeddings
for epoch in range(nb_epochs):
    print("Epoch No:", epoch)
    total_loss = 0
    for context, target in context_grams:
        #print(context,type(target))

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([PHO2id[pho] for pho in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([PHO2id[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    print("Total Loss:",total_loss)
    '''Preprocess the embedding vectors and save it'''
    # create a dummy embedding weights matrix to be filled later on
    embedding_weights_trained = np.zeros((EMBEDDING_DIM + 1, EMBEDDING_DIM)) # one for padding

    for pho, i in PHO2id.items():
        embedding_vector = model.embeddings(torch.tensor([PHO2id[pho]], dtype=torch.long)).detach().numpy()[0]
        embedding_weights_trained[i+1] = embedding_vector

    try:
        writePickle(embedding_weights_trained, "embedding_weights_trained_0.5Msamples"+str(nb_epochs))
    except:
        writePickle(embedding_weights_trained, "embedding_weights_trained_0.5Msamples_10_epochs")
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


