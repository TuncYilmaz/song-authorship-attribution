# Song-Authorship-Attribution
Thesis work on Song Authorship Attribution by analyzing linguistic features in song lyrics.

Aims to collect a comprehensive lyrics dataset and a corresponding song metadata, preprocess the data on character, sub-word and phoneme level, run several CNN models and capture similar distribution over the input data.

Please follow the enumeration of file names to have an hierarchical and chronological path in the project construction. 

### A. Scripts:
-------


[1_Dataset_Preparation.ipynb](../master/1_Dataset_Preparation.ipynb): 
- takes the complete Wasabi song metadata csv file as an input and performs an initial preprocessing. this csv file is around 5 gb, and cannot be uploaded due to size restrictions. please contact me for the csv file
- by using lyricwikia, retrieves actual song lyrics. records everything in a dictionary. uses the helper [script](../master/1.1_Helper_Lyrics_Retriever.py) to do that.
- performs additional preprocessing such as removing non-English songs, dealing with N\A genre values, deleting duplicate entries etc.
- contains a comprehensive genre mapping dictionary which maps all genres to 14 comprehensive parent genre classes
- using the refined and preprocessed dataset version, plots certain graphs to analyze particular data statistics

[1.1_Helper_Lyrics_Retriever.py](../master/1.1_Helper_Lyrics_Retriever.py): works as a helper script for [1_Dataset_Preparation.ipynb](../master/1_Dataset_Preparation.ipynb)
- takes an initial metadata dictionary from [1_Dataset_Preparation.ipynb](../master/1_Dataset_Preparation.ipynb)
- gets lyrics when possible from lyricwikia
- calculates certain additional metadata such as song_length, line_length etc.
- returns (saves) the more comprehensive metadata dictionary version for later use by [1_Dataset_Preparation.ipynb](../master/1_Dataset_Preparation.ipynb)
- the input metadata dictionary variable (metadata_dict.pkl) and the output metadata dictionary variable (complete_metadata_dict.pkl) are too large and cannot be added into this depository. please contact me if you need access to these files.

[2.0_Dataset_Formation.ipynb](../master/2.Mini%20Models/2.0_Dataset_Formation.ipynb):
- retrieves certain dictionaries created by [1_Dataset_Preparation.ipynb](../master/1_Dataset_Preparation.ipynb), and processes the data to obtain a sub-dataset that contains 10 artists from each parent genre
- writes everything to a csv file called sub_dataset.csv
- also yields certain dictionaries (stored under [pickle_vars](../master/2.Mini%20Models/pickle_vars)) used for indexing purposes
- in the final section, includes a few plots displaying basic dataset statistics


##### Until here, all the provided scripts have been used for (99%) preprocessing purposes. Most of the variables obtained couldn't be uploaded into the repository due to file size restrictions. If you want to skip the preprocessing phase, the resulting processed dataset (that will be used in all models henceforth) is stored under the file: sub_dataset.csv

[2.1_Character_Model_Preprocessing.ipynb](../master/2.Mini%20Models/2.1_Character_Model_Preprocessing.ipynb):
- forms character embeddings that are derived from GloVe word embeddings. specifically uses the glove.6B.300d.txt model
- creates a collection of (lowercase) characters that exists in the complete dataset and also in the GloVe word embeddings
- converts the whole dataset samples into numbers representing known (and also unknown) characters in our collection
- prepares the embedding matrix for all characters, retrieved from the embedding indices in the initial GloVe model
- preprocesses the dataset to get ready for keras model input requirements (padding, data-splitting, etc.)
- saves important model input variables to pickle files

[2.2_Sub_Word_Model_Preprocessing.ipynb](../master/2.Mini%20Models/2.2_Sub_Word_Model_Preprocessing.ipynb):
- forms sub-word embeddings that are derived from the [sub-word level embedding package](https://github.com/bheinzerling/bpemb). specifically uses the English BPEmb model with default vocabulary size (10k) and 50-dimensional embeddings
- creates a collection of sub-word pieces recognized by the BPEmb package
- converts the whole dataset samples into sub-word representations
- preprocesses the dataset to get ready for keras model input requirements (padding, data-splitting, etc.)
- prepares the 50-dimensional embedding matrix for all sub-word pieces
- saves important model input variables to pickle files

[2.3_Phoneme_Model_Preprocessing.ipynb](../master/2.Mini%20Models/2.3_Phoneme_Model_Preprocessing.ipynb):
- starts by converting the CMU Pronouncing Dictionary (which can be found [here](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)) into a json file
- converts all words that exist in the CMU dictionary into their equivalent pronounciation versions
- for non-translatable words, uses other techniques for phoneme translation
- narrows the whole thing down to a set of 44 words without known pronounciations. these will be referred to as 'UNK' later on
- converts the datasets into phoneme versions, generates one-hot embeddings, and saves them as pickle variables

##### Now, with the scripts starting with 2.1_, 2.2_ and 2.3_, we have generated all the embeddings, converted datasets and mapping dictionaries. They are all stored under relevant sections of the [pickle_vars](../master/2.Mini%20Models/pickle_vars) folder. Hereafter we'll have model training and evaluation scripts

[2.4.1_Model_Training_Kim2014.py](../master/2.Mini%20Models/2.4.2_Model_Training_Zhang2016.py): model training script that is inspired by the architecture of [Kim](https://arxiv.org/pdf/1408.5882.pdf)
- this is a model architecture with a single convolutional layer with multiple kernels. **this is the model version that yields the best results for all input types. therefore all the models mentioned in the project report have been obtained by this script**
- inspired by this [script](https://github.com/Jverma/cnn-text-classification-keras/blob/master/text_cnn.py)
- depending on which input type to work with, model hyper-parameters can be set between lines 44-72
- script outputs saved files such as predictions and model history

[2.4.2_Model_Training_Zhang2016.py](../master/2.Mini%20Models/2.4.2_Model_Training_Zhang2016.py): model training script that mimics the architecture introduced by [Zhang et. al.](https://arxiv.org/pdf/1509.01626.pdf)
- this is a model architecture with 6 consecutive convolutional layers. this model yields worse results, therefore its results haven't been used & reported 
- we have trained this model only for all input types. however this file includes a combined version for only sub_word and char embeddings
- variables and hyper-parameters can be set after line 152
- script outputs saved files such as predictions, model parameters and model history
- needs debugging for version that doesn't involve early stopping

[2.4.3_Phoneme_Embedding_Training.py](../master/2.Mini%20Models/2.4.3_Phoneme_Embedding_Training.py): a small pytorch script for training phoneme embeddings
- this script uses the song lyrics as inputs. due to file size, the variable containing the lyrics cannot be accessed, therefore this training cannot be replicated. for viewing purposes only
- randomly samples 500000 3-grams of phoneme symbols taken from the complete dataset
- then trains a 10-epoch embedding mini-model over these samples, to get an embedding matrix that is of shape (89,88) (i.e. one row for each phoneme symbol in the vocabulary + a row for padding; column number is equal to the vocabulary length)
- the embedding matrix (i.e. the output of the script) can be found under the rhyme model [pickle variables folder](../master/2.Mini%20Models/pickle_vars/rhyme)

[2.4.4_Ensemble_Training.py](../master/2.Mini%20Models/2.4.4_Ensemble_Training.py): this is the ensemble model training attempted with the char and sub_word models obtained from [2.4.2_Model_Training_Zhang2016.py](../master/2.Mini%20Models/2.4.2_Model_Training_Zhang2016.py)
- the main idea is to get the outputs of the models before the dropout layer, and concatenate those. afterwards the concatenated layer will go through the dropout layer and the softmax layer to produce predictions
- it didn't have a significantly better effect than basic combination of prediction probability distributions. therefore discarded eventually
- for view purposes only & no effect on the final report

[2.5.1_Baseline_Cosine_Similarity_Model.py](../master/2.Mini%20Models/2.5.1_Baseline_Cosine_Similarity_Model.py):
- this involves the construction of a naive cosine similarity baseline model, where each test sample is compared with all training examples in terms of their cosine similarities.
- in the first section, creates the dataset splits that were used in sub_word and character models.
- uses gensim packages and a fasttext model to tokenize the input samples, and creates a collective similarity matrix.
- in the end it generates a similarity dictionary in which keys unique test sample lyrics, and the values are lists of cosine similarity angles between a given test sample lyrics and all training samples one by one. this dictionary is normally recorded and used by [2.5.2_CosSim_Baseline_Model_Evaluation.py](../master/2.Mini%20Models/2.5.2_CosSim_Baseline_Model_Evaluation.py) to yield accuracy scores. however due to the size of this dictionary it cannot be uploaded. users should run this script with desired configurations and generate their own similarity dictionaries to be saved under [cosine model variables folder](../master/2.Mini%20Models/cosine_model_pickle_vars).

[2.5.2_CosSim_Baseline_Model_Evaluation.py](../master/2.Mini%20Models/2.5.2_CosSim_Baseline_Model_Evaluation.py):
- in the first section, the script again creates the dataset splits that were used in sub_word and character models.
- in the second section, for each test sample, the script finds the closest training sample vector in terms of their cosine similarities. 
- the script can be run for either 'genre' or 'artist' labels. in each case, if the output label of a test sample and its closest training example matches, it counts as a true match, and vice versa.
- so far this baseline model has an accuracy score of 3.25% for artist labels and 11.25% for genre labels!

[2.6.1_Results_and_Evaluation.ipynb](../master/2.Mini%20Models/2.6.1_Results_and_Evaluation.ipynb): a notebook that uses a variety of model predictions to display certain metrics of how the results perform
- takes trained model predictions and histories as input
- 14 model prediction files are uploaded under the [predictions folder](../master/2.Mini%20Models/pickle_vars/predictions); 14 model history files are uploaded under the [history folder](../master/2.Mini%20Models/pickle_vars/history); test labels for genre and artist labels separately are uploaded under [character](../master/2.Mini%20Models/pickle_vars/character), [sub_word](../master/2.Mini%20Models/pickle_vars/sub_word) and [rhyme](../master/2.Mini%20Models/pickle_vars/rhyme) folders. With these files, the evaluation script can work.
- uses certain functions to calculate: model accuracies; confusion matrices for different label types; precision, recall and f-score values for models; plots accuracy and loss graphs across model epochs; precision and recall plots for genre label models; refined test accuracy results, combine models results...

[2.6.2_Occlusion_Operation_ArtistLabels.py](../master/2.Mini%20Models/2.6.2_Occlusion_Operation_ArtistLabels.py)
- occlusions work basically like: we have an input, and a corresponding output distribution over labels. whenever we occlude/block/hinder/disallow certain parts of the input, the model will yield a slightly different output distribution. so we take any given input, start occluding it section by section, and record how the prediction certainty percentage of the model for its best prediction changes
- we do this recording for every input instance in the test set, and record the resulting changes into a pickle variable
- the window size used in this script was 3. that means, given any sub-unit of any test instance, the corresponding recorded probability change will reflect the case where this sub-unit along with the sub-unit on the left and on the right (if any) are occluded
- although the output pickle variable is extremely large (and therefore not stored in this repository), the input variables (best phoneme model, its predictions, sub_dataset.csv, etc.) for this script have been provided. therefore the output can be theoretically regenerated by running this script

[2.6.3_Artist_Label_Occlusions.ipynb](../master/2.Mini%20Models/2.6.3_Artist_Label_Occlusions.ipynb):
- the results of the occlusion implementations are presented
- this file works with two large files that could not be contained in this repository. one of them is the occlusion probabilities obtained by [2.6.2_Occlusion_Operation_ArtistLabels.py](../master/2.Mini%20Models/2.6.2_Occlusion_Operation_ArtistLabels.py). the other one is the pickle file version of the whole dataset. therefore this file is also read-only. there are a few examples to show how it works. 
- the examples are taken from the best phoneme model predictions and the phoneme-level test set
- in the first part, we upload important variables to be used later (also the occlusion probability change list)
- then, we create a dataframe that reveals how certain song entries/artists are really close to each other or how they stand out from the others
- when we decide on the interesting examples, we just plot their occlusion color graphs. for this graphs, there are three distinct types. the difference between these types are explained in detail in the script

### B. Folders:
-------

- [2.Mini Models](../master/2.Mini%20Models): includes files and scripts used for generating the models that take a sub dataset of 12000 lyrics.
  - [cosine_model_pickle_vars](../master/2.Mini%20Models/cosine_model_pickle_vars): the folder that contains the naive baseline model variables. currently empty due to file size limit. existing model scripts direct to this folder.
  - [occlusion_graphs](../master/2.Mini%20Models/occlusion_graphs): contains a video about how occlusions work. additionally, occlusion plots are saved in this folder if triggerred by scripts.
  - [pickle_vars](../master/2.Mini%20Models/pickle_vars): the folder in which mini model variables are saved. contains the following sub-folders:
    - [character](../master/2.Mini%20Models/pickle_vars/character): character model variables are saved here.
    - [sub_word](../master/2.Mini%20Models/pickle_vars/sub_word): sub-word model variables are saved here.
    - [history](../master/2.Mini%20Models/pickle_vars/history): a selection of model history files are saved here.
    - [predictions](../master/2.Mini%20Models/pickle_vars/predictions): a selection of model prediction files are saved here.
  - [saved_models](../master/2.Mini%20Models/saved_models): the actual models (with parameters, weights, etc.) are saved here. due to size restrictions only one sample model file (the best individual phoneme artist label model = **PH1AR-F**) is kept. running model training scripts would save additional model files in this folder.
  
### C. File Name Mapping:
-------

The model, prediction and history file names have been set with certain convention rules to include hyper-parameter differences. However in the report, the model names were given in abbreviated forms for simplicity. Here you can find the conversions: <br/>
_Model File Ending --> Model Name Abbreviation_ <br/>
04:27:34 --> CH1AR-S <br/>
02:15:43 --> CH1AR-L <br/>
20:24:13 --> SW1AR-50 <br/>
19:37:27 --> SW1AR-100 <br/>
15:03:22 --> PH1AR-F <br/>
01:26:30 --> PH1AR-T <br/>
03:05:46 --> CH1GE <br/>
12:04:38 --> SW1GE <br/>
19:27:51 --> PH1GE-F <br/>
15:08:30 --> PH1GE-T <br/>
