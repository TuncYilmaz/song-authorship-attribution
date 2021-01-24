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

[2.5.0_Results_and_Evaluation.ipynb](../master/2.Mini%20Models/2.5.0_Results_and_Evaluation.ipynb): a notebook that uses a variety of model predictions to display certain metrics of how the results perform
- takes trained model predictions and histories as input
- 16 model prediction files are uploaded under [predictions folder](../master/2.Mini%20Models/pickle_vars/predictions); 4 model history files are uploaded under [history folder](../master/2.Mini%20Models/pickle_vars/history); test labels for genre and artist labels separately are uploaded under [character](../master/2.Mini%20Models/pickle_vars/character) and [sub_word](../master/2.Mini%20Models/pickle_vars/sub_word) folders.

[2.3.4_Prediction_Evaluation.ipynb](../master/2.Mini%20Models/2.3.4_Prediction_Evaluation.ipynb):
- takes trained model predictions and history as its input
- 16 model prediction files are uploaded under [predictions folder](../master/2.Mini%20Models/pickle_vars/predictions); 4 model history files are uploaded under [history folder](../master/2.Mini%20Models/pickle_vars/history); test labels for genre and artist labels separately are uploaded under [character](../master/2.Mini%20Models/pickle_vars/character) and [sub_word](../master/2.Mini%20Models/pickle_vars/sub_word) folders.
With these files, the evaluation script can work; but for generating other evaluations, variables should be generated via different files!
- uses certain functions to calculate: model accuracies; confusion matrices for different label types; precision, recall and f-score values for models; plot accuracy and loss plots across model epochs; precision and recall plots for genre label models; refined test accuracy results

[2.4.1_Occlusion_Probabilities.py](../master/2.Mini%20Models/2.4.1_Occlusion_Probabilities.py): 
- for any given selected model, partially occludes the input areas one by one to record the effect of occlusion on output predictions. to have an idea about how occlusions work, please check out the short video at the very beginning of this [notebook](../master/2.Mini%20Models/2.4.2_1D_Genre_Occlusions.ipynb).
- the script takes one model and one label at a time.
- the actual model file used in the script is too big and doesn't comply with GitHub's 25mb file upload limit. therefore another (worse performing) smaller model was uploaded under [saved models folder](../master/2.Mini%20Models/saved_models). you can change line 20 in the script and use the uploaded model instead. the uploaded model is the best character embedding model for genre labeling. also you can change line 23 to write the corresponding predictions file stored under [predictions folder](../master/2.Mini%20Models/pickle_vars/predictions).
- it iterates over all model inputs and their respective predicted labels. for any given input, if the top two model predictions contain our desired label, then the scripts starts occluding the input piece by piece, and record how the ouput probability for that label changes with respect to each input piece. with each iteration, the findings are cumulatively recorded in an occlusion probability dictionary for that label specifically, where keys are input pieces and values are lists of probability changes.
- the output occlusion probabilities are transfered to [2.4.2_1D_Genre_Occlusions.ipynb](../master/2.Mini%20Models/2.4.2_1D_Genre_Occlusions.ipynb)
- by default the script handles occlusion with a span of 3 input areas. This number can be modified or can be arranged as modifiable in future versions!

[2.4.2_1D_Genre_Occlusions.ipynb](../master/2.Mini%20Models/2.4.2_1D_Genre_Occlusions.ipynb): 
- in the first part, the script shows how to generate occlusion graphs for any given input. as an example, the script takes a correctly labeled test input (from any of our working models) and applies a 3-piece-span occlusion function to record how each input piece actually effects the outcome. then the input is depicted in a 2D heatmap in which dark colors signify high positive effects (= high negative probability changes) and bright colors signify negative or no effects.
- in the second part, the script takes the occlusion probability dictionaries for each genre that were calculated by [2.4.1_Occlusion_Probabilities.py](../master/2.Mini%20Models/2.4.1_Occlusion_Probabilities.py), and plots the most effective 40 inputs pieces (20 with negative and 20 with positive effects) for each genre label separately. feel free to play with the plot configurations to create your own analysis.
- for the second part, since the script works with probability dictionaries, you need to create your own by using [2.4.1_Occlusion_Probabilities.py](../master/2.Mini%20Models/2.4.1_Occlusion_Probabilities.py). For an easy simulation, you can find and use the [predictions for Hip-Hop](../master/2.Mini%20Models/pickle_vars/sub_word/prob_change_dict_Hip%20Hop.pkl) for the sub_word model. 
- overall please mind the stored file names. you should either create your own model and prediction files by using scripts introduced earlier, or use the ones provided in this repository to be able to use the notebook.

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

The model, prediction and history file names have been set with certain convention rules to include hyper-parameter differences. However in the report, the model names were given in abbreviated forms for simplicity. Here you can find the conversions:
_Model File Ending --> Model Name Abbreviation_
04:27:34 --> CH1AR-S
02:15:43 --> CH1AR-L
20:24:13 --> SW1AR-50
19:37:27 --> SW1AR-100
15:03:22 --> PH1AR-F
01:26:30 --> PH1AR-T
03:05:46 --> CH1GE
12:04:38 --> SW1GE
19:27:51 --> PH1GE-F
15:08:30 --> PH1GE-T
