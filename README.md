# Song-Authorship-Attribution
Thesis work on Song Authorship Attribution by analyzing linguistic features in song lyrics

Please follow the enumeration of file names to have an hierarchical and chronological path in the project construction. 

### A. Scripts:
-------


[1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb): 
- takes the complete Wasabi song metadata csv file as an input and performs an initial preprocessing. this csv file is around 5 gb, and cannot be uploaded due to size restrictions. please contact me for the csv file
- by using lyricwikia, retrieves actual song lyrics. records everything in a dictionary. uses the helper [script](../master/1.1Helper_Lyrics_Retriever.py) to do that.
- performs additional preprocessing such as removing non-English songs, dealing with N\A genre values, etc.
- contains a comprehensive genre mapping dictionary which maps all genres to 14 comprehensive parent genre classes
- using the refined and preprocessed dataset version, plots certain graphs to analyze particular data statistics

[1.1Helper_Lyrics_Retriever.py](../master/1.1Helper_Lyrics_Retriever.py): works as a helper script for [1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb)
- takes an initial metadata dictionary from [1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb)
- gets lyrics when possible from lyricwikia
- calculates certain additional metadata such as song_length, line_length etc.
- returns (saves) the more comprehensive metadata dictionary version for later use by [1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb)

[2.0_Dataset_Formation.ipynb](../master/2.Mini%20Models/2.0_Dataset_Formation.ipynb):
- retrieves certain dictionaries created by [1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb), and processes the data to obtain a sub-dataset that contains 10 artists from each parent genre
- writes everything to a csv file called sub_dataset.csv
- also yields certain dictionaries used for indexing purposes

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

[2.3.1_Character_Model.py](../master/2.Mini%20Models/2.3.1_Character_Model.py): script that builds the model architecture with character embeddings as the input
- model parameters can be tuned after line 160
- script outputs saved files such as text predictions, model parameters and model history
- needs debugging for version that doesn't involve early stopping

[2.3.2_Sub_Word_Model.py](../master/2.Mini%20Models/2.3.2_Sub_Word_Model.py): script that builds the model architecture with sub-word embeddings as the input
- model parameters can be tuned after line 172
- script outputs saved files such as text predictions, model parameters and model history
- needs debugging for version that doesn't involve early stopping

[2.3.3_Overall_Model_Training.py](../master/2.Mini%20Models/2.3.3_Overall_Model_Training.py): script that combines [2.3.1_Character_Model.py](../master/2.Mini%20Models/2.3.1_Character_Model.py) and [2.3.2_Sub_Word_Model.py](../master/2.Mini%20Models/2.3.2_Sub_Word_Model.py) in a single function
- model parameters can be tuned after line 170
- script outputs saved files such as text predictions, model parameters and model history
- needs debugging for version that doesn't involve early stopping
- can be modified further to include automatic parameter value assignment (such as vocab_size = 160 if input_type == 'char')

[2.3.4_Prediction_Evaluation.ipynb](../master/2.Mini%20Models/2.3.4_Prediction_Evaluation.ipynb):
- takes trained model predictions and history as its input
- 16 model prediction files are uploaded under [predictions folder](../master/2.Mini%20Models/pickle_vars/predictions); 4 model history files are uploaded under [history folder](../master/2.Mini%20Models/pickle_vars/history); test labels for genre and artist labels separately are uploaded under [character](../master/2.Mini%20Models/pickle_vars/character) and [sub_word](../master/2.Mini%20Models/pickle_vars/sub_word) folders.
With these files, the evaluation script can work; but for generating other evaluations, variables should be generated via different files!
- uses certain functions to calculate: model accuracies; confusion matrices for different label types; precision, recall and f-score values for models; plot accuracy and loss plots across model epochs; precision and recall plots for genre label models; refined test accuracy results

[2.4.1_Occlusion_Probabilities.py](../master/2.Mini%20Models/2.4.1_Occlusion_Probabilities.py): 
<video src="../master/2.Mini%20Models/occlusion_graphs/occlusion-gif.mp4" width="320" height="200" controls preload></video>
- for any given selected model, partially occludes the input areas one by one to record the effect of occlusion on output predictions. to have an idea about how occlusions work, please check out the short video at the very beginning of this [notebook](../master/2.Mini%20Models/2.4.2_1D_Genre_Occlusions.ipynb).
- the script takes one model and one label at a time.
- it iterates over all model inputs and their respective predicted labels. for any given input, if the top two model predictions contain our desired label, then the scripts starts occluding the input piece by piece, and record how the ouput probability for that label changes with respect to each input piece. with each iteration, the findings are cumulatively recorded in an occlusion probability dictionary for that label specifically, where keys are input pieces and values are lists of probability changes.
- the output occlusion probabilities are transfered to [2.4.2_1D_Genre_Occlusions.ipynb](../master/2.Mini%20Models/2.4.2_1D_Genre_Occlusions.ipynb)
- by default the script handles occlusion with a span of 3 input areas. This number can be modified or can be arranged as modifiable in future versions!

[2.4.2_1D_Genre_Occlusions.ipynb](../master/2.Mini%20Models/2.4.2_1D_Genre_Occlusions.ipynb): 
- in the first part, the script shows how to generate occlusion graphs for any given input. as an example, the script takes a correctly labeled test input (from any of our working models) and applies a 3-piece-span occlusion function to record how each input piece actually effects the outcome. then the input is depicted in a 2D heatmap in which dark colors signify high positive effects (= high negative probability changes) and bright colors signify negative or no effects.
- in the second part, the script takes the occlusion probability dictionaries for each genre that were calculated by [2.4.1_Occlusion_Probabilities.py](../master/2.Mini%20Models/2.4.1_Occlusion_Probabilities.py), and plots the most effective 40 inputs pieces (20 with negative and 20 with positive effects) for each genre label separately. feel free to play with the plot configurations to create your own analysis.


### B. Files:
-------

### C. Folders:
-------
