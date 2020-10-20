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


''' SELECTION FIELDS (BEGIN)'''
# select your model
from tensorflow import keras
model = keras.models.load_model("saved_models/Equalsub_word_pre_trained_genre_30batch_40epoch_4Stop_filters=[3, 3, 3, 3, 3, 3]_pools=[7, 7, 7]_112filters_1024dense_outputs_3674length.keras")

# select your equivalent prediction file
predictions_file_name = "PredictionsEqual_sub_word_pre_trained_genre_30batch_40epoch_4Stop_filters=[3, 3, 3, 3, 3, 3]_pools=[7, 7, 7]_112filters_1024dense_outputs_3674length"

# select your types
model_type = "sub_word" # or "character"
label_type = "genre" # or 'artist'
''' SELECTION FIELDS (END)'''

test_labels = readPickle(str(model_type +"/"+"y_te_"+label_type+"_equal"))
predictions = readPickle(str("predictions/"+ predictions_file_name))
    
label_dictionary = readPickle(str("id2"+label_type))

test_inputs = readPickle(str(model_type +"/x_te_"+label_type+"_equal"))

# vocabulary will be used later on to show input in real lyric string form
vocabulary = readPickle(model_type+"/vocabulary")
vocabulary_reversed = dict()
for key, value in vocabulary.items():
    vocabulary_reversed[value] = key
vocabulary_reversed[0] = 'PAD'


import math
import copy


def occluded_prob_retriever(label):
    counter = 0
    prob_change_dict = dict()
    for input_sample in test_inputs: 
        input_copy = copy.deepcopy(input_sample)
        inp = input_copy.reshape(1, len(input_sample))
        prediction = model.predict(inp)[0]
        best_two_predictions = [label_dictionary[np.argsort(prediction)[-1]+1],label_dictionary[np.argsort(prediction)[-2]+1]]
        if label not in best_two_predictions:
            print(counter, "not in top two predictions")
            counter += 1
            continue
        elif label == label_dictionary[np.argsort(prediction)[-1]+1]:
            base_prob = prediction[np.argsort(prediction)[-1]]
        elif label == label_dictionary[np.argsort(prediction)[-2]+1]:
            base_prob = prediction[np.argsort(prediction)[-2]]
        print(counter, "out of", len(test_inputs), "predictions, being processed for occluded probabilities")

        for i in range(len(input_sample)):
            input_copy2 = copy.deepcopy(input_sample) # use a copy to keep the original undistorted
            size = int((3 - 1)/2) #!!!!!!!!
            begin = max(i-size,0)
            end = min(i+size,len(input_sample)-1)

            input_copy2[begin:end] = 0  # !!! THIS CONSTANT CAN BE CHANGED !!!
            inp = input_copy2.reshape(1,len(input_sample))
            occluded_pred = model.predict(inp)[0]
            best_guess = label_dictionary[np.argsort(prediction)[-1]+1]
            second_guess = label_dictionary[np.argsort(prediction)[-2]+1]
            best_guess_prob = occluded_pred[np.argsort(occluded_pred)[-1]]
            second_guess_prob = occluded_pred[np.argsort(occluded_pred)[-2]]

            converted_string = vocabulary_reversed[i]

            # check whether the new probabilities changed the guessing order
            if best_guess == label:
                try:
                    prob_change_dict[converted_string].append(best_guess_prob - base_prob)
                except:
                    prob_change_dict[converted_string] = list()
                    prob_change_dict[converted_string].append(best_guess_prob - base_prob)
            elif second_guess == label:
                try:
                    prob_change_dict[converted_string].append(second_guess_prob - base_prob)
                except:
                    prob_change_dict[converted_string] = list()
                    prob_change_dict[converted_string].append(second_guess_prob - base_prob)
            else:
                raise ValueError('The probabilities changed dramatically!!!')

        counter +=1
        
    return prob_change_dict, label

prob_change_dict, label = occluded_prob_retriever('Jazz')
print(prob_change_dict)

writePickle(prob_change_dict, str(model_type+"/prob_change_dict_"+label))