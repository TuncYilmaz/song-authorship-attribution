# add pickle saving and loading functions
import pickle
def writePickle( Variable, fname):
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

# import necessary spacy packages for english tokenizer
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# import lyricwikia for retrieving lyrics
import lyricwikia

# load the metadata dictionary
metadata_dict = readPickle("metadata_dict")

counter = 1170000 # use a counter to report the progress
for song_id, entry in metadata_dict.items():
    print("Entry number:", counter)
    # check if the entry has a song written in English:
    if entry[6] == 'english':
        try: # we're not sure whether the lyrics are available
            lyric = lyricwikia.get_lyrics(entry[2], entry[-1])
            # add the lyric to the entry
            entry.append(lyric)
            # calculate the song length, # of blank lines, min_tokens and max_tokens
            blank_line_count = 0
            line_count = 0
            token_counts = []
            for line in lyric.split("\n"):
                if line == "":
                    blank_line_count += 1
                else:
                    line_count += 1
                    doc = nlp(line)
                    token_counts.append(len(doc))

             # add calculated metrics to the entry
            try:
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)
            except:
                min_tokens = 0
                max_tokens = 0
            entry.append(line_count)
            entry.append(blank_line_count)
            entry.append(min_tokens)
            entry.append(max_tokens)

        except:
            print("Lyrics not found")
            for i in range(5): # last five appended items in the entry should be 'NaN'
                entry.append('NaN')

    # if the song is not in English:
    else:
        print("Lyrics not in English")
        for i in range(5): # last five appended items in the entry should be 'NaN'
            entry.append('NaN')

    # increase counter
    counter +=1

    # gradually save the resulting dictionary to a pickle file as a precaution for abrupt errors
    if counter % 10000 == 0: # once in every 10000 completed entries
        print("Saving intermediate pickle file... The last song_id was:", song_id)
        writePickle(metadata_dict, "complete_metadata_dict")
    

# save the final outcome as a pickle file
writePickle(metadata_dict, "complete_metadata_dict")
