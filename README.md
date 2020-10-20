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

[1.1Helper_Lyrics_Retriever.py](../master/1.1Helper_Lyrics_Retriever.py):
- takes an initial metadata dictionary from [1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb)
- gets lyrics when possible from lyricwikia
- calculates certain additional metadata such as song_length, line_length etc.
- returns (saves) the more comprehensive metadata dictionary version for later use by [1.Dataset Preparation.ipynb](../master/1.Dataset&#32;Preparation.ipynb)

### B. Files:
-------

### C. Folders:
-------
