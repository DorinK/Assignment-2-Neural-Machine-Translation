README for Analysis.py:

In order for Analysis.py to run you should have seaborn library installed:
pip install seaborn

In the current working directory you should have the following files:
1. Analysis.py
3. Utils.py

Four parameters to Analysis.py:
1. Path to src train file
2. Path to trg train file
3. Path to src dev file
4. Path to trg dev file

For example in the command line you should enter: 
python3 Analysis.py _srcTrainFile_ _trgTrainFile_ _srcDevFile_ _trgDevFile_
python3 Analysis.py data/train.src data/train.trg data/dev.src data/dev.trg

A file and a directory will be outputted to a new directory named 'Outputs_Part_3':
1. 'Heatmap Plots' directory - which will contain all the 10 heatmap plots (1 for each epoch)
2. Attention_Weights.json - holds all the attention weights for each of the 10 epochs.