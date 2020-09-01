README for AttentionBasedTrain.py and AttentionBasedEvaluate.py:

******************************************************************************************

For AttentionBasedTrain.py:

In the current working directory you should have the following files:
1. AttentionBasedTrain.py
2. Utils.py

Four parameters to AttentionBasedTrain.py:
1. Path to src train file
2. Path to trg train file
3. Path to src dev file
4. Path to trg dev file

For example in the command line you should enter: 
python3 AttentionBasedTrain.py _srcTrainFile_ _trgTrainFile_ _srcDevFile_ _trgDevFile_
python3 AttentionBasedTrain.py data/train.src data/train.trg data/dev.src data/dev.trg

Three files will be outputted to a new directory named 'Outputs_Part_2':
1. encoderFile - best model checkpoint according to the BLEU score.
2. decoderFile - best model checkpoint according to the BLEU score.
3. dictFile - stores the vocabularies of src and trg (of the train).

******************************************************************************************

For AttentionBasedEvaluate.py:

In the current working directory you should have the following files:
1. AttentionBasedEvaluate.py
2. Utils.py

Five parameters to AttentionBasedEvaluate.py are:
1. Path to encoderFile	(output of AttentionBasedTrain.py)
2. Path to decoderFile	(output of AttentionBasedTrain.py)
3. Path to dictFile	(output of AttentionBasedTrain.py)
4. Path to src test file 
5. Path to trg test file

For example in the command line you should enter: 
python3 AttentionBasedEvaluate.py Outputs_Part_2/encoderFile Outputs_Part_2/decoderFile Outputs_Part_2/dictFile data/test.src data/test.trg

One additional file will be outputted to 'Outputs_Part_2':
AttentionTest.pred - which holds the decoder's predictions to the test src file.