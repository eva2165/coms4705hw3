# Neural network dependency parsing
This is a homework submission for COMSW4705 Natural Language Processing.

The program requires the packages Keras and TensorFlow.

A feed-forward neural network model is trained to predict the transitions of an arc-standard dependency parser. The vocabulary (which words appear in the data, and a mapping from words to indices) was obtained by running `python get_vocab.py data/train.conll data/words.vocab data/pos.vocab`. `get_input_representation` converts the input and output pairs into one-hot concatenations of the input words and a one-hot vector of length 91. The training data was then used to train a model `model.h5`.

The parser can be evaluated with `python evaluate.py model.h5 data/dev.conll`.
