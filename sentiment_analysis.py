'''
This program aims to classify the sentiment for the movie reviews from IMDB dataset.

Data preparation:
1) The IMDB movie review dataset:
It can be downloaded from:
http://ai.stanford.edu/%7Eamaas/data/sentiment/
Unpack the downloaded IMDB package to the folder: ./aclImdb

2) The GLOVE (Global Vectors for Word Representation) pretrained word vectors 
It can be downloaded from:
http://nlp.stanford.edu/data/glove.6B.zip
Unpack the downloaded glove package to the folder: ./glove.6B/

This program uses Keras deep learning library.

This program achieved an average accuracy of 0.90 over 10000 test samples. 
'''

import pyprind
import pandas as pd
import numpy as np
import os
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding
from keras.models import Model

# the path to the review texts and sentiment labels
data_path = './aclImdb'
# the path to the glove vectors
glove_path = './glove.6B/'
# max number of words in the texts to be vectorized (choose the frequent words)
max_nb_words = 20000
# max number of words in a review (the review is padded or trucated to the number)
num_words_per_review = 1000
# glove embedding dimension
glove_dim = 100
# the validation split
validation_ratio = 0.2

# fix the random seed
np.random.seed(123)

# load the movie review texts and sentiment labels
labels = {'pos': 1, 'neg': 0}
# there are totally 50,000 review texts
print '\n'
print 'Loading review texts and sentiment labels ...'
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(data_path, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
            
df.columns = ['review', 'sentiment']
texts = df['review'].values.tolist()
labels = df['sentiment'].values.tolist()

# load the glove vectors
print 'Loading GLOVE word vectors ...'
# the dictionary for maping a word to a 100-dim vector
glove_embedding = {}
f = open(os.path.join(glove_path, 'glove.6B.100d.txt'))
for line in f:
    fields = line.split()
    word = fields[0] # the first element is the word
    word_vector = np.asarray(fields[1:], dtype='float32') 
    glove_embedding[word] = word_vector
f.close()

# tokenize the words in the texts
tokenizer = Tokenizer(nb_words = max_nb_words) 
tokenizer.fit_on_texts(texts) 
# convert each review text into a sequence of word-indices
matrix_word_indices = tokenizer.texts_to_sequences(texts)
# the dictionary for mapping a word to an index
dictionary_word_index = tokenizer.word_index

# pad each review text to a fixed length of word sequence
matrix_word_indices_fixed_length = pad_sequences(matrix_word_indices, maxlen = num_words_per_review)
# convert to numpy arrays 
data = np.array(matrix_word_indices_fixed_length)
labels = np.array(labels)

# shuffle the data 
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# percentage of validation data
nb_validation_samples = int(validation_ratio*data.shape[0])

# allocation of training data and validation data
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_validation = data[-nb_validation_samples:]
y_validation = labels[-nb_validation_samples:]

# prepare embedding matrix
num_words = min(max_nb_words, len(dictionary_word_index))
# embedding_matrix[0] is a all-zero vector representing no word
embedding_matrix = np.zeros((num_words+1, glove_dim)) 
print 'Vectorizing the words ...'
for word, index in dictionary_word_index.items():
    if index > max_nb_words:
        continue 
    # get the glove vector for the word
    glove_vector = glove_embedding.get(word) 
    if glove_vector is not None: 
        embedding_matrix[index] = glove_vector

# define the model
# layer 0: the input layer
sequence_input = Input(shape=(num_words_per_review,), dtype='int32')
# layer-1: the embedding layer
embedding_layer = Embedding(num_words+1, glove_dim, weights=[embedding_matrix], input_length=num_words_per_review, trainable=True)
embedded_output = embedding_layer(sequence_input)
# layer-2: the first convolution layer
x = Conv1D(nb_filter=128, filter_length=5, activation='relu')(embedded_output)
# layer-3: the first pooling layer
x = MaxPooling1D(pool_length=5)(x)
# layer-4: the second convolution layer
x = Conv1D(128, 5, activation='relu')(x)
# layer-5: the second pooling layer
x = MaxPooling1D(pool_length = 5)(x)
# flatten layer
x = Flatten()(x)
# layer-6: the first dense layer
x = Dense(output_dim = 128, activation='relu')(x)
# layer-7: the second dense layer
x = Dense(output_dim = 128, activation='relu')(x)
# layer-8: the output layer
final_output = Dense(1, activation='sigmoid')(x)

# define the model
model = Model(input=sequence_input, output=final_output)
# compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

# training and validation
print 'Training the model ...'
model.fit(x=x_train, y=y_train, validation_data=(x_validation, y_validation), nb_epoch=5, batch_size=128, verbose=1)

# evaluate the model
print 'Evaluating the model ...'
test_accuracy = model.evaluate(x_validation, y_validation, verbose=1)
print '\nThe average accuracy on the evaluation data set is %.3f.' % test_accuracy[1]

 