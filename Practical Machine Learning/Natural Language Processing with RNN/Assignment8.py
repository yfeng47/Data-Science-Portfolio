#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
import pandas as pd

# Visuals
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Python chakin package previously installed by 
#    pip install chakin
import chakin  

import json
import os
from collections import defaultdict


# In[2]:


chakin.search(lang='English')  # lists available indices in English

# Specify English embeddings file to download and install
# by index number, number of dimensions, and subfoder name
# Note that GloVe 50-, 100-, 200-, and 300-dimensional folders
# are downloaded with a single zip download
CHAKIN_INDEX = 18
NUMBER_OF_DIMENSIONS = 50
SUBFOLDER_NAME = "glove.twitter.27B"

DATA_FOLDER = "embeddings"
ZIP_FILE = os.path.join(DATA_FOLDER, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(DATA_FOLDER, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(
        UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(
        SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))
    
if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
    print("Downloading embeddings to '{}'".format(ZIP_FILE))
    chakin.download(number=CHAKIN_INDEX, save_dir='./{}'.format(DATA_FOLDER))
else:
    print("Embeddings already downloaded.")

if not os.path.exists(UNZIP_FOLDER):
    import zipfile
    if not os.path.exists(ZIP_FILE) and os.path.exists(ZIP_FILE_ALT):
        ZIP_FILE = ZIP_FILE_ALT
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
        zip_ref.extractall(UNZIP_FOLDER)
else:
    print("Embeddings already extracted.")

print('\nRun complete')

#-------------------------------------------------------------------------

CHAKIN_INDEX = 11
NUMBER_OF_DIMENSIONS = 50
SUBFOLDER_NAME = "gloVe.6B"

DATA_FOLDER = "embeddings"
ZIP_FILE = os.path.join(DATA_FOLDER, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(DATA_FOLDER, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(
        UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(
        SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))


if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
    # GloVe by Stanford is licensed Apache 2.0:
    #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
    #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
    #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
    print("Downloading embeddings to '{}'".format(ZIP_FILE))
    chakin.download(number=CHAKIN_INDEX, save_dir='./{}'.format(DATA_FOLDER))
else:
    print("Embeddings already downloaded.")

if not os.path.exists(UNZIP_FOLDER):
    import zipfile
    if not os.path.exists(ZIP_FILE) and os.path.exists(ZIP_FILE_ALT):
        ZIP_FILE = ZIP_FILE_ALT
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
        zip_ref.extractall(UNZIP_FOLDER)
else:
    print("Embeddings already extracted.")

print('\nRun complete')


# In[3]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os  # operating system functions
import os.path  # for manipulation of file path names

import re  # regular expressions

from collections import defaultdict

import nltk
from nltk.tokenize import TreebankWordTokenizer

import tensorflow as tf

RANDOM_SEED = 9999





import pandas as pd
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import Conv1D


# In[4]:


# Create file paths to gdrive
RANDOM_SEED = 9999

def reset_graph(seed=RANDOM_SEED):
  '''Makes output stable across runs'''
  tf.reset_default_graph()
  tf.set_random_seed(seed)
  np.random.seed(seed)

# Declare Variables
REMOVE_STOPWORDS = False  # no stopword removal
EVOCABSIZE = 10000  # specify desired size of pre-defined embedding vocabulary

# Select the pre-defined embeddings source
embeddings_dir_one = "/Users/dannyarenson/Desktop/MSDS422/run-jump-start-rnn-sentiment-v002/embeddings/gloVe.6B"
embeddings_dir_two = "/Users/dannyarenson/Desktop/MSDS422/run-jump-start-rnn-sentiment-v002/embeddings/gloVe.6B"

filename_50 = "glove.6B.50d.txt"
filename_100 = "glove.6B.100d.txt"

embeddings_filename_50 = os.path.join(
    embeddings_dir_one, 
    embeddings_dir_two , 
    filename_50
)

embeddings_filename_100 = os.path.join(
    embeddings_dir_one, 
    embeddings_dir_two , 
    filename_100
)


# In[5]:


def reset_graph(seed= RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

REMOVE_STOPWORDS = False # no stopword removal 
EVOCABSIZE = 10000  # specify desired size of pre-defined embedding vocabulary

embeddings_directory = 'embeddings/glove.twitter.27B'
filename = 'glove.twitter.27B.100d.txt'

embeddings_filename = os.path.join(embeddings_directory, filename)
embeddings_filename


# In[6]:


# Utility function for loading embeddings
# Creates the Python defaultdict dictionary word_to_embedding_dict
# for the requested pre-trained word embeddings
# Note the use of defaultdict data structure from the Python Standard Library
# collections_defaultdict.py lets the caller specify a default value up front
# The default value will be retuned if the key is not a known dictionary key
# That is, unknown words are represented by a vector of zeros
# For word embeddings, this default value is a vector of zeros

def load_embedding_from_disks(embeddings_filename, with_indexes=True):
    """
    Read a embeddings txt file. If `with_indexes=True`,
    we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`,
    otherwise we return only a direct
    `word_to_embedding_dict` dictionnary mapping
    from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []

    else:
        word_to_embedding_dict = dict()

    with open(embeddings_filename, "r", encoding="utf-8") as embeddings_file:
        for (i, line) in enumerate(embeddings_file):

            split = line.split(" ")

            word = split[0]

            representation = split[1:]
            representation = np.array([float(val) for val in representation])

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    # Empty representation for unknown words.
    _WORD_NOT_FOUND = [0.0] * len(representation)
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(
            index_to_embedding_array + [_WORD_NOT_FOUND]
        )
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict
    
print('\nLoading embeddings from', embeddings_filename)
word_to_index, index_to_embedding =     load_embedding_from_disks(embeddings_filename, with_indexes=True)
print("Embedding loaded from disks.")


# In[7]:


# Load embeddings for glove 50
print("Loading embeddings from", embeddings_filename_50)

word_to_index_50, index_to_embedding_50 = load_embedding_from_disks(
    embeddings_filename_50, with_indexes=True
)

print("Embedding loaded from disks.")


# In[8]:


# Load embeddings for glove 100
print("Loading embeddings from", embeddings_filename_100)

word_to_index_100, index_to_embedding_100 = load_embedding_from_disks(
    embeddings_filename_100, with_indexes=True
)

print("Embedding loaded from disks.")


# In[10]:


# Check embedding size/shapes
vocab_size_50, embedding_dim_50 = index_to_embedding_50.shape
print("50 Dimension embedding ---------------------------------------")
print("Embedding is of shape: {}".format(index_to_embedding_50.shape))
print("The first words are words that tend occur more often.\n")

vocab_size_100, embedding_dim_100 = index_to_embedding_100.shape
print("100 Dimension embedding ---------------------------------------")
print("Embedding is of shape: {}".format(index_to_embedding_100.shape))
print("This means (number of words, number of dimensions per word)\n")
print("The first words are words that tend occur more often.")

vocab_size, embedding_dim = index_to_embedding.shape
print("Embedding is of shape: {}".format(index_to_embedding.shape))
print("This means (number of words, number of dimensions per word)\n")
print("The first words are words that tend occur more often.")

print("Note: for unknown words, the representation is an empty vector,\n"
      "and the index is the last one. The dictionnary has a limit:")
print("    {} --> {} --> {}".format("A word", "Index in embedding", 
      "Representation"))
word = "worsdfkljsdf"  # a word obviously not in the vocabulary
idx = word_to_index[word] # index for word obviously not in the vocabulary
complete_vocabulary_size = idx 
embd = list(np.array(index_to_embedding[idx], dtype=int)) # "int" compact print
#print("    {} --> {} --> {}".format(word, idx, embd))
word = "the"
idx = word_to_index[word]
embd = list(index_to_embedding[idx])


# In[11]:


# Show how to use embeddings dictionaries with a test sentence
# This is a famous typing exercise with all letters of the alphabet
# https://en.wikipedia.org/wiki/The_quick_brown_fox_jumps_over_the_lazy_dog
a_typing_test_sentence = "The quick brown fox jumps over the lazy dog"
print("\nTest sentence: ", a_typing_test_sentence, "\n")
words_in_test_sentence = a_typing_test_sentence.split()

for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = index_to_embedding_50[word_to_index_50[word_]]
    print(word_ + ": ", embedding)


# In[12]:


# Check 100 embeddings
for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = index_to_embedding_100[word_to_index_100[word_]]
    print(word_ + ": ", embedding)


# ### Model Preprocessing
# #### File Load Preparation

# In[13]:


# Define vocabulary size for the language model
# To reduce the size of the vocabulary to the n most frequently used words

def default_factory():
    return EVOCABSIZE  # last/unknown-word row in limited_index_to_embedding


# dictionary has the items() function, returns list of (key, value) tuples
limited_word_to_index_50 = defaultdict(
    default_factory, {k: v for k, v in word_to_index_50.items() if v < EVOCABSIZE}
)

# Select the first EVOCABSIZE rows to the index_to_embedding
limited_index_to_embedding_50 = index_to_embedding_50[0:EVOCABSIZE, :]

# Set the unknown-word row to be all zeros as previously
limited_index_to_embedding_50 = np.append(
    limited_index_to_embedding_50,
    index_to_embedding_50[index_to_embedding_50.shape[0] - 1, :].reshape(1, embedding_dim_50),
    axis=0,
)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_50


# In[14]:


# dictionary has the items() function, returns list of (key, value) tuples
limited_word_to_index_100 = defaultdict(
    default_factory, {k: v for k, v in word_to_index_100.items() if v < EVOCABSIZE}
)

# Select the first EVOCABSIZE rows to the index_to_embedding
limited_index_to_embedding_100 = index_to_embedding_100[0:EVOCABSIZE, :]

# Set the unknown-word row to be all zeros as previously
limited_index_to_embedding_100 = np.append(
    limited_index_to_embedding_100,
    index_to_embedding_100[index_to_embedding_100.shape[0] - 1, :].reshape(1, embedding_dim_100),
    axis=0,
)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_100


# In[15]:


# code for working with movie reviews data
# Source: Miller, T. W. (2016). Web and Network Data Science.
#    Upper Saddle River, N.J.: Pearson Education.
#    ISBN-13: 978-0-13-388644-3
# This original study used a simple bag-of-words approach
# to sentiment analysis, along with pre-defined lists of
# negative and positive words.
# Code available at:  https://github.com/mtpa/wnds
# ------------------------------------------------------------
# Utility function to get file names within a directory
def listdir_no_hidden(path):
    start_list = os.listdir(path)
    end_list = []
    for file in start_list:
        if not file.startswith("."):
            end_list.append(file)
    return end_list


# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ["\r", "\n", "\t"]


# In[16]:


# We will not remove stopwords in this exercise because they are
# important to keeping sentences intact
if REMOVE_STOPWORDS:
    print(nltk.corpus.stopwords.words("english"))

    # previous analysis of a list of top terms showed a number of words, along
    # with contractions and other word strings to drop from further analysis, add
    # these to the usual English stopwords to be dropped from a document collection
    more_stop_words = [
        "cant",
        "didnt",
        "doesnt",
        "dont",
        "goes",
        "isnt",
        "hes",
        "shes",
        "thats",
        "theres",
        "theyre",
        "wont",
        "youll",
        "youre",
        "youve",
        "br" "ve",
        "re",
        "vs",
    ]

    some_proper_nouns_to_remove = [
        "dick",
        "ginger",
        "hollywood",
        "jack",
        "jill",
        "john",
        "karloff",
        "kudrow",
        "orson",
        "peter",
        "tcm",
        "tom",
        "toni",
        "welles",
        "william",
        "wolheim",
        "nikita",
    ]

    # start with the initial list and add to it for movie text work
    stoplist = (
        nltk.corpus.stopwords.words("english")
        + more_stop_words
        + some_proper_nouns_to_remove
    )


# In[17]:


# text parsing function for creating text documents
# there is more we could do for data preparation
# stemming... looking for contractions... possessives...
# but we will work with what we have in this parsing function
# if we want to do stemming at a later time, we can use
#     porter = nltk.PorterStemmer()
# in a construction like this
#     words_stemmed =  [porter.stem(word) for word in initial_words]
def text_parse(string):
    # replace non-alphanumeric with space
    temp_string = re.sub("[^a-zA-Z]", "  ", string)
    # replace codes with space
    for i in range(len(codelist)):
        stopstring = " " + codelist[i] + "  "
        temp_string = re.sub(stopstring, "  ", temp_string)
    # replace single-character words with space
    temp_string = re.sub("\s.\s", " ", temp_string)
    # convert uppercase to lowercase
    temp_string = temp_string.lower()
    if REMOVE_STOPWORDS:
        # replace selected character strings/stop-words with space
        for i in range(len(stoplist)):
            stopstring = " " + str(stoplist[i]) + " "
            temp_string = re.sub(stopstring, " ", temp_string)
    # replace multiple blank characters with one blank character
    temp_string = re.sub("\s+", " ", temp_string)
    return temp_string


# ### Load Review Data

# In[18]:


# -----------------------------------------------
# gather data for 500 negative movie reviews
# -----------------------------------------------
dir_name = 'movie-reviews-negative'
    
filenames = listdir_no_hidden(path=dir_name)
num_files = len(filenames)

for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
    assert file_exists
print('\nDirectory:',dir_name)    
print('%d files found' % len(filenames))

# Read data for negative movie reviews
# Data will be stored in a list of lists where the each list represents 
# a document and document is a list of words.
# We then break the text into words.

def read_data(filename):

  with open(filename, encoding='utf-8') as f:
    data = tf.compat.as_str(f.read())
    data = data.lower()
    data = text_parse(data)
    data = TreebankWordTokenizer().tokenize(data)  # The Penn Treebank

  return data

negative_documents = []

print('\nProcessing document files under', dir_name)
for i in range(num_files):
    ## print(' ', filenames[i])

    words = read_data(os.path.join(dir_name, filenames[i]))

    negative_documents.append(words)
    # print('Data size (Characters) (Document %d) %d' %(i,len(words)))
    # print('Sample string (Document %d) %s'%(i,words[:50]))


# In[19]:


# -----------------------------------------------
# gather data for 500 positive movie reviews
# -----------------------------------------------
dir_name = 'movie-reviews-positive'  
filenames = listdir_no_hidden(path=dir_name)
num_files = len(filenames)

for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
    assert file_exists
print('\nDirectory:',dir_name)    
print('%d files found' % len(filenames))

# Read data for positive movie reviews
# Data will be stored in a list of lists where the each list 
# represents a document and document is a list of words.
# We then break the text into words.

def read_data(filename):

  with open(filename, encoding='utf-8') as f:
    data = tf.compat.as_str(f.read())
    data = data.lower()
    data = text_parse(data)
    data = TreebankWordTokenizer().tokenize(data)  # The Penn Treebank

  return data

positive_documents = []

print('\nProcessing document files under', dir_name)
for i in range(num_files):
    ## print(' ', filenames[i])

    words = read_data(os.path.join(dir_name, filenames[i]))

    positive_documents.append(words)
    # print('Data size (Characters) (Document %d) %d' %(i,len(words)))
    # print('Sample string (Document %d) %s'%(i,words[:50]))


# In[20]:


# -----------------------------------------------------
# convert positive/negative documents into numpy array
# note that reviews vary from 22 to 1052 words
# so we use the first 20 and last 20 words of each review
# as our word sequences for analysis
# -----------------------------------------------------
max_review_length = 0  # initialize
for doc in negative_documents:
    max_review_length = max(max_review_length, len(doc))
for doc in positive_documents:
    max_review_length = max(max_review_length, len(doc))
print("max_review_length:", max_review_length)

min_review_length = max_review_length  # initialize
for doc in negative_documents:
    min_review_length = min(min_review_length, len(doc))
for doc in positive_documents:
    min_review_length = min(min_review_length, len(doc))
print("min_review_length:", min_review_length)


# In[21]:


# construct list of 1000 lists with 40 words in each list
from itertools import chain

documents = []

for doc in negative_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20 : len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))
    
for doc in positive_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20 : len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))


# In[22]:


# create list of lists of lists for embeddings
embeddings_50 = []

for doc in documents:
    embedding = []
    for word in doc:
        embedding.append(limited_index_to_embedding_50[limited_word_to_index_50[word]])
    embeddings_50.append(embedding)


# In[23]:


# create list of lists of lists for embeddings
embeddings_100 = []

for doc in documents:
    embedding = []
    for word in doc:
        embedding.append(limited_index_to_embedding_100[limited_word_to_index_100[word]])
    embeddings_100.append(embedding)


# In[24]:


# -----------------------------------------------------
# Check on the embeddings list of list of lists
# -----------------------------------------------------
# Show the first word in the first document
test_word = documents[0][0]
print("First word in first document:", test_word)
print(
    "Embedding for this word:\n",
    limited_index_to_embedding_50[limited_word_to_index_50[test_word]],
)
print(
    "Corresponding embedding from embeddings list of list of lists\n",
    embeddings_50[0][0][:],
)


# In[25]:


# -----------------------------------------------------
# Make embeddings a numpy array for use in an RNN
# Create training and test sets with Scikit Learn
# -----------------------------------------------------
embeddings_array_50 = np.array(embeddings_50)

# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate(
    (np.zeros((500), dtype=np.int32), np.ones((500), dtype=np.int32)), axis=0
)


# In[26]:


embeddings_array_100 = np.array(embeddings_100)


# In[27]:


# Review the shape
print(embeddings_array_50.shape)
print(embeddings_array_100.shape)
print(thumbs_down_up.shape)


# ### Modeling
# #### Train Test Split
# 

# In[28]:


# Scikit Learn for random splitting of the data
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    embeddings_array_50, thumbs_down_up, test_size=0.20, random_state=RANDOM_SEED
)


# In[29]:


# Random splitting of the data in to training (80%) and test (20%)
X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(
    embeddings_array_100, thumbs_down_up, test_size=0.20, random_state=RANDOM_SEED
)


# In[30]:


print(X_train.shape)
print(X_train_100.shape)


# #### Model 1:
# * 20 Neurons
# * 50 dimensions of pre-trained embeddings
# * 50 epochs
# * 100 batch size

# In[31]:


# Build Model
reset_graph()

n_steps = embeddings_array_50.shape[1]  # number of words per document
n_inputs = embeddings_array_50.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 20  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 50
batch_size = 100

with tf.Session() as sess:
    init.run()
    print("Start of Training: Embeddings = 50")
    epochs = []
    train_acc = []
    test_acc = []
    for epoch in range(n_epochs):
        print("\n  ---- Epoch ", epoch + 1, " ----")
        epochs.append(epoch + 1)
        for iteration in range(y_train.shape[0] // batch_size):
            X_batch = X_train[iteration * batch_size : (iteration + 1) * batch_size, :]
            y_batch = y_train[iteration * batch_size : (iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Train accuracy:", acc_train, "Test accuracy:", acc_test)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
    print("End of Training")


# #### Model 2:
# * 30 Neurons
# * 50 dimensions of pre-trained embeddings
# * 25 epochs
# * 100 batch size

# In[32]:


# Build the model
reset_graph()

n_steps = embeddings_array_50.shape[1]  # number of words per document
n_inputs = embeddings_array_50.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 30  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 25
batch_size = 100

with tf.Session() as sess:
    init.run()
    print("Start of Training: Embeddings = 50")
    epochs = []
    train_acc = []
    test_acc = []
    for epoch in range(n_epochs):
        print("\n  ---- Epoch ", epoch + 1, " ----")
        epochs.append(epoch + 1)
        for iteration in range(y_train.shape[0] // batch_size):
            X_batch = X_train[iteration * batch_size : (iteration + 1) * batch_size, :]
            y_batch = y_train[iteration * batch_size : (iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Train accuracy:", acc_train, "Test accuracy:", acc_test)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
    print("End of Training")


# #### Model 3:
# * 20 neurons
# * 100 dimensions of pre-trained embeddings
# * 50 epochs
# * 100 batch size

# In[33]:


# Build the model
reset_graph()

n_steps = embeddings_array_100.shape[1]  # number of words per document
n_inputs = embeddings_array_100.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 20  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 50
batch_size = 100

with tf.Session() as sess:
    init.run()
    print("Start of Training: Embeddings = 100")
    epochs = []
    train_acc = []
    test_acc = []
    for epoch in range(n_epochs):
        print("\n  ---- Epoch ", epoch + 1, " ----")
        epochs.append(epoch + 1)
        for iteration in range(y_train.shape[0] // batch_size):
            X_batch = X_train_100[iteration * batch_size : (iteration + 1) * batch_size, :]
            y_batch = y_train_100[iteration * batch_size : (iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test_100, y: y_test_100})
        print("Train accuracy:", acc_train, "Test accuracy:", acc_test)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
    print("End of Training")


# #### Model 4:
# * 30 Neurons
# * 100 dimensions of pre-trained embeddings
# * 25 epochs
# * 100 batch size

# In[34]:


#Build the model
reset_graph()

n_steps = embeddings_array_100.shape[1]  # number of words per document
n_inputs = embeddings_array_100.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 30  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 25
batch_size = 100

with tf.Session() as sess:
    init.run()
    print("Start of Training: Embeddings = 100")
    epochs = []
    train_acc = []
    test_acc = []
    for epoch in range(n_epochs):
        print("\n  ---- Epoch ", epoch + 1, " ----")
        epochs.append(epoch + 1)
        for iteration in range(y_train.shape[0] // batch_size):
            X_batch = X_train_100[iteration * batch_size : (iteration + 1) * batch_size, :]
            y_batch = y_train_100[iteration * batch_size : (iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test_100, y: y_test_100})
        print("Train accuracy:", acc_train, "Test accuracy:", acc_test)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
    print("End of Training")


# #### Model 5:
# * 30 Neurons
# * 100 dimensions of pre-trained embeddings
# * 40 epochs
# * 100 batch size

# In[35]:


# Build the model
reset_graph()

n_steps = embeddings_array_100.shape[1]  # number of words per document
n_inputs = embeddings_array_100.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 20  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 40
batch_size = 100

with tf.Session() as sess:
    init.run()
    print("Start of Training: Embeddings = 50")
    epochs = []
    train_acc = []
    test_acc = []
    for epoch in range(n_epochs):
        print("\n  ---- Epoch ", epoch + 1, " ----")
        epochs.append(epoch + 1)
        for iteration in range(y_train.shape[0] // batch_size):
            X_batch = X_train_100[iteration * batch_size : (iteration + 1) * batch_size, :]
            y_batch = y_train_100[iteration * batch_size : (iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test_100, y: y_test_100})
        print("Train accuracy:", acc_train, "Test accuracy:", acc_test)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
    print("End of Training")


# In[36]:


def default_factory():
    return EVOCABSIZE  # last/unknown-word row in limited_index_to_embedding
# dictionary has the items() function, returns list of (key, value) tuples
limited_word_to_index = defaultdict(default_factory,     {k: v for k, v in word_to_index.items() if v < EVOCABSIZE})

# Select the first EVOCABSIZE rows to the index_to_embedding
limited_index_to_embedding = index_to_embedding[0:EVOCABSIZE,:]
# Set the unknown-word row to be all zeros as previously
limited_index_to_embedding = np.append(limited_index_to_embedding, 
    index_to_embedding[index_to_embedding.shape[0] - 1, :].\
        reshape(1,embedding_dim), 
    axis = 0)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding

# Verify the new vocabulary: should get same embeddings for test sentence
# Note that a small EVOCABSIZE may yield some zero vectors for embeddings
print('\nTest sentence embeddings from vocabulary of', EVOCABSIZE, 'words:\n')
for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = limited_index_to_embedding[limited_word_to_index[word_]]
    print(word_ + ": ", embedding)


# In[37]:


# -----------------------------------------------------
# convert positive/negative documents into numpy array
# note that reviews vary from 22 to 1052 words   
# so we use the first 20 and last 20 words of each review 
# as our word sequences for analysis
# -----------------------------------------------------
max_review_length = 0  # initialize
for doc in negative_documents:
    max_review_length = max(max_review_length, len(doc))    
for doc in positive_documents:
    max_review_length = max(max_review_length, len(doc)) 
print('max_review_length:', max_review_length) 

min_review_length = max_review_length  # initialize
for doc in negative_documents:
    min_review_length = min(min_review_length, len(doc))    
for doc in positive_documents:
    min_review_length = min(min_review_length, len(doc)) 
print('min_review_length:', min_review_length) 

# construct list of 1000 lists with 20 words in each list
from itertools import chain
documents = []
for doc in negative_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    
for doc in positive_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    

# create list of lists of lists for embeddings
embeddings = []    
for doc in documents:
    embedding = []
    for word in doc:
       embedding.append(limited_index_to_embedding[limited_word_to_index[word]]) 
    embeddings.append(embedding)
    
#embeddings


# In[38]:


# -----------------------------------------------------    
# Check on the embeddings list of list of lists 
# -----------------------------------------------------
# Show the first word in the first document
test_word = documents[0][0]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[0][0][:])

# Show the seventh word in the tenth document
test_word = documents[6][9]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[6][9][:])

# Show the last word in the last document
test_word = documents[999][39]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[999][39][:])


# In[39]:


embeddings_array = np.array(embeddings)
# -----------------------------------------------------    
# Make embeddings a numpy array for use in an RNN 
# Create training and test sets with Scikit Learn
# -----------------------------------------------------
embeddings_array = np.array(embeddings)

# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate((np.zeros((500), dtype = np.int32), 
                      np.ones((500), dtype = np.int32)), axis = 0)

# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test =     train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# #### Model 6:
# * 20 Neurons
# * 100 dimensions of pre-trained embeddings
# * 50 epochs
# * 100 batch size

# In[41]:


reset_graph()

n_steps = embeddings_array.shape[1]  # number of words per document 
n_inputs = embeddings_array.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 20  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 50
batch_size = 100

with tf.Session() as sess:
    init.run()
    print("Start of Training: Embeddings = 100")
    epochs = []
    train_acc = []
    test_acc = []
    for epoch in range(n_epochs):
        print('\n  ---- Epoch ', epoch, ' ----\n')
        for iteration in range(y_train.shape[0] // batch_size):          
            X_batch = X_train[iteration*batch_size:(iteration + 1)*batch_size,:]
            y_batch = y_train[iteration*batch_size:(iteration + 1)*batch_size]
            print('  Batch ', iteration, ' training observations from ',  
                  iteration*batch_size, ' to ', (iteration + 1)*batch_size-1,)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print('\n  Train accuracy:', acc_train, 'Test accuracy:', acc_test)


# In[89]:


# Save off the data for comparison dataframe
model_six = {
    "epochs": epochs,
    "train_acc": train_acc,
    "test_acc": test_acc,
    "neurons": n_neurons
}


# In[44]:


def reset_graph(seed= RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

REMOVE_STOPWORDS = False  # no stopword removal 
EVOCABSIZE = 400000

embeddings_directory = 'embeddings/glove.6B'
filename = 'glove.6B.300d.txt'

embeddings_filename = os.path.join(embeddings_directory, filename)
embeddings_filename


# In[45]:


def load_embedding_from_disks(embeddings_filename, with_indexes=True):
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
  
    else:
        word_to_embedding_dict = dict()

    with open(embeddings_filename, 'r') as embeddings_file:
        for (i, line) in enumerate(embeddings_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    # Empty representation for unknown words.
    _WORD_NOT_FOUND = [0.0] * len(representation)
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(
            lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(
            index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict

print('\nLoading embeddings from', embeddings_filename)
word_to_index, index_to_embedding =     load_embedding_from_disks(embeddings_filename, with_indexes=True)
print("Embedding loaded from disks.")


# In[46]:


vocab_size, embedding_dim = index_to_embedding.shape
print("Embedding is of shape: {}".format(index_to_embedding.shape))
print("This means (number of words, number of dimensions per word)\n")
print("The first words are words that tend occur more often.")

print("Note: for unknown words, the representation is an empty vector,\n"
      "and the index is the last one. The dictionnary has a limit:")
print("    {} --> {} --> {}".format("A word", "Index in embedding", 
      "Representation"))
word = "worsdfkljsdf"  # a word obviously not in the vocabulary
idx = word_to_index[word] # index for word obviously not in the vocabulary
complete_vocabulary_size = idx 
embd = list(np.array(index_to_embedding[idx], dtype=int)) # "int" compact print
#print("    {} --> {} --> {}".format(word, idx, embd))
word = "the"
idx = word_to_index[word]
embd = list(index_to_embedding[idx])  # "int" for compact print only.
#print("    {} --> {} --> {}".format(word, idx, embd))


# In[47]:


def default_factory():
    return EVOCABSIZE  # last/unknown-word row in limited_index_to_embedding
# dictionary has the items() function, returns list of (key, value) tuples
limited_word_to_index = defaultdict(default_factory,     {k: v for k, v in word_to_index.items() if v < EVOCABSIZE})

# Select the first EVOCABSIZE rows to the index_to_embedding
limited_index_to_embedding = index_to_embedding[0:EVOCABSIZE,:]
# Set the unknown-word row to be all zeros as previously
limited_index_to_embedding = np.append(limited_index_to_embedding, 
    index_to_embedding[index_to_embedding.shape[0] - 1, :].\
        reshape(1,embedding_dim), 
    axis = 0)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding

# Verify the new vocabulary: should get same embeddings for test sentence
# Note that a small EVOCABSIZE may yield some zero vectors for embeddings
print('\nTest sentence embeddings from vocabulary of', EVOCABSIZE, 'words:\n')
for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = limited_index_to_embedding[limited_word_to_index[word_]]
    print(word_ + ": ", embedding)


# In[48]:


# -----------------------------------------------------
# convert positive/negative documents into numpy array
# note that reviews vary from 22 to 1052 words   
# so we use the first 20 and last 20 words of each review 
# as our word sequences for analysis
# -----------------------------------------------------
max_review_length = 0  # initialize
for doc in negative_documents:
    max_review_length = max(max_review_length, len(doc))    
for doc in positive_documents:
    max_review_length = max(max_review_length, len(doc)) 
print('max_review_length:', max_review_length) 

min_review_length = max_review_length  # initialize
for doc in negative_documents:
    min_review_length = min(min_review_length, len(doc))    
for doc in positive_documents:
    min_review_length = min(min_review_length, len(doc)) 
print('min_review_length:', min_review_length) 

# construct list of 1000 lists with 40 words in each list
from itertools import chain
documents = []
for doc in negative_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    
for doc in positive_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    

# create list of lists of lists for embeddings
embeddings = []    
for doc in documents:
    embedding = []
    for word in doc:
       embedding.append(limited_index_to_embedding[limited_word_to_index[word]]) 
    embeddings.append(embedding)


# In[49]:


# -----------------------------------------------------    
# Check on the embeddings list of list of lists 
# -----------------------------------------------------
# Show the first word in the first document
test_word = documents[0][0]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[0][0][:])

# Show the seventh word in the tenth document
test_word = documents[6][9]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[6][9][:])

# Show the last word in the last document
test_word = documents[999][39]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[999][39][:])


# In[50]:


RANDOM_SEED = 9999
# -----------------------------------------------------    
# Make embeddings a numpy array for use in an RNN 
# Create training and test sets with Scikit Learn
# -----------------------------------------------------
embeddings_array = np.array(embeddings)

# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate((np.zeros((500), dtype = np.int32), 
                      np.ones((500), dtype = np.int32)), axis = 0)

# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test =     train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# #### Model 7:
# * 20 Neurons
# * 300 dimensions of pre-trained embeddings
# * 50 epochs
# * 100 batch size

# In[98]:


get_ipython().run_cell_magic('time', '', 'reset_graph()\n\nlstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)\n\nn_steps = embeddings_array.shape[1]  # number of words per document \nn_inputs = embeddings_array.shape[2]  # dimension of  pre-trained embeddings\nn_neurons = 20  # analyst specified number of neurons\nn_outputs = 2  # thumbs-down or thumbs-up\nn_layers = 3\n\nlearning_rate = 0.0001\n\nX = tf.placeholder(tf.float32, [None, n_steps, n_inputs])\ny = tf.placeholder(tf.int32, [None])\n\nlstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)\n              for layer in range(n_layers)]\nmulti_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)\noutputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)\ntop_layer_h_state = states[-1][1]\nlogits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")\nxentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)\nloss = tf.reduce_mean(xentropy, name="loss")\noptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\ntraining_op = optimizer.minimize(loss)\ncorrect = tf.nn.in_top_k(logits, y, 1)\naccuracy = tf.reduce_mean(tf.cast(correct, tf.float32))\n\ninit = tf.global_variables_initializer()\n\nn_epochs = 50\nbatch_size = 100\n\nresults = {}\n\nwith tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs):\n        for iteration in range(y_train.shape[0] // batch_size):          \n            X_batch = X_train[iteration*batch_size:(iteration + 1)*batch_size,:]\n            y_batch = y_train[iteration*batch_size:(iteration + 1)*batch_size]\n            print(\'  Batch \', iteration, \' training observations from \',  \n                  iteration*batch_size, \' to \', (iteration + 1)*batch_size-1,)\n            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})\n        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})\n        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})\n        results[epoch] = {\'Train\' : acc_train, \'Test\' : acc_test}\n        print("Epoch", epoch, "Train accuracy =", acc_train, "Test accuracy =", acc_test)')


# #### Model 8:
# * 30 Neurons
# * 300 dimensions of pre-trained embeddings
# * 50 epochs
# * 100 batch size

# In[52]:


reset_graph()

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

n_steps = embeddings_array.shape[1]  # number of words per document 
n_inputs = embeddings_array.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 30  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up
n_layers = 3

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 50
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(y_train.shape[0] // batch_size):          
            X_batch = X_train[iteration*batch_size:(iteration + 1)*batch_size,:]
            y_batch = y_train[iteration*batch_size:(iteration + 1)*batch_size]
            print('  Batch ', iteration, ' training observations from ',  
                  iteration*batch_size, ' to ', (iteration + 1)*batch_size-1,)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Epoch", epoch, "Train accuracy =", acc_train, "Test accuracy =", acc_test)


# In[3]:


nn_summary_df = pd.DataFrame(
    {
        "Word Vector": ["GloVe.6B.50d", "GloVe.6B.50d","GloVe.6B.100d","GloVe.6B.100d","GloVe.6B.100d","GloVe.Twitter.100d","GloVe.6B.300d", "GloVe.6B.300d"],
        "RNN Model Type": ["Basic", "Basic","Basic","Basic","Basic","Basic","LSTM", "LSTM"],
        "Neurons": [20, 30, 20, 30, 30, 20, 20, 30],
        "Dimensions": [50, 50, 100, 100, 100, 100, 300, 300],
        "Epochs": [50, 25, 50, 25, 40, 25, 50, 50],
        "TrainingAccuracy": [0.86, 0.82, 0.94, 0.86, 0.89, 0.93, 0.88, 1.0],
        "TestingAccuracy": [0.68, 0.63, 0.67, 0.66, 0.70, 0.67, 0.76, 0.71],
    },
    index=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8"]
)

nn_summary_df


# In[123]:


# Based on model performance, we recommend Model 7 and its Training & Testing Accuracy Visualization is shown as below
fig, ax = plt.subplots()
fig.patch.set_alpha(0.5)
results_df.plot(kind='bar', legend = False, figsize=(15,10), logy=True, colormap='ocean_r', ax = ax)
ax.patch.set_alpha(0.6)

ax.set_xlabel( 'Accuracy', rotation=0, fontsize=15, labelpad=20)
ax.set_ylabel( 'Train Iterations', rotation=90, fontsize=15, labelpad=20)

for label in ax.get_xticklabels():
    label.set_rotation(0)
    label.set_size(15)
        
plt.show()

