#!/usr/bin/env python
# coding: utf-8

# ## MSDS 422 Assignment #6 - Artificial Neural Networks
# 
# **Table of contents:**
# 
# * System & Data Setup
# * Model 1 - 2 layer model - 10 neurons per layer
# * Model 2 - 2 layer model - 20 neurons per layer
# * Model 3 - 5 layer model - 10 neurons per layer
# * Model 4 - 5 layer model - 20 neurons per layer
# * Model 5 - 2 layer model - 300 neurons per layer
# * Model 6 - 3 layer model - 300 neurons per layer
# * Model 7 - 2 layer model - 150 neurons per layer
# * Model 8 - 3 layer model - 150 neurons per layer
# * Conclusion

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

#suppress tf.logging
import logging
logging.getLogger('tensorflow').disabled = True

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import time
import warnings
import datetime

#Scikit-Learn
from sklearn.preprocessing import StandardScaler


# In[3]:


print(tf.version.VERSION)


# In[4]:


#split train & test data
(X_train1, y_train), (X_test1, y_test) = tf.keras.datasets.mnist.load_data()
X_train1 = X_train1.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test1 = X_test1.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)


# In[5]:


#test split 'train'
X_train1.shape


# In[6]:


#test split 'test'
X_test1.shape


# In[7]:


#check data in training set as visual:
print('Training data shape', X_train1.shape)
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(X_train1[0].reshape(28, 28), cmap=plt.cm.Greys);
ax2.imshow(X_train1[1].reshape(28, 28), cmap=plt.cm.Greys);


# In[8]:


#define confusion matrix plot
def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


# In[9]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop


# In[9]:


# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X_train1))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
X_train = scaler.fit_transform(X_train1)


# In[10]:


# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X_test1))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
X_test = scaler.fit_transform(X_test1)


# ### Model 1

# In[10]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf1 = tf.estimator.DNNClassifier(hidden_units=[10,10], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn1 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf1.train(input_fn=input_fn1)\nduration1 = time.process_time() - start')


# In[11]:


#evaluate training accuracy (2a)
train1_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train1 = dnn_clf1.evaluate(input_fn=train1_input_fn)
eval_results_train1


# In[12]:


a1_train = eval_results_train1['accuracy']


# In[13]:


#evaluate test accuracy (2a)
test1_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test1 = dnn_clf1.evaluate(input_fn=test1_input_fn)
eval_results_test1


# ### Model 2

# In[14]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf2 = tf.estimator.DNNClassifier(hidden_units=[20,20], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn2 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf2.train(input_fn=input_fn2)\nduration2 = time.process_time() - start')


# In[15]:


#evaluate training accuracy (2b)
train2_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train2 = dnn_clf2.evaluate(input_fn=train2_input_fn)
eval_results_train2


# In[16]:


#evaluate test accuracy (2b)
test2_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test2 = dnn_clf2.evaluate(input_fn=test2_input_fn)
eval_results_test2


# ### Model 3

# In[17]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf3 = tf.estimator.DNNClassifier(hidden_units=[10,10,10,10,10], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn3 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf3.train(input_fn=input_fn3)\nduration3 = time.process_time() - start')


# In[18]:


#evaluate training accuracy (5a)
train3_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train3 = dnn_clf3.evaluate(input_fn=train3_input_fn)
eval_results_train3


# In[19]:


#evaluate test accuracy (5a)
test3_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test3 = dnn_clf3.evaluate(input_fn=test3_input_fn)
eval_results_test3


# ### Model 4

# In[20]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf4 = tf.estimator.DNNClassifier(hidden_units=[20,20,20,20,20], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn4 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf4.train(input_fn=input_fn4)\nduration4 = time.process_time() - start')


# In[21]:


#evaluate training accuracy (5a)
train4_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train4 = dnn_clf4.evaluate(input_fn=train4_input_fn)
eval_results_train4


# In[22]:


#evaluate test accuracy (5a)
test4_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test4 = dnn_clf4.evaluate(input_fn=test4_input_fn)
eval_results_test4


# ### Model 5

# In[26]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf5 = tf.estimator.DNNClassifier(hidden_units=[300,300], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn5 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf5.train(input_fn=input_fn5)\nduration5 = time.process_time() - start')


# In[27]:


#evaluate training accuracy (5a)
train5_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train5 = dnn_clf5.evaluate(input_fn=train5_input_fn)
eval_results_train5


# In[28]:


#evaluate test accuracy (5a)
test5_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test5 = dnn_clf5.evaluate(input_fn=test5_input_fn)
eval_results_test5


# ### Model 6

# In[29]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf6 = tf.estimator.DNNClassifier(hidden_units=[300,300,300], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn6 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf6.train(input_fn=input_fn6)\nduration6 = time.process_time() - start')


# In[30]:


#evaluate training accuracy (5a)
train6_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train6 = dnn_clf6.evaluate(input_fn=train6_input_fn)
eval_results_train6


# In[31]:


#evaluate test accuracy (5a)
test6_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test6 = dnn_clf6.evaluate(input_fn=test6_input_fn)
eval_results_test6


# ### Model 7

# In[32]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf7 = tf.estimator.DNNClassifier(hidden_units=[150,150], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn7 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf7.train(input_fn=input_fn7)\nduration7 = time.process_time() - start')


# In[34]:


#evaluate training accuracy (5a)
train7_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train7 = dnn_clf7.evaluate(input_fn=train7_input_fn)
eval_results_train7


# In[35]:


#evaluate test accuracy (5a)
test7_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test7 = dnn_clf7.evaluate(input_fn=test7_input_fn)
eval_results_test7


# ### Model 8

# In[36]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfeature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]\ndnn_clf8 = tf.estimator.DNNClassifier(hidden_units=[150,150,150], n_classes=10,\n                                     feature_columns=feature_cols)\n\ninput_fn8 = tf.estimator.inputs.numpy_input_fn(\n    x={"X": X_train}, y=y_train, num_epochs=20, batch_size=100, shuffle=True)\ndnn_clf8.train(input_fn=input_fn8)\nduration8 = time.process_time() - start')


# In[37]:


#evaluate training accuracy (5a)
train8_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, shuffle=False)
eval_results_train8 = dnn_clf8.evaluate(input_fn=train8_input_fn)
eval_results_train8


# In[38]:


#evaluate test accuracy (5a)
test8_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results_test8 = dnn_clf8.evaluate(input_fn=test8_input_fn)
eval_results_test8


# In[39]:


a1_train = eval_results_train1['accuracy']
a1_test = eval_results_test1['accuracy']
a2_train = eval_results_train2['accuracy']
a2_test = eval_results_test2['accuracy']
a3_train = eval_results_train3['accuracy']
a3_test = eval_results_test3['accuracy']
a4_train = eval_results_train4['accuracy']
a4_test = eval_results_test4['accuracy']
a5_train = eval_results_train5['accuracy']
a5_test = eval_results_test5['accuracy']
a6_train = eval_results_train6['accuracy']
a6_test = eval_results_test6['accuracy']
a7_train = eval_results_train7['accuracy']
a7_test = eval_results_test7['accuracy']
a8_train = eval_results_train8['accuracy']
a8_test = eval_results_test8['accuracy']

def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%02d:%02d" % (minutes, seconds)

d1 = convert(duration1)
d2 = convert(duration2)
d3 = convert(duration3)
d4 = convert(duration4)
d5 = convert(duration5)
d6 = convert(duration6)
d7 = convert(duration7)
d8 = convert(duration8)


# In[41]:


nn_summary_df = pd.DataFrame(
    {
        "Number Of Layers": [2, 2, 5, 5, 2, 3, 2, 3],
        "Nodes Per Layer": [10, 20, 10, 20, 300, 300, 150, 150],
        "Processing Time": [d1, d2, d3, d4, d5, d6, d7, d8],
        "Training Set Accuracy": [a1_train, a2_train, a3_train, a4_train, a5_train, a6_train, a7_train, a8_train],
        "Test Set Accuracy": [a1_test, a2_test, a3_test, a4_test, a5_test, a6_test, a7_test, a8_test]
    },
    index=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8"]
)

nn_summary_df

