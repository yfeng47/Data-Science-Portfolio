#!/usr/bin/env python
# coding: utf-8

# ## MSDS422 Assignment 5 - Principal Component Analysis
# 
# **Table of contents:**
# 
# *   System & Data Setup (MNIST)
# *   Model 1: Random Forest Fit w/ out PCA
# *   Model 2: Principal Component Analysis
# *   Model 3: Random Forest w/ reduced data set
# *   Analysis: Model Comparison and Timing
# *   Model 4: re-run dimension reduction on training data & re-run model
# *   Conclusion

# ## Setup

# In[1]:


from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from sklearn.model_selection import train_test_split


# ### MNIST Dataset

# In[2]:


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# In[3]:


# Load MNIST dataset
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
mnist["data"], mnist["target"]


# In[4]:


X, y = mnist["data"], mnist["target"]
X.shape


# In[5]:


y.shape


# In[6]:


# Plot digits and check data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")


# In[7]:


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# In[8]:


# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


# In[9]:


plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
#save_fig("more_digits_plot")
plt.show()


# In[10]:


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[11]:


y_test.shape


# In[12]:


import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# ### Model 1: Random Forest Classifier - all variables

# In[14]:


get_ipython().run_cell_magic('time', '', "start = time.process_time()\n\nfrom sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier(max_features = 'sqrt', n_estimators=784)\nclf.fit(X_train, y_train)\nduration1 = time.process_time() - start")


# In[15]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\nfrom sklearn.model_selection import cross_val_predict\ny_scores = cross_val_predict(clf, X_train, y_train)\nduration2 = time.process_time() - start')


# In[16]:


y_scores.shape


# In[17]:


# Access classification performance on training data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,                                classification_report, confusion_matrix
print('f1 score = {0}'.format(f1_score(y_train, y_scores, average="macro")))
print('precision = {0}'.format(precision_score(y_train, y_scores, average="macro")))
print('recall = {0}'.format(recall_score(y_train, y_scores, average="macro")))


# In[18]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\ny_test_scores = cross_val_predict(clf, X_test, y_test)\nduration3 = time.process_time() - start\ny_test.shape')


# In[19]:


# Access classification performance on testing data
print('f1 score = {0}'.format(f1_score(y_test, y_test_scores, average="macro")))
print('precision = {0}'.format(precision_score(y_test, y_test_scores, average="macro")))
print('recall = {0}'.format(recall_score(y_test, y_test_scores, average="macro")))


# In[20]:


from sklearn.metrics import confusion_matrix
conf_mx1 = confusion_matrix(y_test, y_test_scores)
print(conf_mx1)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mx1)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[21]:


row_sums = conf_mx1.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx1 / row_sums
np.fill_diagonal(norm_conf_mx, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(norm_conf_mx)
plt.title('Confusion matrix errors')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[22]:


Model1 = duration1+duration2+duration3


# ## Model 2: Principal Components Analysis (PCA)
# 

# In[24]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import PCA\nstart = time.process_time()\npca = PCA()\npca.fit(mnist["data"])\ncumsum = np.cumsum(pca.explained_variance_ratio_)\nd = np.argmax(cumsum >= 0.95) + 1\nduration4 = time.process_time() - start')


# In[25]:


#principle components from PCA
d


# In[26]:


# Plot the PCA Cumulative Variance Sum Curve
x_len = len(pca.explained_variance_ratio_)
x = np.linspace(1, x_len, x_len, dtype=int)
plt.plot(x, cumsum)
plt.hlines(y=cumsum[d], xmin=0, xmax=d, linestyles="dashed")
plt.vlines(x=d, ymin=0, ymax=cumsum[d], linestyles="dashed")
plt.title("Dimensions vs Explained Variance")
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.annotate(s="{}".format(d), xy=(d+10, cumsum[d]-0.05))


# ### Model 3: Random Forest - reduced data set

# In[27]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\npca = PCA(n_components = 154)\nX_reduced = pca.fit_transform(X_train)\nX_test_reduced = pca.fit_transform(X_test)\nduration5 = time.process_time() - start')


# In[28]:


X_reduced.shape


# In[29]:


get_ipython().run_cell_magic('time', '', "start = time.process_time()\nclf_2 = RandomForestClassifier(max_features = 'sqrt',n_estimators=154)\nclf_2.fit(X_reduced, y_train)\nduration6 = time.process_time() - start")


# In[30]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\ny_scores_2 = cross_val_predict(clf_2, X_reduced, y_train)\nduration7 = time.process_time() - start')


# In[31]:


Model2= duration4 + duration5 + duration6 + duration7


# In[32]:



y_scores_2.shape


# In[33]:


# Access classification performance on training data
print('f1 score = {0}'.format(f1_score(y_train, y_scores_2, average="macro")))
print('precision = {0}'.format(precision_score(y_train, y_scores_2, average="macro")))
print('recall = {0}'.format(recall_score(y_train, y_scores_2, average="macro")))


# In[34]:


y_test_scores_2 = cross_val_predict(clf_2, X_test_reduced, y_test)
y_test.shape


# In[35]:


# Access classification performance on testing data
print('f1 score = {0}'.format(f1_score(y_test, y_test_scores_2, average="macro")))
print('precision score = {0}'.format(precision_score(y_test, y_test_scores_2, average="macro")))
print('recall = {0}'.format(recall_score(y_test, y_test_scores_2, average="macro")))


# In[36]:


conf_mx2 = confusion_matrix(y_test, y_test_scores_2)
conf_mx2


# In[37]:


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mx2)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[38]:


row_sums = conf_mx2.sum(axis=1, keepdims=True)
norm_conf_mx2 = conf_mx2 / row_sums
np.fill_diagonal(norm_conf_mx2, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(norm_conf_mx2)
plt.title('Confusion matrix errors')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# #### The PCA Model has a design flaw because the experiment is applying principal component analysis (PCA) to the entire set of data. The dataset should be split into train and test sets prior to running PCA. To correct the flaw, the train set will be split into a train and validation set. The PCA will be run on these sets prior to making predictions on the test set
# 
# ### Model 4: re-run dimension reduction on training data & re-run model

# In[40]:


pca_2 = PCA()
pca_2.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d2 = np.argmax(cumsum >= 0.95) + 1


# In[41]:


d2


# In[42]:


# Plot the PCA Cumulative Variance Sum Curve
x_len = len(pca.explained_variance_ratio_)
x = np.linspace(1, x_len, x_len, dtype=int)
plt.plot(x, cumsum)
plt.hlines(y=cumsum[d2], xmin=0, xmax=d2, linestyles="dashed")
plt.vlines(x=d2, ymin=0, ymax=cumsum[d2], linestyles="dashed")
plt.title("Dimensions vs Explained Variance")
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.annotate(s="{}".format(d2), xy=(d2+10, cumsum[d2]-0.05))


# In[43]:


get_ipython().run_cell_magic('time', '', 'start = time.process_time()\npca_3 = PCA(n_components = 149)\nX_train_reduced_3 = pca_3.fit_transform(X_train)\nX_test_reduced_3 = pca_3.fit_transform(X_test)\nduration8 = time.process_time() - start')


# In[44]:


get_ipython().run_cell_magic('time', '', "start = time.process_time()\nclf_3 = RandomForestClassifier(max_features = 'sqrt', n_estimators=149)\nclf_3.fit(X_train_reduced_3, y_train)\nduration9 = time.process_time() - start")


# In[45]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import cross_val_predict\nstart = time.process_time()\ny_train_scores_3 = cross_val_predict(clf_3, X_train_reduced_3, y_train)\ny_test_scores_3 = cross_val_predict(clf_3, X_test_reduced_3, y_test)\nduration10 = time.process_time() - start')


# In[46]:


print(f1_score(y_train, y_train_scores_3, average="macro"))
print(precision_score(y_train, y_train_scores_3, average="macro"))
print(recall_score(y_train, y_train_scores_3, average="macro"))


# In[47]:


print(f1_score(y_test, y_test_scores_3, average="macro"))
print(precision_score(y_test, y_test_scores_3, average="macro"))
print(recall_score(y_test, y_test_scores_3, average="macro"))


# In[48]:


conf_mx3 = confusion_matrix(y_test, y_test_scores_3)
conf_mx3


# In[49]:


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mx3)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[50]:


row_sums = conf_mx3.sum(axis=1, keepdims=True)
norm_conf_mx3 = conf_mx3 / row_sums
np.fill_diagonal(norm_conf_mx3, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(norm_conf_mx3)
plt.title('Confusion matrix errors')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[51]:


Model3=duration8+duration9+duration10


# In[52]:


def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%02d:%02d" % (minutes, seconds)


# In[53]:


# Driver program 
m1 = Model1
m2 = Model2
m3 = Model3
print(convert(m1)) 
print(convert(m2)) 
print(convert(m3)) 


# ### Analysis - Model comparison & timing

# In[55]:


print('Model 1 Original 784-Varibiable Model, Test: = {0}'.format(f1_score(y_test, y_test_scores, average="macro"))),
print('Model 1 Time:')
print(convert(m1)) 
print('Model 3 95% PCA Model, Test: {0}'.format(f1_score(y_test, y_test_scores_2, average="macro")))
print('Model 3 Time:')
print(convert(m2)) 
print('Model 4 Random Forest with reduced variables and PCA, Test: {0}'.format(f1_score(y_test, y_test_scores_3, average="macro")))
print('Model 4 Time:')
print(convert(m3))


# # Conclusion
# 
# **Recommendation to reduce data dimensions using PCA on the training data set (model 3).**  Resulting Random Forest model uses 149 estimators.  The F1 and Precision scores of the recommended model are .94 accuracy on the training data, and .92 accuracy on the test data.  The time it takes to run the model is ~10 minutes, which is a 60% reduction in timing for a slightly improved test result (model 1).
