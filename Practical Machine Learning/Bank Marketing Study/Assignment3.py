#!/usr/bin/env python
# coding: utf-8

# In[1]:



RANDOM_SEED = 1

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import f1_score


# In[2]:


# initial work with the smaller data set
bank = pd.read_csv('bank.csv', sep = ';')  # start with smaller data set
# examine the shape of original input data
print(bank.shape)


# In[3]:


# drop observations with missing data, if any
bank.dropna()
# examine the shape of input data after dropping missing data
print(bank.shape)


# In[4]:


# look at the list of column names, note that y is the response
list(bank.columns.values)


# In[5]:


# look at the beginning of the DataFrame
bank.head()


# In[6]:


# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}


# In[7]:


# define binary variable for having credit in default
default = bank['default'].map(convert_to_binary)


# In[8]:


# define binary variable for having a mortgage or housing loan
housing = bank['housing'].map(convert_to_binary)


# In[9]:


# define binary variable for having a personal loan
loan = bank['loan'].map(convert_to_binary)


# In[10]:


# define response variable to use in the model
response = bank['response'].map(convert_to_binary)


# In[11]:


# Combine into model data frame
model_df = pd.DataFrame({
    'default': default,
    'housing': housing,
    'loan': loan,
    'response': response
})

model_df.head()


# ### Train - Test Split

# In[12]:


# Import sklearn train test split
from sklearn.model_selection import train_test_split


# In[13]:


# Split into train and test sets
train_set, test_set = train_test_split(model_df, test_size=0.2)


# In[14]:


# Verify that the split occurred
train_shape = train_set.shape
test_shape = test_set.shape

print('Train set shape = {0}'.format(train_shape))
print('Test set shape = {0}'.format(test_shape))


# In[15]:



train_set.head()


# In[16]:


# Create copy as to not modify train set during exploration
corr_df = train_set.copy()


# ### Correlation Matrix

# In[17]:


# Correlation Matrix
corr_df.corr()


# In[18]:


# Creat heatmap correlation matrix
sns.heatmap(corr_df.corr(), cmap="RdBu_r")


# In[19]:


# Define function to calculate response ratio
def response_ratio(x):
  
  return np.sum(x) / np.size(x)


# In[20]:



# Overall response ratio in our training data
resp_ratio = response_ratio(corr_df.response)
print('{0}% of respondents subscribed to long-term deposit.'.format(
    round(resp_ratio * 100, 2)))


# In[21]:


# Distributions
for i in corr_df.columns:
  ratio = response_ratio(corr_df[i])
  print('{0}% of respondents had a {1}'.format(round(ratio * 100, 1), i))


# ## Model 1: Logistic Regression
# #### Create Model Object and Fit to Data

# In[22]:



# Create model object
log_reg = LogisticRegression(C=10000)


# In[23]:



# Split the train df into x & y
x_train_lr = train_set[['default','housing', 'loan']]
y_train_lr = train_set['response']

# Fit the model
log_reg.fit(x_train_lr, y_train_lr.ravel())


# In[24]:


print('x training dataset shape: ',x_train_lr.shape)
print('y training dataset shape: ',y_train_lr.shape)


# In[25]:


# Review the coefficients
log_reg.coef_


# In[26]:



# Predict y on the train set
y_pred_train_lr = log_reg.predict(x_train_lr)


# #### Confusion Matrix

# In[27]:


# Create confusion matrix
cm = ConfusionMatrix(log_reg, classes=[0,1])
cm.score(x_train_lr, y_train_lr)
plt.tight_layout()


# 
# ### ROC Curve and AUC - Run Model Against Train Data

# In[28]:


# Calculate Scores
y_scores_train_lr = log_reg.decision_function(x_train_lr)

# Calculate fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_train_lr, y_scores_train_lr)

# Create roc curve plotting function
def plot_roc_curve(fpr, tpr, label=None):
  plt.plot(fpr, tpr, linewidth=2, label=label)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')

# Plot the ROC Curve
plot_roc_curve(fpr, tpr)
plt.title('ROC Curve for Logistic Regression')
plt.show()


# In[29]:


# Calculate the AUC
roc_auc_train_lr = roc_auc_score(y_train_lr,  y_scores_train_lr)
# Calculate the accuracy on the train set
print('Accuracy of logistic regression classifier on train set: {:.4f}'.format(log_reg.score(x_train_lr, y_train_lr)))
print('The area under the curve = {:.4f}'.format(roc_auc_train_lr))


# #### Cross Validation

# In[30]:


# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, x_train_lr=x_train_lr, y_train_lr=y_train_lr, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, x_train_lr, y_train_lr, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[31]:


# Run the cross validation
cross_val_multiple(log_reg)


# #### Run Model Against Test Data

# In[32]:


# Split the test df into x & y
x_test_lr = test_set[['housing', 'loan', 'default']]
y_test_lr = test_set['response']


# In[33]:


# Make Predictions on test data
y_test_pred_lr = log_reg.predict(x_test_lr)


# In[34]:


# View probabilities
log_reg.predict_proba(x_test_lr)


# ### Precision and Recall

# In[35]:


# Check precision and recall of model:
from sklearn.metrics import confusion_matrix, classification_report

# Predict test response:
y_pred_train_lr = log_reg.predict(x_train_lr)
y_pred_test_lr = log_reg.predict(x_test_lr)

print(classification_report(y_test_lr,y_pred_test_lr))


# ### ROC Curve and AUC - Run Model Against Test Data

# In[36]:


# Calculate Scores
y_scores_test_lr = log_reg.decision_function(x_test_lr)

# Calculate fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test_lr, y_scores_test_lr)

# Create roc curve plotting function
def plot_roc_curve(fpr, tpr, label=None):
  plt.plot(fpr, tpr, linewidth=2, label=label)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')

# Plot the ROC Curve
plot_roc_curve(fpr, tpr)
plt.title('ROC Curve for Logistic Regression')
plt.show()


# In[37]:



# Calculate Scores
y_scores_test_lr = log_reg.decision_function(x_test_lr)

# Calculate the AUC
roc_auc_test_lr = roc_auc_score(y_test_lr, y_scores_test_lr)


# In[38]:


print('Accuracy of logistic regression classifier on train set: {:.4f}'.format(log_reg.score(x_train_lr, y_train_lr)))
print('The area under the curve = {:.4f}'.format(roc_auc_train_lr))
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(log_reg.score(x_test_lr, y_test_lr)))
print('The area under the curve = {:3f}'.format(roc_auc_test_lr))


# #### Conclusion: Poor fit
# ##### I did not use the cross validation model on the test set due to no increase in accuracy.
# ##### Overall, the model performed as expected on the test data, but is still a poor predictor. 

# In[39]:


# BernoulliNB is designed for binary/boolean features
from sklearn.naive_bayes import BernoulliNB


# ## Model 2: Naive Bayes
# #### Create Model Object and Fit to Data

# In[40]:


# Create model object
nb = BernoulliNB()


# In[41]:


# Split the train df into x & y
x_train_nb = train_set[['default','housing', 'loan']]
y_train_nb = train_set['response']

# Fit the model
nb.fit(x_train_nb, y_train_nb)


# In[42]:


print('x training dataset shape: ',x_train_nb.shape)
print('y training dataset shape: ',y_train_nb.shape)


# In[43]:


# Predict y on the train set
y_pred_train_nb = nb.predict(x_train_nb)


# In[44]:


# View probabilities
nb.predict_proba(x_train_nb)


# #### Confusion Matrix

# In[45]:


# Create confusion matrix
cm = ConfusionMatrix(nb, classes=[0,1])
cm.score(x_train_nb, y_train_nb)
plt.tight_layout()


# ### ROC Curve and AUC - Run Model Against Train Data

# In[46]:


# Calculate Scores
y_scores_train_nb = nb.predict_proba(x_train_nb)

# Calculate fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_train_nb, y_scores_train_nb[:,1])

# Plot the ROC Curve
plot_roc_curve(fpr, tpr)
plt.title('ROC Curve for Naive Bayes')
plt.show()


# In[47]:


# Calculate the AUC
roc_auc_train_nb = roc_auc_score(y_train_nb, y_scores_train_nb[:,1])


# In[48]:


print('Accuracy of Naive Bayes classifier on train set: {:.4f}'.format(
    nb.score(x_train_nb, y_train_nb)))
print('The area under the curve = {:.4f}'.format(roc_auc_train_nb))


# #### Cross Validation

# In[49]:


# Run the cross validation
cross_val_multiple(nb)


# #### Run Model Against Test Data

# In[50]:


# Split the test df into x & y
x_test_nb = test_set[['housing', 'loan', 'default']]
y_test_nb = test_set['response']

# Fit the model
nb.fit(x_test_nb, y_test_nb)


# In[51]:


# Predict y on the train set
y_pred_test_nb = nb.predict(x_test_nb)


# In[52]:


# View probabilities
nb.predict_proba(x_test_nb)


# In[53]:


# Check precision and recall of model:
from sklearn.metrics import confusion_matrix, classification_report

# Predict test response:
y_pred_train_nb = nb.predict(x_train_nb)
y_pred_test_nb = nb.predict(x_test_nb)

print(classification_report(y_test_nb,y_pred_test_nb))


# ### ROC Curve and AUC - Run Model Against Test Data

# In[54]:


# Calculate Scores
y_scores_test_nb = nb.predict_proba(x_test_nb)

# Calculate fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test_nb, y_scores_test_nb[:,1])

# Plot the ROC Curve
plot_roc_curve(fpr, tpr)
plt.title('ROC Curve for Naive Bayes')
plt.show()


# In[55]:


# Calculate the AUC
roc_auc_test_nb = roc_auc_score(y_test_nb, y_scores_test_nb[:,1])


# In[56]:


print('Accuracy of Naive Bayes classifier on train set: {:.4f}'.format(
    nb.score(x_train_nb, y_train_nb)))
print('The area under the curve = {:.4f}'.format(roc_auc_train_nb))

print('Accuracy of naive bayes classifier on test set: {:.4f}'.format(nb.score(x_test_nb, y_test_nb)))
print('The area under the curve = {:.4f}'.format(roc_auc_test_nb))


# #### Conculsion: Not a great fit, but better than Logistic Regression Model
# ##### Accuracy was similar for both models (.88), but NB had a higher AUC than Logistic Regression
