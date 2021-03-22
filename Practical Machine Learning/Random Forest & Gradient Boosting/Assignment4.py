#!/usr/bin/env python
# coding: utf-8

# ## MSDS 422 Assignment 4 - Decision Trees & Random Forests

# ## Data Import

# In[2]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True


# In[3]:


# import base packages into the namespace for this program
import numpy as np
import pandas as pd

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split
from math import sqrt  # for root mean-squared error calculation
import matplotlib # import matplotlib
import matplotlib.pyplot as plt  # static plotting
import seaborn as sns  # pretty plotting, including heat map

from pandas.plotting import scatter_matrix  # scatter plot matrix
from scipy.stats import uniform  # for training-and-test split
import statsmodels.api as sm  # statistical models (including regression)
import statsmodels.formula.api as smf  # R-like model specification
from sklearn.tree import DecisionTreeRegressor  # machine learning tree
from sklearn.ensemble import RandomForestRegressor # ensemble method

# suppress warning messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[4]:


# read data for the Boston Housing Study
# creating data frame restdata
boston = pd.read_csv('boston.csv')


# In[5]:


#define function 'plot_feature_importances_boston'
def plot_feature_importances_boston(model):
    n_features = Log_model_X.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), Log_model_X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


# In[6]:


# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T


# In[7]:


# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)


# ## Data Exploration & Basic Transformation

# ### Basics - Data Frame, Data Types, re-name columns & replace missing values

# In[8]:


boston = pd.DataFrame(data=boston)
boston.tail()


# In[9]:


boston.dtypes


# In[10]:


boston['neighborhood'].value_counts(); #remove ';' if you want to see result


# In[11]:


boston.columns


# In[12]:


# rename columns to provide more insight to data save as new DataFrame
df_boston = boston.copy()

df_boston = df_boston.rename(index=str, columns={
    'crim': 'crime_rate',
    'zn': 'zoned',
    'indus': 'industrial',
    'chas': 'charles_binary',
    'rooms': 'avg_rooms',
    'age': 'pct_pre1940',
    'dis': 'distance_center',
    'rad': 'highway_access',
    'tax': 'avg_tax',
    'lstat': 'pct_lowIncome',
    'mv': 'response_mv'})

df_boston.columns


# In[13]:


#replace any missing values with mean of column just in case

list = ['crime_rate', 'zoned', 'industrial', 'charles_binary',
       'nox', 'avg_rooms', 'pct_pre1940', 'distance_center', 'highway_access',
       'avg_tax', 'ptratio', 'pct_lowIncome', 'response_mv']

#loop through list of continuous variabls and fill nan values with mean value
for i in list:
    df_boston[i] = df_boston[i].fillna((df_boston[i].mean()))
    
df_boston.describe()


# ### Distplot distributions (High Range & Low Range Variables)

# In[14]:


#plot distribution of column values together in one figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('distplot Distribution of Column Values (% variables)', size = 16, y=.9)
sns.distplot(df_boston["crime_rate"] , color="orange", ax=axes[0, 0])
sns.distplot(df_boston["zoned"] , color="purple", ax=axes[0, 1])
sns.distplot(df_boston["industrial"] , color="blue", ax=axes[1, 0])
sns.distplot(df_boston["pct_pre1940"] , color="pink", ax=axes[1, 1])
f.savefig('distplot Distribution of Column Values (% variables)' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[15]:


#plot distribution of column values together in one figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('distplot Distribution of Column Values (low range variables)', size = 16, y=.9)
sns.distplot(df_boston["avg_rooms"] , color="purple", ax=axes[0, 0])
sns.distplot(df_boston["distance_center"] , color="orange", ax=axes[0, 1])
sns.distplot(df_boston["highway_access"] , color="pink", ax=axes[1, 0])
sns.distplot(df_boston["ptratio"] , color="blue", ax=axes[1, 1])
f.savefig('distplot Distribution of Column Values (low range variables)' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[16]:


sns.distplot(df_boston["avg_tax"] , color="green")


# ### Explore regression & correlation - continuous variables

# In[17]:


# plot linear relationships with response variable 'response_mv' together in one figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('Linear relationships to response variable response_mv (% variables)', size = 16, y=.9)
sns.regplot(x="crime_rate", y="response_mv", data=df_boston, x_jitter=.1, color = "orange", ax=axes[0, 0])
sns.regplot(x="zoned", y="response_mv", data=df_boston, x_jitter=.1, color = "purple", ax=axes[0, 1])
sns.regplot(x="industrial", y="response_mv", data=df_boston, x_jitter=.1, color = "blue", ax=axes[1, 0])
sns.regplot(x="pct_pre1940", y="response_mv", data=df_boston, x_jitter=.1, color = "pink", ax=axes[1, 1])
f.savefig('Linear relationships to response variable response_mv (% variables)' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[18]:


# plot linear relationships with response variable 'response_mv' together in one figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('Linear relationships to response variable response_mv (low range variables)', size = 16, y=.9)
sns.regplot(x="avg_rooms", y="response_mv", data=df_boston, x_jitter=.1, color = "purple", ax=axes[0, 0])
sns.regplot(x="distance_center", y="response_mv", data=df_boston, x_jitter=.1, color = "orange", ax=axes[0, 1])
sns.regplot(x="highway_access", y="response_mv", data=df_boston, x_jitter=.1, color = "pink", ax=axes[1, 0])
sns.regplot(x="ptratio", y="response_mv", data=df_boston, x_jitter=.1, color = "blue", ax=axes[1, 1])
f.savefig('Linear relationships to response response_mv (low range variables)' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[19]:


sns.regplot(x="avg_tax", y="response_mv", data=df_boston, x_jitter=.1, color = "green")


# ### plot correlation - continuous variables

# In[20]:


# correlation heat map setup for seaborn
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)      

np.set_printoptions(precision=3)


# In[21]:


# define subset DataFrame for analysis of software preferences 
Selected = ['crime_rate', 'zoned', 'industrial', 'charles_binary',
       'nox', 'avg_rooms', 'pct_pre1940', 'distance_center', 'highway_access',
       'avg_tax', 'ptratio', 'pct_lowIncome', 'response_mv']

correlation_df = df_boston[Selected]

# examine intercorrelations among software preference variables
# with correlation matrix/heat map
corr_chart(df_corr = correlation_df)  


# ## Model 1- Model exploration with data transformation (log)
# 
# ### Model 1a - Scaled Lasso Regression (Log)
# 
# > #### Training Score: .78 - Test Score: .80 (11 features used)
# Observation - poor fit & overly complex
# ### Model 1b - Decision Tree Regression Model
# 
# > #### Training Score: .92 - Test Score: .68 (Max Depth = 5)
# Observation - as max depth goes up, model is either over-simplified or over fit. 
# ### Model 1c - Random Forest Regression Model
# 
# >#### Training Score: .91 - Test Score: .85 (Max Depth = 4)
# Observation - improved fit and less gap between train & test scores.
# High feature emphasis on 'pct_lowIncome'
# ### Model 1d - Gradient Boosting for best fit
# 
# >#### Training Score: .95 - Test Score: .86 (Max Depth = 2)
# Observation - 'best' fit for transformed data set and good balance of emphasis across features

# ### Model 1 - Data Prep & Training/Test split
# #### Log transformation to data & response

# In[22]:


df_boston2 = df_boston.copy()
df_boston2 = df_boston2.drop(columns=['neighborhood'])


# In[23]:


df_Boston_log = df_boston2.copy()
column_titles = df_boston2.columns

from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)

df_Boston_log = transformer.transform(df_boston2)
df_Boston_log = pd.DataFrame(df_Boston_log, columns = column_titles)
df_Boston_log.describe()


# In[24]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14) 


# In[25]:


Log_model_X = df_Boston_log.copy()
Log_model_y = df_Boston_log.copy()

# split dataframe by data and response variables and make array for train_test_split
#remove response variable from data and save as df_data
Log_model_X = Log_model_X.drop(columns=['response_mv'])
#remove all but response variable and save as df_response
Log_model_y = pd.DataFrame(Log_model_y['response_mv'])

#split data by training & test sets
X_train_Log, X_test_Log, y_train_Log, y_test_Log = train_test_split(    Log_model_X, Log_model_y,test_size=.3, random_state=10)


# In[26]:


X_train_Log.head(); #test split X


# In[27]:


y_train_Log.head(); #test split y


# ### Model 1a Scaled Lasso Regression (Log)
# #### Training Score: .78 - Test Score: .80 (11 features used)
# Observation - poor fit & overly complex

# In[77]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso01 = Lasso(max_iter = 10000).fit(X_train_Log,y_train_Log)

alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
grid = GridSearchCV(estimator=lasso01,param_grid=dict(alpha=alphas))
grid.fit(X_train_Log, y_train_Log)

print("Alpha level: {:.2f}".format(grid.best_estimator_.alpha))
print("Training score: {:.2f}".format(lasso01.score(X_train_Log,y_train_Log)))
print("Test score: {:.2f}".format(lasso01.score(X_test_Log,y_test_Log)))
print("Number of features used: {}".format(np.sum(lasso01.coef_ !=0)))

y_L1_res = y_test_Log.copy()
y_L1_res['predicted']=lasso01.predict(X_test_Log)

# Plot the residuals after fitting a lasso model
sns.residplot(y_L1_res['predicted'], y_L1_res['response_mv'], lowess=True, color="orange")


# In[29]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log, lasso01.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log, lasso01.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log, lasso01.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log, lasso01.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[30]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[31]:


# Run the cross validation
cross_val_multiple(lasso01)


# ### Model 1b Decision Tree
# #### Training Score: .92 - Test Score: .68 (Max Depth = 5)
# Observation - as max depth goes up, model is either over-simplified or over fit.

# In[32]:


tree_model_maker = DecisionTreeRegressor(random_state = 9999, max_depth = 5)


# In[33]:


# fit regression tree using model 1 training/test split
tree_model_fit = tree_model_maker.fit(X_train_Log, y_train_Log)


# In[36]:


# compute the proportion of response variance for training data
X_train_Log_chk = X_train_Log.copy()

X_train_Log_chk['m1_Tree_predicted'] =    tree_model_fit.predict(X_train_Log)
full_tree_train_result =     round(np.power(y_train_Log['response_mv']        .corr(X_train_Log_chk['m1_Tree_predicted']),2),3)
print('\nFull Tree Proportion of Training Set Variance Accounted for: ',    full_tree_train_result)

# compute the proportion of response variance for test data
X_test_Log_chk = X_test_Log.copy()

X_test_Log_chk['m1_Tree_predicted'] =    tree_model_fit.predict(X_test_Log)
full_tree_test_result =     round(np.power(y_test_Log['response_mv']        .corr(X_test_Log_chk['m1_Tree_predicted']),2),3)
print('\nFull Tree Proportion of Test Set Variance Accounted for: ',    full_tree_test_result)


# In[37]:


print("Training set score: {:.2f}".format(tree_model_fit.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(tree_model_fit.score(X_test_Log,y_test_Log)))
#print("Number of features used: {}".format(np.sum(tree_model_fit.coef_ !=0)))

y_Tree1_chk = y_test_Log.copy()
y_Tree1_chk['predicted']=tree_model_fit.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_Tree1_chk['predicted'], y_Tree1_chk['response_mv'], lowess=True, color="orange")


# In[38]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log,tree_model_fit.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log,tree_model_fit.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log,tree_model_fit.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log,tree_model_fit.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[39]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[40]:


# Run the cross validation
cross_val_multiple(tree_model_fit)


# In[41]:


def plot_feature_importances_boston(model):
    n_features = X_train_Log.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train_Log.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_boston(tree_model_fit)


# ### Model 1c - Random Forest Regressor
# #### Training Score: .91 - Test Score: .85 (Max Depth = 4)
# Observation - improved fit and less gap between train & test scores.
# High feature emphasis on 'pct_lowIncome'

# In[42]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

RForest1 = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)

RForest1.fit(X_train_Log,y_train_Log)

print("Training set score: {:.2f}".format(RForest1.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(RForest1.score(X_test_Log,y_test_Log)))
#print("Number of features used: {}".format(np.sum(tree_model_fit.coef_ !=0)))

y_RForest1_chk = y_test_Log.copy()
y_RForest1_chk['predicted']=RForest1.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_RForest1_chk['predicted'], y_RForest1_chk['response_mv'], lowess=True, color="orange")


# In[43]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log,RForest1.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log,RForest1.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log,RForest1.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log,RForest1.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[44]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[45]:


# Run the cross validation
cross_val_multiple(RForest1)


# In[46]:


plot_feature_importances_boston(RForest1)


# ### Model 1d Gradient Boosting
# #### Training Score: .95 - Test Score: .86 (Max Depth = 2)
# Observation - 'best' fit for transformed data
# improved distribution between feature emphasis

# In[47]:


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train_Log,y_train_Log)

errors = [mean_squared_error(y_test_Log, y_pred)
          for y_pred in gbrt.staged_predict(X_test_Log)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train_Log,y_train_Log)


# In[48]:


print("Training set score: {:.2f}".format(gbrt_best.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(gbrt_best.score(X_test_Log,y_test_Log)))
#print("Number of features used: {}".format(np.sum(tree_model_fit.coef_ !=0)))

y_gbrt_best_chk = y_test_Log.copy()
y_gbrt_best_chk['predicted']= gbrt_best.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_gbrt_best_chk['predicted'], y_gbrt_best_chk['response_mv'], lowess=True, color="orange")


# In[49]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log,gbrt_best.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log,gbrt_best.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log,gbrt_best.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log,gbrt_best.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[50]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[51]:


# Run the cross validation
cross_val_multiple(gbrt_best)


# In[52]:


plot_feature_importances_boston(gbrt_best)


# ## Model 2 - Model exploration with data transformation (log)
# 
# ### Model 2a - Regularized Linear Model (Ridge)
# 
# > #### Training Score: .78 - Test Score: .81 (12 features used)
# Observation - poor fit & overly complex
# 
# ### Model 2b - Decision Tree Regression Model
# 
# 
# > #### Training Score: .87 - Test Score: .66 (Max Depth 4)
# Observation - as max depth goes up, model is either over-simplified or over fit. 
# 
# 
# ### Model 2c - Random Forest Regression Model
# 
# 
# > #### Training Score: .91 - Test Score: .85 (Max Depth 4)
# Observation - improvement on decision tree model - high feature emphasis on 'pct_lowIncome'
# 
# 
# ### Model 2d - Gradient Boosting for best fit
# 
# 
# >#### Training Score: .94 - Test Score: .86 (Max Depth 2)
# Observation - 'best' fit for transformed data set and good balance of emphasis across features
# 

# 

# ### Model 2a - Scaled Ridge Regression (Log)
# #### Training Score: .78 - Test Score: .81 (12 features used)
# Observation - poor fit & overly complex

# In[53]:


from sklearn.linear_model import Ridge

ridge2 = Ridge(solver="cholesky").fit(X_train_Log,y_train_Log)


alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
grid = GridSearchCV(estimator=ridge2,param_grid=dict(alpha=alphas))
grid.fit(X_train_Log,y_train_Log)

print("Alpha level: {:.2f}".format(grid.best_estimator_.alpha))
print("Training set score: {:.2f}".format(ridge2.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(ridge2.score(X_test_Log,y_test_Log)))
print("Number of features used: {}".format(np.sum(ridge2.coef_ !=0)))

y_R2_chk = y_test_Log.copy()
y_R2_chk['predicted']=ridge2.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_R2_chk['predicted'], y_R2_chk['response_mv'], lowess=True, color="purple")


# In[54]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log, ridge2.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log, ridge2.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log, ridge2.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log, ridge2.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[55]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[56]:


# Run the cross validation
cross_val_multiple(ridge2)


# ### Model 2b - Decision Tree Regression (Log)
# #### Training Score: .87 - Test Score: .66 (Max Depth 4)
# Observation - as max depth goes up, model is either over-simplified or over fit.. 

# In[57]:


tree_model_maker2 = DecisionTreeRegressor(random_state = 9999, max_depth = 4)


# In[58]:


# fit regression tree using model 1 training/test split
tree_model_fit2 = tree_model_maker2.fit(X_train_Log,y_train_Log)


# In[59]:


print("Training set score: {:.2f}".format(tree_model_fit2.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(tree_model_fit2.score(X_test_Log,y_test_Log)))
#print("Number of features used: {}".format(np.sum(tree_model_fit.coef_ !=0)))

y_Tree2_chk = y_test_Log.copy()

y_Tree2_chk['predicted']=tree_model_fit2.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_Tree2_chk['predicted'], y_Tree2_chk['response_mv'], lowess=True, color="purple")


# In[60]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log,tree_model_fit2.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log,tree_model_fit2.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log,tree_model_fit2.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log,tree_model_fit2.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[61]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[62]:


# Run the cross validation
cross_val_multiple(tree_model_fit2)


# In[63]:


print("Feature importances:\n{}".format(tree_model_fit2.feature_importances_))


# In[64]:


plot_feature_importances_boston(tree_model_fit2)


# ### Model 2c Random Forest
# #### Training Score: .91 - Test Score: .85 (Max Depth 4)
# Observation - improvement on decision tree model - high feature emphasis on 'pct_lowIncome'

# In[65]:


RForest2 = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)

RForest2.fit(X_train_Log,y_train_Log)

print("Training set score: {:.2f}".format(RForest2.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(RForest2.score(X_test_Log,y_test_Log)))
#print("Number of features used: {}".format(np.sum(tree_model_fit.coef_ !=0)))

y_RForest2_chk = y_test_Log.copy()
y_RForest2_chk['predicted']=RForest2.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_RForest2_chk['predicted'], y_RForest2_chk['response_mv'], lowess=True, color="purple")


# In[66]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log,RForest2.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log,RForest2.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log,RForest2.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log,RForest2.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[67]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, X_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[68]:


# Run the cross validation
cross_val_multiple(RForest2)


# In[69]:


plot_feature_importances_boston(RForest2)


# ### Model 2d Gradient Boosting
# #### Training Score: .94 - Test Score: .86 (Max Depth 2)
# Observation - 'best' fit for transformed data set and good balance of emphasis across features

# In[70]:


gbrt2 = GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=42)
gbrt2.fit(X_train_Log,y_train_Log)


# In[71]:


print("Training set score: {:.2f}".format(gbrt2.score(X_train_Log,y_train_Log)))
print("Test set score: {:.2f}".format(gbrt2.score(X_test_Log,y_test_Log)))

y_gbrt2_chk = y_test_Log.copy()
y_gbrt2_chk['predicted']=gbrt2.predict(X_test_Log)
# Plot the residuals after fitting a linear model
sns.residplot(y_gbrt2_chk['predicted'], y_gbrt2_chk['response_mv'], lowess=True, color="purple")


# In[72]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_Log,gbrt2.predict(X_train_Log))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_Log,gbrt2.predict(X_test_Log))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_Log,gbrt2.predict(X_train_Log))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_Log,gbrt2.predict(X_test_Log))
print("R-Square, testing data: ${}".format(rs_testing))


# In[73]:


from sklearn.model_selection import cross_val_score
# Create Cross Validation List to try different values
cv_list = [3,5,10]

# Define function that will return mean accuracy for different CV values
def cross_val_multiple(model, X_test_Log=X_test_Log, y_test_Log=y_test_Log, cv_list=cv_list):
  
  
#accuracy of k fold  
  for i in cv_list:
    cv_accuracy = cross_val_score(model, y_test_Log, y_test_Log, cv=i)
    print('The mean accuracy for {} cross fold validation = {:.6f}'.format(
        i,np.mean(cv_accuracy)))


# In[74]:


# Run the cross validation
cross_val_multiple(gbrt2)


# In[75]:


plot_feature_importances_boston(gbrt2)


# ## Conclusion
# 
# It appears that using a Gradient Boosting method works best in both the Lasso Regression model and the Ridge Regression model.  By limiting the depth to Max Depth = 2, the model does not over-fit, and there is a good balance of emphasis across all the features in the model.
# 
# > ### Recommended model: Model 2d - Gradient Boosting with Max Depth 2
# 
