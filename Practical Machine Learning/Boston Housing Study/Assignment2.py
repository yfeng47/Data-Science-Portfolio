#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
from pandas import DataFrame, Series
import pandas_profiling
from pandas_profiling import ProfileReport
import seaborn as sns


# In[2]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True


# In[3]:


# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')


# In[4]:


# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())


# In[5]:


# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T


# In[6]:


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


# ## EDA

# In[7]:


boston.profile_report()


# In[12]:


profile = pandas_profiling.ProfileReport(boston)


# In[9]:


html_str_output = pandas_profiling.ProfileReport(boston)


# In[13]:


profile.to_file(output_file="boston_eda.html")


# In[14]:


boston_input.dropna()


# In[7]:


boston.dropna()


# In[8]:


boston_subset = boston[['mv','zn','crim','rooms','dis']]


# In[17]:



sns.distplot(boston_subset["mv"] , color="blue")


# In[18]:


f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('Linear relationships to response variable mv', size = 16, y=.9)
sns.regplot(x="rooms", y="mv", data=boston_subset,  x_jitter=.1, color = "black", ax=axes[0, 0])
sns.regplot(x="zn", y="mv", data=boston_subset, x_jitter=.1, color = "black", ax=axes[0, 1])
sns.regplot(x="dis", y="mv", data=boston_subset, x_jitter=.1, color = "black", ax=axes[1, 0])
sns.regplot(x="crim", y="mv", data=boston_subset, x_jitter=.1, color = "black", ax=axes[1, 1])
f.savefig('Linear relationships to response variable mv' '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[19]:



sns.jointplot(x="rooms", y="mv", data=boston_subset, kind="reg");
sns.jointplot(x="zn", y="mv", data=boston_subset, kind="reg");
sns.jointplot(x="dis", y="mv", data=boston_subset, kind="reg");
sns.jointplot(x="crim", y="mv", data=boston_subset, kind="reg");


# In[9]:


# Correlation matrix for the whole data set:

cm = np.corrcoef(boston.values.T)
sns.set(font_scale=1.5)
heat_map = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=['crim', 'zn', 'indus', 'chas', 'nox',' rooms', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat', 'mv'], xticklabels=['crim', 'zn', 'indus', 'chas', 'nox',' rooms', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat', 'mv'])


# In[20]:


#pairgrid
g = sns.PairGrid(boston_subset)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);


# In[21]:


sns.regplot(x="rooms", y="mv", data=boston_subset, x_jitter=.1, color = "black")


# In[16]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(boston))
print(z)


# In[17]:


threshold = 3
print(np.where(z > 3))


# In[18]:


boston_df_o = boston[(z < 3).all(axis=1)]


# In[19]:



boston_df_o.shape


# In[20]:


sns.regplot(x="rooms", y="mv", data=boston_df_o, x_jitter=.1, color = "black")


# In[21]:



boston_df_o1 = boston


# In[22]:



Q1 = boston_df_o1.quantile(0.25)
Q3 = boston_df_o1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

boston_df_out = boston_df_o1[~((boston_df_o1 < (Q1 - 1.5 * IQR)) |(boston_df_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[23]:


boston_df_out.shape


# In[24]:


sns.regplot(x="rooms", y="mv", data=boston_df_out, x_jitter=.1, color = "black")


# ## Model Deployment

# ### Test model 1 - Linear regression using only numeric variables (drop Neighborhood)
# #### test model - split data from response (without neighborhood)

# In[22]:



test_model_X = boston_input.copy()
test_model_y = boston_input.copy()

# split dataframe by data and response variables and make array for train_test_split
#remove response variable from data and save as df_data
test_model_X = test_model_X.drop(columns=['neighborhood','mv'])
#remove all but response variable and save as df_response
test_model_y = pd.DataFrame(test_model_y['mv'])

#check data df
test_model_X.head()


# In[23]:


#check response df
test_model_y.head()


# #### Test Model - train test split

# In[24]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#setting up plot learning curve to determine train test split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        


# In[25]:


#split data by training & test sets
X_train_test, X_test_test, y_train_test, y_test_test = train_test_split(    test_model_X, test_model_y,test_size=.3, random_state=10)
print(X_train_test.shape)
print(X_test_test.shape)
print(y_train_test.shape)
print(y_test_test.shape)


# In[26]:


X_train_test.describe()


# In[27]:


y_train_test.describe()


# #### test model - linear regression with validation

# In[28]:


#using sklearn LinearRegression, create test model:
from sklearn.linear_model import LinearRegression
lr_test = LinearRegression().fit(X_train_test,y_train_test)

#Cross Validation Scores: 
print("Training score: {:.2f}".format(lr_test.score(X_train_test, y_train_test)))
print("Test score: {:.2f}".format(lr_test.score(X_test_test, y_test_test)))

#plot residuals to check fit:
y_test_res = y_test_test.copy()
y_test_res['predicted']=lr_test.predict(X_test_test)
# Plot the residuals after fitting a linear model

sns.residplot(y_test_res['predicted'], y_test_res['mv'], lowess=True, color="maroon")
plt.xlabel("Predicted Home Value")
plt.ylabel("Median Home Value")
plt.title("Linear Regression: Median Home Value vs. Predicted Home Value")


# #### Analysis: Low accuracy, poor fit
# * Training Data = Mild 
# * Test Data =  Mild

# ### Dummy data and recheck model
# #### Convert Categorical neighborhood data into "Dummy" data, now will be quantitative variables

# In[29]:


boston_df2 = boston_input.copy()

#create a list of columns with yes/no values to convert to 0 and 1
list_dummies = ['neighborhood']

#use the pd 'get_dummies' method to create dummy variables for all object lists: 
for column_name in list_dummies:
    dummies = pd.get_dummies(boston_df2[column_name], prefix='value', prefix_sep='_')
    col_names_dummies = dummies.columns.values

    #then you can append new columns to the dataframe
    for i,value in enumerate(col_names_dummies):
        boston_df2[value] = dummies.iloc[:,i]

#drop categorical columns and keep dummy columns:
boston_df2 = boston_df2.drop(columns=['neighborhood'], axis = 1)
boston_df2.describe()


# In[30]:


print(boston_df2.shape)


# #### Test Model 2 - split data from response - with neighborhood

# In[31]:


test2_model_X = boston_df2.copy()
test2_model_y = boston_df2.copy()

# split dataframe by data and response variables and make array for train_test_split
#remove response variable from data and save as df_data
test2_model_X = test2_model_X.drop(columns=['mv'])
#remove all but response variable and save as df_response
test2_model_y = pd.DataFrame(test2_model_y['mv'])

#check data df
test2_model_X.head()


# #### Test model 2 = train test split and validation

# In[32]:



#split data by training & test sets
X_train_test2, X_test_test2, y_train_test2, y_test_test2 = train_test_split(    test2_model_X, test2_model_y,test_size=.3, random_state=10)
print(X_train_test2.shape)
print(X_test_test2.shape)
print(y_train_test2.shape)
print(y_test_test2.shape)


# In[33]:


#using sklearn LinearRegression, create test model:
from sklearn.linear_model import LinearRegression
lr_test2 = LinearRegression().fit(X_train_test2,y_train_test2)

#Cross Validation Scores: 
print("Training score: {:.2f}".format(lr_test2.score(X_train_test2, y_train_test2)))
print("Test score: {:.2f}".format(lr_test2.score(X_test_test2, y_test_test2)))
print("Number of features used: {}".format(np.sum(lr_test2.coef_ !=0)))

#plot residuals to check fit:
y_test2_res = y_test_test2.copy()
y_test2_res['predicted']=lr_test2.predict(X_test_test2)
# Plot the residuals after fitting a linear model
sns.residplot(y_test2_res['predicted'], y_test2_res['mv'], lowess=True, color="r")
plt.xlabel("Predicted Home Value")
plt.ylabel("Median Home Value")


# ### Analysis: The regression model is overfitting: 
# * Training Data = High Accuracy 
# * Test Data =  Low Accuracy

# # Regularization of Models

# ### Model 1  Ridge Regression - split data from response (w/ neighborhoods)

# In[34]:


R1_model_X = boston_df2.copy()
R1_model_y = boston_df2.copy()

# split dataframe by data and response variables and make array for train_test_split
#remove response variable from data and save as df_data
R1_model_X = R1_model_X.drop(columns=['mv'])
#remove all but response variable and save as df_response
R1_model_y = pd.DataFrame(R1_model_y['mv'])

#split data by training & test sets
X_train_R1, X_test_R1, y_train_R1, y_test_R1 = train_test_split(    R1_model_X, R1_model_y,test_size=.3, random_state=10)


# In[35]:


X_train_R1.head()


# In[36]:


y_train_R1.head()


# ### Model 1 - Ridge Regression 

# In[37]:



from sklearn.linear_model import Ridge
ridge1 = Ridge(alpha=.001, solver="cholesky").fit(X_train_R1,y_train_R1)
#Cross Validation Scores: 
print("Training score: {:.2f}".format(ridge1.score(X_train_R1,y_train_R1)))
print("Test score: {:.2f}".format(ridge1.score(X_test_R1,y_test_R1)))
print("Number of features used: {}".format(np.sum(ridge1.coef_ !=0)))

y_R1_res = y_test_R1.copy()
y_R1_res['predicted']=ridge1.predict(X_test_R1)
# Plot the residuals after fitting a linear model
sns.residplot(y_R1_res['predicted'], y_R1_res['mv'], lowess=True, color="cadetblue")
plt.xlabel("Predicted Home Value")
plt.ylabel("Median Home Value")


# In[38]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_R1, ridge1.predict(X_train_R1))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_R1, ridge1.predict(X_test_R1))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_R1, ridge1.predict(X_train_R1))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_R1, ridge1.predict(X_test_R1))
print("R-Square, testing data: ${}".format(rs_testing))


# #### Analysis: Moderate accuracy
# * Training Data = Moderate
# * Test Data =  Moderate

# ### Model 2 -  Lasso Regresssion Train Test split

# In[39]:



L1_model_X = boston_df2.copy()
L1_model_y = boston_df2.copy()

# split dataframe by data and response variables and make array for train_test_split
#remove response variable from data and save as df_data
L1_model_X = L1_model_X.drop(columns=['mv'])
#remove all but response variable and save as df_response
L1_model_y = pd.DataFrame(L1_model_y['mv'])

#split data by training & test sets
X_train_L1, X_test_L1, y_train_L1, y_test_L1 = train_test_split(    L1_model_X, L1_model_y,test_size=.3, random_state=10)


# In[40]:


X_train_L1.head()


# In[41]:


y_train_L1.head()


# ### Model 2 -  Lasso Regression

# In[42]:


from sklearn.linear_model import Lasso

lasso01 = Lasso(alpha=.001, max_iter = 10000).fit(X_train_L1,y_train_L1)
#Cross Validation Scores: 
print("Training score: {:.2f}".format(lasso01.score(X_train_L1,y_train_L1)))
print("Test score: {:.2f}".format(lasso01.score(X_test_L1,y_test_L1)))
print("Number of features used: {}".format(np.sum(lasso01.coef_ !=0)))

y_L1_res = y_test_L1.copy()
y_L1_res['predicted']=lasso01.predict(X_test_L1)

# Plot the residuals after fitting a lasso model
sns.residplot(y_L1_res['predicted'], y_L1_res['mv'], lowess=True, color="mediumslateblue")
plt.xlabel("Predicted Home Value")
plt.ylabel("Median Home Value")


# In[43]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_L1, lasso01.predict(X_train_L1))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_L1, lasso01.predict(X_test_L1))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_L1,lasso01.predict(X_train_L1))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_L1,lasso01.predict(X_test_L1))
print("R-Square, testing data: ${}".format(rs_testing))


# #### Analysis: Moderate accuracy with fewer variables than Model 1 and 3
# * Training Data = Moderate
# * Test Data =  Moderate

# ### Model 3 -  ElasticNet Regresssion Train Test split

# In[44]:


EL1_model_X = boston_df2.copy()
EL1_model_y = boston_df2.copy()

# split dataframe by data and response variables and make array for train_test_split
#remove response variable from data and save as df_data
EL1_model_X = EL1_model_X.drop(columns=['mv'])
#remove all but response variable and save as df_response
EL1_model_y = pd.DataFrame(EL1_model_y['mv'])

#split data by training & test sets
X_train_EL1, X_test_EL1, y_train_EL1, y_test_EL1 = train_test_split(    EL1_model_X, EL1_model_y,test_size=.3, random_state=10)


# In[45]:


X_train_EL1.head()


# In[46]:


y_train_EL1.head()


# In[47]:


from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=.001).fit(X_train_EL1,y_train_EL1)

print("Training set score: {:.2f}".format(elastic.score(X_train_EL1,y_train_EL1)))
print("Test set score: {:.2f}".format(elastic.score(X_test_EL1,y_test_EL1)))
print("Number of features used: {}".format(np.sum(elastic.coef_ !=0)))

y_EL1_res = y_test_EL1.copy()
y_EL1_res['predicted']=elastic.predict(X_test_EL1)
# Plot the residuals after fitting a linear model
sns.residplot(y_EL1_res['predicted'], y_EL1_res['mv'], lowess=True, color="mediumseagreen")
plt.xlabel("Predicted Home Value")
plt.ylabel("Median Home Value")


# In[48]:


# Calculate variance/error metrics:

import math

# Mean Squared Error
mse_train_linear = sklearn.metrics.mean_squared_error(y_train_EL1, elastic.predict(X_train_EL1))
print("Mean Squared Error, training data: ${}".format(mse_train_linear))

mse_test_linear = sklearn.metrics.mean_squared_error(y_test_EL1, elastic.predict(X_test_EL1))
print("Mean Squared Error, testing data: ${}".format(mse_test_linear))

# Root Mean Squared Error
rmse_train_linear = math.sqrt(mse_train_linear)
print("Root Mean Squared Error, training data: ${}".format(rmse_train_linear))

rmse_test_linear = math.sqrt(mse_test_linear)
print("Root Mean Squared Error, testing data: ${}".format(rmse_test_linear))

# R-Square
rs_training = sklearn.metrics.r2_score(y_train_EL1,elastic.predict(X_train_EL1))
print("R-Square, training data: ${}".format(rs_training))

rs_testing = sklearn.metrics.r2_score(y_test_EL1,elastic.predict(X_test_EL1))
print("R-Square, testing data: ${}".format(rs_testing))


# #### Analysis: Moderate accuracy with fewer variables than Model 1
# * Training Data = Moderate
# * Test Data =  Moderate

# In[52]:


from notebooktoall.transform import transform_notebook

