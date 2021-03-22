#!/usr/bin/env python
# coding: utf-8

# In[323]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from matplotlib import pyplot as plt

#With the CSV file saved locally, I can get started inspecting the Historical Hourly Weather dataset from Kaggle (https://www.kaggle.com/selfishgene/historical-hourly-weather-data)
temperature_df = pd.read_csv('temperature.csv')

print(temperature_df.head(5))
print(temperature_df.tail(5))


# ### Dataset Overview
# The inspection of the dataset reveals the following insights
# 1. The extent of this data set begins on October 1st, 2012 and ends on on November 29th, 2017.
# 2. Not all cities have sufficient data which begins and ends on these dates.
# 3. Data has been taken at 1-hour intervals, 24 times per day.
# 4. Temperatures are in Kelvin: the world's most useless unit of measurement.

# In[324]:


df = temperature_df[['datetime','New York','Boston','Seattle','Los Angeles']]
#In addition, I removed all empty cells from this incomplete dataset.
df.dropna(how='any', inplace=True)
#There are 24 separate temperature readings every day. For simplicity, I took recorded temperatures down to one reading per day by including only one out of every 24 rows.
df = df.iloc[::24]

print(df.head(5))
print(df.tail(5))


# In[325]:


# Here is the line of code that allow me to examine the data types of my df dataset.
print(df.info())


# Notice: Pandas notoriously stores data types from CSVs as objects when it doesn't know what's up. "Object" is a fancy Pandas word for "uselessly broad classification of data type." Pandas sees the special characters in this column's data, thus immediately surrenders any attempt to logically parse said data. 

# In[326]:


#The following code is used to convert datetime variable into date format.
df['date'] = pd.to_datetime(df['datetime'])


# In[327]:


#The following code is used to create "label" which will group all values of the same year together. 
df['year'] = df['date'].dt.year


# In[328]:


#The following code is used to create another "label" which will group all values of the same day of one year together.
df['day'] = df['date'].dt.dayofyear


# In[329]:


#The formula below is used for converting New York temperature from Kelvin to Fahrenheit 
df['New York'] = df['New York'].apply(lambda x: (x-273.15) * 9/5 + 32)


# In[330]:


#The formula below is used for converting Boston temperature from Kelvin to Fahrenheit 
df['Boston'] = df['Boston'].apply(lambda x: (x-273.15) * 9/5 + 32)


# In[331]:


#The formula below is used for converting Seattle temperature from Kelvin to Fahrenheit 
df['Seattle'] = df['Seattle'].apply(lambda x: (x-273.15) * 9/5 + 32)


# In[332]:


#The formula below is used for converting Los Angeles temperature from Kelvin to Fahrenheit 
df['Los Angeles'] = df['Los Angeles'].apply(lambda x: (x-273.15) * 9/5 + 32)


# In[361]:


# The following line of code is used to extract data for 2015 by selecting monthly records of 2015 (the median value of year)
p = df[df['year']== df['year'].median()].squeeze()
p = p.iloc[::31]
p


# In[369]:


# The following codes are used for creating a multiple-line chart of four major U.S. cities'temperature values in 2015
plt.plot( 'day', 'New York', data=p, marker='', color='blue', linewidth=4)
plt.plot( 'day', 'Boston', data=p, marker='', color='purple', linewidth=4)
plt.plot( 'day', 'Seattle', data=p, marker='', color='orange', linewidth=4)
plt.plot( 'day', 'Los Angeles', data=p, marker='', color='red', linewidth=4)
plt.legend()
plt.title('U.S. Cities Weather in 2015')
plt.xlabel('Days')
plt.ylabel('Temperature')


# In[370]:


#The following codes are used for extracting only numeric temperature values of target cities before making an index chart
df_new = df[['New York','Boston','Seattle','Los Angeles']]
print(df_new.head(5))
print(df_new.tail(5))


# In[371]:


#The following codes are used for adjusting temperature values of target cities so that they are equal to each other in a given starting time period.
# All temperature values are set to equal to 100 on October 1st, 2012 (1st day in the dataset)
x = df_new[df_new.index== df_new.index.min()].squeeze()
df_ct = 100 + ((df_new - x) / x) * 100
df_ct


# In[372]:


# The following codes are used for adding corresponding day and year values to the indexed dataset
df_ct['year'] = df['year']
df_ct['day'] = df['day']


# In[373]:


# Here is the overview of the indexed dataset with year and day values
df_ct


# In[375]:


# The following codes are used for extracting 2015 monthly temperature data from the indexed dataset
z = df_ct[df_ct['year']== df_ct['year'].median()].squeeze()
z = z.iloc[::31]
z


# In[377]:


# The following codes are used for creating a multiple-line index chart of four major U.S. cities'temperature values in 2015
plt.plot( 'day', 'New York', data=z, marker='', color='blue', linewidth=4)
plt.plot( 'day', 'Boston', data=z, marker='', color='purple', linewidth=4)
plt.plot( 'day', 'Seattle', data=z, marker='', color='orange', linewidth=4)
plt.plot( 'day', 'Los Angeles', data=z, marker='', color='red', linewidth=4)
plt.legend()
plt.title('U.S. Cities Temperature Growth in 2015')
plt.xlabel('Days')
plt.ylabel('Index')

