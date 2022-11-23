#!/usr/bin/env python
# coding: utf-8

# In[32]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as MSE
from mpl_toolkits.mplot3d import Axes3D


# In[6]:


data= pd.read_csv('HR_comma_sep.csv')


# In[11]:


data


# In[10]:


data['Department'].value_counts()


# In[13]:


data['left'].value_counts()


# In[16]:


data['salary'].value_counts()


# In[17]:


## Feature Selection 
#Using Pearson Correlation
corr = data.corr()


# In[18]:


corr


# In[21]:


#plt.figure(figsize=(12,10))
#sns.heatmap(corr,annot=True)
#plt.show()


# In[ ]:


#plt.figure(figsize=(12,10))
#sns.scatterplot(data=df, x="last_evaluation", y="satisfaction_level",z="average_mountly_hours", hue='left')
#plt.show()


# In[31]:


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = data['last_evaluation']
y = data['satisfaction_level']
z = data['average_montly_hours']

ax.set_xlabel("Happiness")
ax.set_ylabel("Economy")
ax.set_zlabel("Health")

ax.scatter(x, y, z)

plt.show()


# In[ ]:




