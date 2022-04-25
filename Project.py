#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import os, os.path
import seaborn as sns


# In[46]:


train_path = os.path.abspath(os.getcwd())+"/train.csv"
train_df = pd.read_csv(train_path)
scores_path = os.path.abspath(os.getcwd())+"/train_target_and_scores.csv"
scores_df = pd.read_csv(scores_path)
test_path = os.path.abspath(os.getcwd())+"/test.csv"
test_df = pd.read_csv(test_path)


# In[48]:


train_df.head()


# In[49]:


test_df.head()


# In[50]:


sns.countplot(x='is_cup', data=train_df)


# In[51]:


sns.countplot(x='target', data=scores_df)


# In[52]:


train_df.set_index(keys='id', inplace=True)
test_df.set_index(keys='id', inplace=True)


# In[ ]:


# do preprocessing together so we combine train and test first.


# realizing there is an extra column 'target' has to be dropped because the test dataset does not have it

# In[53]:


train_df.drop(['target'], axis=1, inplace=True)


# In[63]:


train_n = train_df.shape[0]
print(train_df.shape,test_df.shape)


# In[57]:


all_df = pd.concat((train_df, test_df))
all_df


# In[58]:


# convert is_cup to a binary variable
all_df['is_cup'] = all_df['is_cup'].map({False: 0, True: 1})
# convert target results to 0,1,2
scores_df = scores_df['target'].map({'home': 0, 'draw': 1, 'away': 2})


# In[ ]:


# remove all the coach and dates, team names and league name as they are not useful in training


# In[60]:


all_df.drop(['home_team_name', 'away_team_name', 'league_name'], axis=1, inplace=True)
all_df.drop(all_df.filter(regex='date').columns, axis=1, inplace = True)
all_df.drop(all_df.filter(regex='coach').columns, axis=1, inplace = True)


# In[59]:


scores_df


# In[62]:


# reseparate train and test


# In[71]:


X_train = all_df[:train_n].to_numpy()


# In[72]:


X_test = all_df[train_n:].to_numpy()


# In[74]:


y_train = scores_df.to_numpy()


# In[75]:





# In[ ]:




