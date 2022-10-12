#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#https://www.kaggle.com/code/vishalrsm/fifa-world-cup-2022-winner-prediction/notebook


# In[2]:


df = pd.read_csv(r"C:\Users\SHIVARJUN\Downloads\international_matches.csv\international_matches.csv", 
                 usecols = ['home_team', 'home_team_fifa_rank', 
                            'home_team_total_fifa_points', 'home_team_score', 'home_team_result', 'home_team_goalkeeper_score',
                           'home_team_mean_defense_score', 'home_team_mean_offense_score',
       'home_team_mean_midfield_score'])


# In[3]:



df['home_team_result'] = df['home_team_result'].replace('Draw', np.nan)
df.dropna(axis = 0, inplace = True)
df = df.reset_index(drop = True)


# In[4]:


df


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


target = df[['home_team_result']]
target[target == 'Win'] = 1.0
target[target == 'Lose'] = 0.0
#target[target == 'Draw'] = 2.0


# In[7]:


target


# In[8]:


targetdummies = pd.get_dummies(df[['home_team_result']])


# In[9]:


disc = ['home_team']
df2 = pd.get_dummies(df[['home_team']])
df = df.drop(['home_team', 'home_team_result'], axis = 1)
df = df2.join(df)
df = df


# In[10]:


df


# In[11]:


target = target.values
target = target.astype(int)
target.reshape(-1,1)


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(df.values, target, test_size = .30, random_state = 1)

#xtrain = xtrain.reset_index(drop=True)
#xtest  = xtest.reset_index(drop=True)
#ytrain = ytrain.reset_index(drop=True)
#ytest  = ytest.reset_index(drop=True)


# In[12]:


import mlclass2
# till here
from mlclass2 import simplemetrics, plot_decision_2d_lda


# In[13]:


from sklearn import preprocessing
stdscaler = preprocessing.StandardScaler().fit(xtrain)
X_scaled  = stdscaler.transform(df.values)
X_train_scaled = stdscaler.transform(xtrain)
X_test_scaled  = stdscaler.transform(xtest)


# In[18]:


from sklearn.naive_bayes import GaussianNB
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
gnb = GaussianNB()
gnb.fit(X_train_scaled, ytrain)
predicted = gnb.predict(X_test_scaled)
simplemetrics(ytest,predicted)
#plot_decision_2d_lda(gnb,X_train_scaled,ytrain,padding=-0.2,discriminant=False,title="Full Data Set",lda_on=False)
#plot_decision_2d_lda(gnb,X_train_scaled,ytrain,padding=5,discriminant=True,title="Full Data Set",lda_on=False)










# In[19]:



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, ytrain)
predicted = qda.predict(X_test_scaled)
simplemetrics(ytest,predicted)
#plot_decision_2d_lda(qda,X_train_scaled,ytrain,padding=-0.2,discriminant=False,title="Full Data Set",lda_on=True)
#plot_decision_2d_lda(qda,X_train_scaled,ytrain,padding=5,discriminant=True,title="Full Data Set",lda_on=True)


# In[ ]:





# In[20]:



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, ytrain)
predicted = lda.predict(X_test_scaled)
simplemetrics(ytest,predicted)
#plot_decision_2d_lda(lda,X_train_scaled,ytrain,padding=-0.2,discriminant=False,title="Full Data Set",lda_on=True)
#plot_decision_2d_lda(lda,X_train_scaled,ytrain,padding=5,discriminant=True,title="Full Data Set",lda_on=True)


# In[ ]:




