#!/usr/bin/env python
# coding: utf-8

# In[280]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score


# In[281]:


ds2train = pd.read_csv("E:/ML stuffs/train.csv")


# In[282]:


ds2train.head()


# In[283]:


ds2train.isnull()


# In[284]:


ds2train.isnull().sum()


# In[285]:


sex_count = ds2train["Sex"].value_counts()
print(sex_count)
#counting total no. of males and females


# In[286]:


ds2train.drop("Cabin", axis = 1, inplace = True)


# In[287]:


ds2train.dropna(inplace=True)


# In[288]:


ds2train.isnull().sum()


# In[289]:


ds2train.head()


# In[290]:


pd.get_dummies(ds2train["Sex"])


# In[291]:


Embark=pd.get_dummies(ds2train['Embarked'],drop_first=True)
Embark.head()


# In[292]:


sex=pd.get_dummies(ds2train['Sex'],drop_first=True)
sex.head()


# In[293]:


Pcl=pd.get_dummies(ds2train['Pclass'],drop_first=True)
Pcl.head()


# In[294]:


ds2train = pd.concat([ds2train , Embark , sex , Pcl], axis=1)


# In[295]:


ds2train.head()


# In[296]:


X = ds2train.drop(["Name" ,"Sex", "Embarked" , "Survived" , "Ticket"], axis = 1)
Y= ds2train.iloc[:,1].values
X


# In[297]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[298]:



lig = LogisticRegression() 
lig.fit(X_train, Y_train)


# In[299]:


predict = lig.predict(X_test)


# In[300]:


classification_report(Y_test , predict)


# In[ ]:





# In[301]:


cm(Y_test , predict)


# In[302]:



accuracy_score(Y_test , predict)


# In[ ]:




