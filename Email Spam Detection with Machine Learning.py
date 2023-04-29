#!/usr/bin/env python
# coding: utf-8

# In[32]:


#OASIS INFOBYTE INTERNSHIP TASK-4(EMAIL SPAM DETECTION WITH MACHINE LEARNING)
#PRESENTED BY : AKANKSHA SINGH
#INTERN ID: OIB/A1/IP4603


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("spam.csv", encoding =("ISO-8859-1"))


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.size


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True )
df


# In[11]:


df.rename(columns = {'v1' : 'Target', 'v2':'Message'}, inplace = True)
df


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df.drop_duplicates(keep = 'first', inplace = True)


# In[15]:


df.duplicated().sum()


# In[16]:


df.size


# In[17]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
df['Target']


# In[18]:


df.head()


# In[19]:


plt.pie(df['Target'].value_counts(), labels = ['ham', 'spam'], autopct = "%0.2f")
plt.show()


# In[20]:


x = df['Message']
y = df['Target']


# In[21]:


print(x)


# In[22]:


print(y)


# In[23]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 3)


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# In[25]:


cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)


# In[26]:


print(x_train_cv)


# In[27]:


from sklearn.linear_model import LogisticRegression 
lr_model = LogisticRegression()


# In[28]:


lr_model.fit(x_train_cv, y_train)
Predection_train = lr_model.predict(x_train_cv)


# In[29]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,Predection_train)*100)


# In[30]:


Predection_test = lr_model.predict(x_test_cv)


# In[31]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,Predection_test)*100)


# In[ ]:




