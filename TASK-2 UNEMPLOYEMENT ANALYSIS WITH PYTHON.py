#!/usr/bin/env python
# coding: utf-8

# In[40]:


#OASIS INFOBYTE INTERNSHIP TASK-2 (UNEMPLOYEMENT ANALYSIS WITH PYTHON)
#PRESENTED BY : AKANKSHA SINGH
#INTERN ID: OIB/A1/IP4603


# In[41]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[42]:


data1=pd.read_csv("Unemployment in India.csv")
data1


# In[43]:


data2=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
data2


# In[44]:


# Store the lengths
data1_len=len(data1)
data1_len


# In[45]:


data2_len=len(data2)
data2_len


# In[46]:


data2.head()


# In[47]:


data2.tail()


# In[48]:


data2.info()


# In[49]:


data2.shape


# In[50]:


data2.isnull().sum()


# In[51]:


data2.describe()


# In[52]:


# Region
colors=sns.color_palette('pastel')
labels=data2['Region'].dropna().unique()
plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
plt.title('Region_Percentage')
plt.pie(data2['Region'].value_counts(),labels=labels,colors=colors,autopct='%.2f%%')


# In[53]:


# Region.1
plt.figure(figsize=(30,8))
sns.countplot(x='Region.1',data=data2)
plt.show


# In[54]:


data2.corr()


# In[55]:


sns.set()
sns.heatmap(data2.corr(),annot = True)


# In[56]:


# pairplot
sns.pairplot(data2,palette="hls")


# In[57]:


#Unemployment rate according to different regions of India
data2.columns=['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate (%)','Region.1','longitude','latitude']
plt.figure(figsize=(8,6))
sns.histplot(x='Estimated Unemployment Rate',hue='Region.1',data=data2)
plt.show()


# In[58]:


import plotly.express as px


# In[39]:


Unemployment=data2[["States","Estimated Unemployment Rate","Region.1"]]
figure= px.sunburst(Unemployment,path=['States','Region.1'],
                    values='Estimated Unemployment Rate',
                    width=600,height=600,color_continuous_scale="RdY1Gn",
                    title="Indias Unemployment Rate")
figure.show()


# In[ ]:




