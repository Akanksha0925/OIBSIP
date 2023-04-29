#!/usr/bin/env python
# coding: utf-8

# In[43]:


#OASIS INFOBYTE INTERNSHIP TASK-3 (CAR PRICE PREDICTION)
#PRESENTED BY : AKANKSHA SINGH
#INTERN ID: OIB/A1/IP4603


# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[45]:


#loading and displayiong of data set
data_car = pd.read_csv("CarPrice_Assignment.csv")


# In[46]:


data_car


# In[47]:


data_car.head(20)


# In[48]:


data_car.tail(20)


# In[49]:


data_car.info()


# In[50]:


data_car.describe()


# In[51]:


data_car.size


# In[52]:


data_car.shape


# In[53]:


data_car.isnull().sum


# In[54]:


data_car.duplicated().sum()


# In[55]:


data_car.columns


# In[56]:


print(data_car.fueltype.value_counts())
print(data_car.aspiration.value_counts())
print(data_car.doornumber.value_counts())
print(data_car.carbody.value_counts())
print(data_car.drivewheel.value_counts())
print(data_car.enginelocation.value_counts())
print(data_car.fuelsystem.value_counts())


# In[57]:


#relationship between Sybolling and price
pt.figure(figsize=(20,8))
pt.subplot(1,2,2)
pt.title('Symboling vs Price')
sb.boxplot(x=data_car.symboling, y=data_car.price, palette=("cubehelix"))
pt.show()


# In[58]:


#relationship between Fuelsystem and price
pt.subplots(figsize=(40,30))
a=sb.boxplot(x='fuelsystem',y='price',data=data_car)
a.set_xticklabels(a.get_xticklabels(),rotation=90,ha='right')
pt.show()


# In[59]:


#relationship between Engine and price
pt.subplots(figsize=(40,30))
a=sb.boxplot(x='enginetype',y='price',data=data_car)
a.set_xticklabels(a.get_xticklabels(),rotation=90,ha='right')
pt.show()


# In[ ]:


pt.figure(figsize=(20,8))
pt.subplot(1,2,1)
pt.title('Car Price Distribution Plot')
sb.distplot(data_car.price)
pt.show()


# In[23]:


pt.figure(figsize=(20,8))
pt.subplot(1,2,2)
pt.title('Car Price Spread')
sb.boxplot(y=data_car.price)


# In[24]:


#heatmap
pt.figure(figsize=(20,8))
corrrr=data_car.corr()
sb.heatmap(corrrr, xticklabels=corrrr.columns, yticklabels=corrrr.columns, annot=True)


# In[25]:


sb.pairplot(data_car)


# In[26]:


data_car = pd.get_dummies(data_car, columns = ['fueltype','aspiration','doornumber','carbody','drivewheel','cylindernumber','enginelocation','enginetype', 'fuelsystem'])
print(data_car)


# In[27]:


x_ = data_car.drop(['CarName','price'],axis=1)
y_ = data_car[['price']]


# In[28]:


x_


# In[29]:


y_


# In[30]:


x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.25,random_state=50)


# In[31]:


x_train


# In[32]:


y_train


# In[33]:


model1 = LinearRegression()
model1.fit(x_train,y_train)


# In[34]:


y_predt= model1.predict(x_test)


# In[35]:


Mean_absolute_error = mean_absolute_error(y_test,y_predt)
print('Mean_absolute_error:',Mean_absolute_error)


# In[36]:


Mean_squared_error = mean_squared_error(y_test,y_predt)
print('Mean_squared_error:',Mean_squared_error)


# In[37]:


Root_Mean_squared_error = np.sqrt(Mean_squared_error)
print('Root_Mean_squared_error:',Root_Mean_squared_error)


# In[38]:


R_Squared = r2_score(y_test,y_predt)
print('R-Squared:',R_Squared)


# In[40]:


y_predt


# In[41]:


y_test


# In[42]:


y_test - y_predt


# In[ ]:




