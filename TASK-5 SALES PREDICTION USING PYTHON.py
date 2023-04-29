#!/usr/bin/env python
# coding: utf-8

# In[7]:


#OASIS INFOBYTE INTERNSHIP TASK-5 (SALES PREDICTION USING PYTHON)
#PRESENTED BY : AKANKSHA SINGH
#INTERN ID: OIB/A1/IP4603


# In[8]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[9]:


df=pd.read_csv('Advertising.csv')


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


df.columns.values.tolist()


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


fig,axs=plt.subplots(3,figsize=(5,5))
plt1=sns.boxplot(df['TV'],ax=axs[0])
plt2=sns.boxplot(df['Newspaper'],ax=axs[1])
plt3=sns.boxplot(df['Radio'],ax=axs[2])
plt.tight_layout()
                    


# In[18]:


sns.distplot(df['Newspaper'])


# In[19]:


iqr=df.Newspaper.quantile(0.75)-df.Newspaper.quantile(0.25)


# In[20]:


lower_bridge=df["Newspaper"].quantile(0.25)-(iqr*1.5)
upper_bridge=df["Newspaper"].quantile(0.75)-(iqr*1.5)
print(lower_bridge)
print(upper_bridge)


# In[21]:


data=df.copy()


# In[22]:


data.loc[data['Newspaper']>=93,"Newspaper"]=93


# In[23]:


sns.boxplot(data["Newspaper"])


# In[24]:


sns.pairplot(data,x_vars=['TV','Newspaper','Radio'],y_vars='Sales',height=4,aspect=1,kind='scatter')
plt.show()


# In[25]:


sns.heatmap(data.corr(),cmap="YlGnBu",annot=True)
plt.show()


# In[26]:


import_features=list(df.corr()['Sales'][(df.corr()['Sales']>+0.5)|(df.corr()['Sales']<-0.5)].index)


# In[27]:


print(import_features)


# In[28]:


x=data['TV']
y=data['Sales']


# In[29]:


x=x.values.reshape(-1,1)


# In[30]:


x


# In[31]:


y


# In[32]:


print(x.shape,y.shape)


# In[33]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# In[34]:


print(x_train.shape,y_train.shape)


# In[35]:


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[36]:


knn=KNeighborsRegressor().fit(x_train,y_train)
knn


# In[37]:


knn_train_pred=knn.predict(x_train)


# In[38]:


knn_test_pred=knn.predict(x_test)


# In[39]:


print(knn_train_pred,knn_test_pred)


# In[40]:


Results=pd.DataFrame(columns=["Model","Train R2","Test R2","Test RMSE","Variance"])


# In[41]:


r2=r2_score(y_test,knn_test_pred)
r2_train=r2_score(y_train,knn_train_pred)
rmse=np.sqrt(mean_squared_error(y_test,knn_test_pred))
variance=r2_train-r2
Results=Results.append({"Model":"K-Nearest Neighbours","Train R2":r2_train,"Test R2":r2,"Test RMSE":rmse,"Variance":variance},ignore_index=True)
print("R2:",r2)
print("RMSE:",rmse)


# In[42]:


Results.head()


# In[43]:


svr=SVR().fit(x_train,y_train)
svr


# In[44]:


svr_train_pred=svr.predict(x_train)
svr_test_pred=svr.predict(x_test)


# In[45]:


print(svr_train_pred,svr_test_pred)


# In[46]:


r2=r2_score(y_test,svr_test_pred)
r2_train=r2_score(y_train,svr_train_pred)
rmse=np.sqrt(mean_squared_error(y_test,svr_test_pred))
variance=r2_train-r2
Results=Results.append({"Model":"Support Vector Machine","Train R2":r2_train,"Test R2":r2,"Test RMSE":rmse,"Variance":variance},ignore_index=True)
print("R2:",r2)
print("RMSE:",rmse)


# In[47]:


Results.head()


# In[48]:


import statsmodels.api as sm


# In[49]:


x_train_constant=sm.add_constant(x_train)


# In[50]:


model=sm.OLS(y_train,x_train_constant).fit()


# In[51]:


model.params


# In[52]:


print(model.summary())


# In[53]:


plt.scatter(x_train,y_train)
plt.plot(x_train,6.9955+0.0541*x_train,'y')
plt.show()


# In[54]:


y_train_pred=model.predict(x_train_constant)
res=(y_train-y_train_pred)
res


# In[55]:


fig=plt.figure()
sns.distplot(res,bins=15)
fig.suptitle('Error Terms',fontsize=15)
plt.xlabel('Difference in y_train and y_train_pred',fontsize=15)
plt.show()


# In[56]:


plt.scatter(x_train,res)
plt.show()


# In[57]:


x_test_constant=sm.add_constant(x_test)
y_pred=model.predict(x_test_constant)


# In[58]:


y_pred


# In[59]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[60]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[61]:


r2=r2_score(y_test,y_pred)
r2


# In[63]:


plt.scatter(x_test,y_test)
plt.plot(x_test,6.9966+0.0541*x_test,'y')
plt.show()


# In[ ]:




