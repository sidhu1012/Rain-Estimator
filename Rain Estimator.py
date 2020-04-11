#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


rain_df=pd.read_csv('C:\\Users\sudee\OneDrive\Desktop\perth-temperatures-and-rainfall\PerthTemperatures.csv')
rain_df.head()


# In[ ]:


df=rain_df[['Month','Day','Minimum temperature (Degree C)','Maximum temperature (Degree C)','Rainfall amount (millimetres)']]
df.head()


# In[ ]:


df.hist()


# In[ ]:


plt.scatter(df.Month,df[['Rainfall amount (millimetres)']])
plt.xlabel('Month')
plt.ylabel('Rainfall(mm)')
plt.show()


# In[ ]:


plt.scatter(df[['Minimum temperature (Degree C)']],df[['Rainfall amount (millimetres)']])
plt.xlabel('Minimum temperature (Degree C)')
plt.ylabel('Rainfall(mm)')
plt.show()


# In[ ]:


plt.scatter(df[['Maximum temperature (Degree C)']],df[['Rainfall amount (millimetres)']])
plt.xlabel('Maximum temperature (Degree C)')
plt.ylabel('Rainfall(mm)')
plt.show()


# In[ ]:


df=df.dropna()
df


# In[ ]:


x=df[['Month','Day','Minimum temperature (Degree C)','Maximum temperature (Degree C)']].values
y=df[['Rainfall amount (millimetres)']].values
print(x)
print(y)


# In[ ]:


from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit(x).transform(x)
x


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=150).fit(x_train,y_train)


# In[ ]:


y_hat=rfr.predict(x_test)


# In[ ]:


mse=np.mean((y_hat-y_test)**2)
print("MSE:",mse)


# In[ ]:


s=rfr.score(x,y)
print(s)

