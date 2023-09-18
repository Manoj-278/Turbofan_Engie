#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
np.random.seed(34)
warnings.filterwarnings('ignore')


# In[2]:


index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names


# In[3]:


df_train= pd.read_csv(r"C:\Users\LENOVO\Downloads\archive (2)\CMaps\train_FD001.txt",sep='\s',header=None,names=col_names,index_col=False)
df_test= pd.read_csv(r"C:\Users\LENOVO\Downloads\archive (2)\CMaps\test_FD001.txt",sep='\s',header=None,names=col_names,index_col=False)
df_y= pd.read_csv(r"C:\Users\LENOVO\Downloads\archive (2)\CMaps\RUL_FD001.txt",sep='\s',header=None,names=col_names,index_col=False)


# In[4]:


df_train.shape


# In[43]:


df_train.head(5)


# In[6]:


df_train.isna().sum()


# In[7]:


train= df_train.copy()
test= df_test.copy()


# In[8]:


train.head(2)


# In[9]:


train.iloc[:,0:2].describe() #.transpose()


# When we inspect the descriptive statistics of unit_nr we can see the dataset has a total of 20631 rows, unit numbers start at 1 and end at 100 as expected. What’s interesting, is that the mean and quantiles don’t align neatly with the descriptive statistics of a vector from 1–100, this can be explained due to each unit having different max time_cycles and thus a different number of rows. When inspecting the max time_cycles you can see the engine which failed the earliest did so after 128 cycles, whereas the engine which operated the longest broke down after 362 cycles. The average engine breaks between 199 and 206 cycles, however the standard deviation of 46 cycles is rather big. We’ll visualize this further down below to get an even better understanding.

# In[10]:


train.loc[:,'s_1':].describe().transpose()


# In[11]:


max_time_cycles=train.groupby('unit_number')['time_cycles'].max()
plt.figure(figsize=(20,50))
max_time_cycles.plot(kind='barh',width=0.8,align='center')
plt.title('Turbofan Engine Lifetime',fontweight='bold',size=30)
plt.xlabel('Time Cycle',fontweight='bold',size=30)
plt.xticks(size=15)
plt.ylabel('Unit',fontweight='bold',size=30)
plt.yticks(size=15)
plt.grid(True)
plt.show()


# In[12]:


sns.histplot(data=max_time_cycles,bins=20,kde=True)
plt.xlabel('max_time_cycle')


# We notice that in most of the time, the maximum time cycles that an engine can achieve is between 190 and 210 before HPC failure.
# 
# 
# 

# In[13]:


def add_rul_col(df):
    max_time_cycles= df.groupby(by="unit_number")['time_cycles'].max()
    merged= df.merge(max_time_cycles.to_frame(name='max_time_cycles'),left_on='unit_number',right_index=True)
    merged['RUL'] = merged['max_time_cycles'] - merged['time_cycles']
    merged= merged.drop("max_time_cycles",axis=1)
    return merged


# In[14]:


train=add_rul_col(train)


# In[15]:


train.head(5)


# In[16]:


train[['unit_number','RUL']]


# In[17]:


max_rul=train.groupby('unit_number').max()
max_rul.head()


# In[18]:


corr= train.corr()
plt.figure(figsize=(5,5))
sns.heatmap(corr)


# In[19]:


Sensor_dictionary={}

dict_list=[ "(Fan inlet temperature) (◦R)",
"(LPC outlet temperature) (◦R)",
"(HPC outlet temperature) (◦R)",
"(LPT outlet temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct pressure) (psia)",
"(HPC outlet pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Required fan speed)",
"(Required fan conversion speed)",
"(High-pressure turbines Cool air flow)",
"(Low-pressure turbines Cool air flow)" ]

i=1

for x in dict_list:
    Sensor_dictionary['s_'+str(i)]=x
    i+=1
Sensor_dictionary
    
    


# In[20]:


for x in sensor_names:
    plt.figure(figsize=(13,7))
    plt.boxplot(train[x])
    plt.title(x)
    plt.show()


# Observing the signal plots and the boxplots, we notice that the sensors 1,5,10,16,18,19 are constant, furthermore, we observe that the other sensors aren't well distributed and there are many outliers, then we should scale our data

# In[21]:


train.loc[:,'s_1':].describe().transpose()


# In[22]:


x= train.iloc[:,0:-1]
y= train['RUL']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)
print("x_Train data size is ",x_train.shape)
print("y_Train data size is ",y_train.shape)
print("x_Test data size is ",x_test.shape)
print("y_Test data size is ",y_test.shape)


# In[23]:


from sklearn.preprocessing import MinMaxScaler
scler= MinMaxScaler()
x_train= scler.fit_transform(x_train)
x_test= scler.fit_transform(x_test)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
    

model_logit= LinearRegression()
model_logit.fit(x_train,y_train)
pv= model_logit.predict(x_test)
print("r2 score for mode is ", r2_score(y_test,pv))
print("r2 score for mode is ", mean_squared_error(y_test,pv))


# In[42]:


l=['s_1','s_6','s_10','s_16','s_18','s_19']
df2= train.drop(l,axis=1)


x2= df2.iloc[:,0:-1]
y2= df2['RUL']
x2_train,x2_test,y2_train,y2_test= train_test_split(x,y,test_size=0.3,random_state=42)
print("x_Train data size is ",x2_train.shape)
print("y_Train data size is ",y2_train.shape)
print("x_Test data size is ",x2_test.shape)
print("y_Test data size is ",y2_test.shape)


x2_train= scler.fit_transform(x2_train)
x2_test= scler.fit_transform(x2_test)



model_logit2= LinearRegression()
model_logit2.fit(x2_train,y2_train)
pv= model_logit.predict(x2_test)
print("r2 score for mode is ", r2_score(y2_test,pv))
print("r2 score for mode is ", mean_squared_error(y2_test,pv))


# In[ ]:





# In[ ]:





# In[ ]:




