#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.chdir("C:\\Users\\Lenovo\\Documents\\Python\\Tugas Regresi (weatherww2)")


# In[2]:


#import data
df = pd.read_csv("Summary of Weather.csv")
df.head()


# In[3]:


df=df.drop(['FT','FB','FTI','ITH','SD3','RHX','RHN','RVG','WTE'], axis=1)
df=df.drop(['WindGustSpd','PoorWeather','DR','SPD','SND','PGT','TSHDSBRSGF'], axis=1)
df= df.drop(['STA','Date','YR','MO','DA'], axis=1)


# In[4]:


df.isnull().sum()


# In[5]:


import numpy as np


# In[6]:


df['Precip']= df['Precip'].replace('T', np.nan)
df['PRCP']= df['PRCP'].replace('T', np.nan)
df['SNF']= df['SNF'].replace('T', np.nan)
df['Snowfall']= df['Snowfall'].replace('#VALUE!', np.nan)


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[9]:


df['Precip']=df['Precip'].astype('float')
df['PRCP']=df['PRCP'].astype('float')
df['SNF']=df['SNF'].astype('float')
df['Snowfall']=df['Snowfall'].astype('float')


# ## Fillna diganti dengan Median

# In[10]:


df['Precip'] = df['Precip'].fillna((df['Precip'].median()))
df['Snowfall'] = df['Snowfall'].fillna((df['Snowfall'].median()))
df['PRCP'] = df['PRCP'].fillna((df['PRCP'].median()))
df['MAX'] = df['MAX'].fillna((df['MAX'].median()))
df['MIN'] = df['MIN'].fillna((df['MIN'].median()))
df['MEA'] = df['MEA'].fillna((df['MEA'].median()))
df['SNF'] = df['SNF'].fillna((df['SNF'].median()))


# In[11]:


print(df.shape)
df.head()


# In[12]:


Precip = df['Precip'].values
MaxTemp = df['MaxTemp'].values
MinTemp = df['MinTemp'].values
MeanTemp = df['MeanTemp'].values
Snowfall = df['Snowfall'].values
PRCP = df['PRCP'].values
MAX = df['MAX'].values
MIN = df['MIN'].values
MEA = df['MEA'].values
SNF = df['SNF'].values


# In[13]:


m = len(MaxTemp)
x0 = np.ones(m)
X = np.array([x0, MaxTemp, MinTemp, MeanTemp, Snowfall, PRCP, MAX, MIN, MEA, SNF]).T
# Initial Coefficients
B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
Y = np.array(Precip)
alpha = 0.0001


# In[14]:


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J


# In[15]:


inital_cost = cost_function(X, Y, B)
print(inital_cost)


# In[16]:


def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history


# In[17]:


# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

# New Values of B
print(newB)

# Final Cost of new B
print(cost_history[-1])


# In[18]:


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)

print(rmse(Y, Y_pred))
print(r2_score(Y, Y_pred))


# ## Perbandingan dengan menggunakan Scikit Learn

# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# X and Y Values
X2 = np.array([MaxTemp, MinTemp, MeanTemp, Snowfall, PRCP, MAX, MIN, MEA, SNF ]).T
Y2 = np.array(Precip)

# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(X2, Y2)
# Y Prediction
Y_pred = reg.predict(X2)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y2, Y_pred))
r2 = reg.score(X2, Y2)

print(rmse)
print(r2)

