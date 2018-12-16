# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:48:42 2018

@author: yuanxiaomin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train=pd.read_csv('G:/kaggle/new york city taxi fare prediction/train.csv',nrows=2000000)
test=pd.read_csv('G:/kaggle/new york city taxi fare prediction/test.csv')

train.shape
test.shape

train.head(1)

train.columns

train.dtypes

train.describe

train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
train.isnull()
train=train.drop(train[train.isnull().any(1)].index,axis=0)

train.shape

train['fare_amount'].describe()

from collections import Counter
Counter(train['fare_amount']<0)

train=train.drop(train[train['fare_amount']<0].index,axis=0)

train.shape

train.describe

train[train['passenger_count']>6]

train=train.drop(train[train['passenger_count']==208].index,axis=0)

train['passenger_count'].describe()

train['pickup_latitude'].describe()

Counter(train[train['passenger_count']==0].fare_amount>0)

Counter((train['pickup_latitude']<-90)|(train['pickup_latitude']>90))

train=train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index,axis=0)

train=train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index,axis=0)


train=train.drop(((train[train['dropoff_longitude']<-180])|(train[train['dropoff_longitude']>180])).index,axis=0)
train=train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index,axis=0)


train.shape
train.dtypes

train['key']=pd.to_datetime(train['key'])
train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])

test['key']=pd.to_datetime(test['key'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])

def haver_distance(lat1,lon1,lat2,lon2):
    data_set=[train,test]
    for data in data_set:
        earth_r=6371
        phi1=np.radians(data[lat1])
        phi2=np.radians(data[lat2])
        delta_phi=np.radians(data[lat1]-data[lat2])
        delta_lambda=np.radians(data[lon1]-data[lon2])
        a=np.sin(delta_phi/2.0)**2+np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2.0)**2
        c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        d=(earth_r*c)
        data['distance']=d
    return d
        
haver_distance('pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude')

test.distance

def date_process(data_list):
    for data in data_list:
        data['year']=data['pickup_datetime'].dt.year
        data['month']=data['pickup_datetime'].dt.month
        data['date']=data['pickup_datetime'].dt.day
        data['day of week']=data['pickup_datetime'].dt.dayofweek
        data['hour']=data['pickup_datetime'].dt.hour

data_list=[train,test]
date_process(data_list)

train.columns

train['passenger_count'].max()
train['passenger_count'].min()
plt.figure(figsize=(15,10))
plt.hist(train['passenger_count'],bins=10)
plt.xlabel('passenger_count')
plt.ylabel('frequncy')

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'],y=train['fare_amount'],s=1.0)
plt.xlabel('passenger_count')
plt.ylabel('fare_amount')


