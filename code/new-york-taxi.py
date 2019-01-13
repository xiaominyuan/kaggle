# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:48:42 2018

@author: yuanxiaomin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train=pd.read_csv('G:/kaggle/new-york-city-taxi-fare/train.csv',nrows=1000000)
test=pd.read_csv('G:/kaggle/new-york-city-taxi-fare/test.csv')

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
        delta_phi=np.radians(data[lat2]-data[lat1])
        delta_lambda=np.radians(data[lon2]-data[lon1])
        a=np.sin(delta_phi/2.0)**2+np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2.0)**2
        c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        d=(earth_r*c)
        data['distance']=d
    return d
        
haver_distance('pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')

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

plt.figure(figsize=(10,10))
plt.scatter(x=train['date'],y=train['fare_amount'],s=1.0)
plt.xlabel('date')
plt.ylabel('fare_amount')

plt.figure(figsize=(10,6))
plt.hist(train['hour'],bins=100)
plt.xlabel('hour')
plt.ylabel('frequence')

plt.figure(figsize=(10,6))
plt.scatter(x=train['hour'],y=train['fare_amount'],s=1.0)
plt.xlabel('hour')
plt.ylabel('fare_amount')

plt.figure(figsize=(10,6))
plt.hist(train['day of week'],bins=200)
plt.xlabel('day of week')
plt.ylabel('frequency')

plt.figure(figsize=(10,6))
plt.scatter(x=train['day of week'],y=train['fare_amount'])
plt.xlabel('day of week')
plt.ylabel('fare_amount')

train.columns

train.dtypes

train['distance'].describe()


bins_0=train.loc[train['distance']==0,['distance']]
bins_1=train.loc[(train['distance']>0)&(train['distance']<=10),['distance']]
bins_2=train.loc[(train['distance']>10)&(train['distance']<=50),['distance']]
bins_3=train.loc[(train['distance']>50)&(train['distance']<=100),['distance']]
bins_4=train.loc[(train['distance']>100)&(train['distance']<=150),['distance']]
bins_5=train.loc[(train['distance']>150)&(train['distance']<=200),['distance']]
bins_6=train.loc[train['distance']>200,['distance']]
bins_0['bins']='0'
bins_1['bins']='0-10'
bins_2['bins']='10-50'
bins_3['bins']='50-100'
bins_4['bins']='100-150'
bins_5['bins']='150-200'
bins_6['bins']='>200'
dist_bins=pd.concat([bins_0,bins_1,bins_2,bins_3,bins_4,bins_5,bins_6])

dist_bins.columns


plt.figure(figsize=(10,6))
plt.hist(dist_bins['bins'],bins=10)
plt.xlabel('distance_bins')
plt.ylabel('frequency')

Counter(dist_bins['bins'])

train.columns

train.loc[((train['pickup_latitude']==0)&(train['pickup_longitude']==0))&(
        (train['dropoff_latitude']!=0)&(train['dropoff_longitude']!=0))&
    (train['fare_amount']==0)]

train=train.drop(train.loc[((train['pickup_latitude']==0)&(train['pickup_longitude']==0))&(
        (train['dropoff_latitude']!=0)&(train['dropoff_longitude']!=0))&
    (train['fare_amount']==0)].index,axis=0)

train.loc[((train['pickup_latitude']!=0)&(train['pickup_longitude']!=0))&(
        (train['dropoff_latitude']==0)&(train['dropoff_longitude']==0))&
    (train['fare_amount']==0)]
train=train.drop(train.loc[((train['pickup_latitude']!=0)&(train['pickup_longitude']!=0))&(
        (train['dropoff_latitude']==0)&(train['dropoff_longitude']==0))&
    (train['fare_amount']==0)].index,axis=0)

train.shape

high_distance=train.loc[(train['distance']>200)&(train['fare_amount']!=0)]

high_distance['distance']=high_distance.apply(lambda x:(x['fare_amount']-2.50)/1.56,axis=1)

train.update(high_distance)

train[(train['distance']==0)&(train['fare_amount']==0)]

train=train.drop(train[(train['distance']==0)&(train['fare_amount']==0)].index,axis=0)

rush_hour=train.loc[(((train['hour']>=6)&(train['hour']<=20))&((train['day of week']>=1)&(train['day of week']<=5))
&(train['distance']==0)&(train['fare_amount']<2.5))]

rush_hour


train=train.drop(rush_hour.index,axis=0)

train.loc[(train['distance']!=0)&(train['fare_amount']==0)]

scene_3=train.loc[(train['distance']!=0)&(train['fare_amount']==0)]

scene_3['fare_amount']=scene_3.apply(lambda x: ((x['distance']*1.56)+2.50),axis=1)


scene_3['fare_amount']

train.update(scene_3)

train.loc[(train['distance']==0)&(train['fare_amount']!=0)]

scene4=train.loc[(train['distance']==0)&(train['fare_amount']!=0)]

scene4_sub=scene4.loc[(scene4['fare_amount']>3.0)&(scene4['distance']==0)]

scene4_sub.shape
len(scene4_sub)


scene4_sub['distance']=scene4_sub.apply(lambda x: ((x['fare_amount']-2.5)/1.56),axis=1)

train.update(scene4_sub)

train.shape

train=train.drop(['key','pickup_datetime'],axis=1)
test=test.drop(['key','pickup_datetime'],axis=1)

x_train=train.iloc[:,train.columns!='fare_amount']
y_train=train['fare_amount']
x_test=test

from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import GridSearchCV


para_test={'n_estimators':range(80,121,10)}
gridsearch1=GridSearchCV(estimator=rf(),param_grid=para_test,scoring='neg_mean_absolute_error',cv=5)
gridsearch1.fit(x_train,y_train)
gridsearch1.best_params_
#{'n_estimators': 120}

para_test2={'max_depth':range(3,14,2),'min_samples_split':range(50,201,20)}
gridsearch2=GridSearchCV(estimator=rf(n_estimators=120),param_grid=para_test2,scoring='neg_mean_absolute_error',cv=5)
gridsearch2.fit(x_train,y_train)
gridsearch2.best_params_
#{'max_depth': 13, 'min_samples_split': 50}
