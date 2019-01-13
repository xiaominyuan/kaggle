# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:32:27 2019

@author: yuanxiaomin
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

train_1=pd.read_csv('G:/kaggle/web-traffic-time-series-forecasting/train_1.csv')
train_1.columns
train_1.head(10)
train_1.isna().sum()
train_1.shape

train_1=train_1.fillna(0)
train_1.isna().sum()
train_1.dtypes
train_1.info()

train_1.loc[0,'Page']

def search_lang(page):
    res=re.search('[a-z][a-z].wikipedia.org',page)
    if res is not None:
        return res[0][0:2]
    return 'na'

train_1['language']=train_1['Page'].map(search_lang)
from collections import Counter

Counter(train_1['language'])

lang_set={}
lang_set['zh']=train_1[train_1['language']=='zh'].iloc[:,0:-1]
lang_set['fr']=train_1[train_1['language']=='fr'].iloc[:,0:-1]
lang_set['en']=train_1[train_1['language']=='en'].iloc[:,0:-1]
lang_set['na']=train_1[train_1['language']=='na'].iloc[:,0:-1]
lang_set['ru']=train_1[train_1['language']=='ru'].iloc[:,0:-1]
lang_set['de']=train_1[train_1['language']=='de'].iloc[:,0:-1]
lang_set['ja']=train_1[train_1['language']=='ja'].iloc[:,0:-1]
lang_set['es']=train_1[train_1['language']=='es'].iloc[:,0:-1]

sum_set={}
for key in lang_set:
    sum_set[key]=lang_set[key].iloc[:,1:].sum(axis=0)/lang_set[key].shape[0]
    
days=[]
for r in range(sum_set['en'].shape[0]):
    days.append(r)
days

plt.figure(1,figsize=[10,10])
plt.ylabel('view')
plt.xlabel('date')
lang_label={'en':'English','ja':'Japanese','de':'German','na':'Media','fr':'French','zh':'Chinese','ru':'Russian','es':'Spanish'}

for key in sum_set:
    plt.plot(days,sum_set[key],label=lang_label[key])
plt.legend()

def language_plot(key):
    plt.figure(figsize=[15,5])
    plt.xlabel('date')
    plt.ylabel('view of '+key)
    plt.plot(days,sum_set[key])
    
for key in sum_set:
    language_plot(key)


from scipy.fftpack import fft

def fft_plot(key):
    plt.figure(1,figsize=[15,5])
    plt.




