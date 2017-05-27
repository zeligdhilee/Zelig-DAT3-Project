# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:09:22 2017

@author: zelig
"""

import pandas as pd
import matplotlib.pyplot as plt

#data exploration 
df = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\volcanoeruption.csv')
df.head()
len(df)
df.VolcanoName.value_counts()
#Top 5 - Bezymianny (46), Etna (44), Klyuchevskoy (40), Fournaise, Piton de la (37), Ruapehu (35)
df.VolcanoName.value_counts().plot(kind='bar', title='List of Volcanos with Eruptions') #too much info!
len(df.VolcanoName.value_counts()) #325 Volcanoes with name

df.ContinuingEruption.value_counts() #1944 False, 41 True 
df.ExplosivityIndexMax.value_counts()
df.ExplosivityIndexMax.value_counts().plot(kind='bar',title='Explosivity Index of Eruptions') 

df.StartDateYear.value_counts() #Top 5 values all in the 2000s! 
df.StartDateYear.value_counts().plot(kind='barh',title ='Number of Eruptions by year, 1774-2015') #looks ugly 
df.StartDateYear.value_counts().plot(kind='line',title ='Number of Eruptions by year, 1774-2015') #looks ugly 
#line chart clearer to show by chronological order. 

cols = ['VolcanoNumber', 'VolcanoName',	 'ExplosivityIndexMax',	'StartDate',	
        'StartDateYear',	'StartDateMonth',	'StartDateDay',	'EndDate',	'EndDateYear',	
        'EndDateMonth',	'EndDateDay',	 'ContinuingEruption',	'LatitudeDecimal',	
        'LongitudeDecimal', 	'GeoLocation', 	'Activity_ID'] 


df.plot(kind='scatter', x='StartDate',y='ExplosivityIndexMax') 

df.plot(kind='scatter', x='StartDate',y='ExplosivityIndexMax') #2010-2016
plt.xlim(20100101,20161231)

df.plot(kind='scatter', x='StartDate',y='ExplosivityIndexMax') #2000-2009
plt.xlim(20000101,20091231)

df.plot(kind='scatter', x='StartDate',y='ExplosivityIndexMax') #1990-1999
plt.xlim(19900101,19991231) 
       
df.plot(kind='scatter', x='StartDate',y='ExplosivityIndexMax') #1980-1989
plt.xlim(19800101,19891231) 

#Hmm like not very meaningful here, the results even by decades 

df.plot(kind='scatter', x='VolcanoNumber',y='ExplosivityIndexMax') 
df.plot(kind='scatter',x='LatitudeDecimal',y='ExplosivityIndexMax') 
df.plot(kind='scatter',x='LongitudeDecimal',y='ExplosivityIndexMax')
df.plot(kind='scatter', x='StartDateMonth', y='ExplosivityIndexMax')
df.plot(kind='scatter',x='LongitudeDecimal',y='LatitudeDecimal')

df.describe()
#1,985 eruptions 
#Explosivity index - mean - 1.637280; std = 0.934563, min = 0, 25% = 1, 50% = 2, 75% = 2, max = 6
#StartDateMonth - mean = 6.293 - usually around June?? 

len(df) #1985 eruptions recorded

#KMeans Clustering 

#K-means clustering of volcanic eruptions- will it replicate geographical trends?
X = df.drop(['VolcanoNumber', 'VolcanoName', 'ContinuingEruption','GeoLocation', 'Activity_ID','StartDate','StartDateYear','StartDateMonth','StartDateDay','EndDate','EndDateYear','EndDateMonth','EndDateDay','Activity_ID','Duration'], axis=1) 

        
from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, random_state=1)
km.fit(X)
km.labels_ 
df['cluster'] = km.labels_
df.sort('cluster')  
km.cluster_centers_
centers = df.groupby('cluster').mean()

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
            
# create a "colors" array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow','orange','black','purple','brown'])

plt.scatter(df.LongitudeDecimal, df.LatitudeDecimal, c=colors[df.cluster], s=50)
plt.scatter(centers.LongitudeDecimal, centers.LatitudeDecimal, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('longitude')
plt.ylabel('latitude')

#not sure which value most reflects geographical reality

#Anthony's feedback look at trends by subregion or by recent timeframe. 
#Also for the last Eruption Date - replacing NaN with 0 may skew data - to relook. 

#Time Series Analysis

df['StartDate'] = pd.to_datetime(df['StartDate'],format='%Y%m%d',errors='ignore')
df.head()
df.set_index('StartDate', inplace=True)

#df['StartDateYear'] = df.index.year
#df['StartDateMonth'] = df.index.month
               
df['ExplosivityIndexMax'].autocorr(lag=1) #0.367577 
df['ExplosivityIndexMax'].autocorr(lag=30) #0.08590
df['ExplosivityIndexMax'].autocorr(lag=365) #-0.097 

df[['ExplosivityIndexMax']].resample('D').mean().expanding().mean().head()  #1.63728
  
df['ExplosivityIndexMax'].apply(['median','mean']).head() #error! 

from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(df.ExplosivityIndexMax) 
#both positive / negative values, quite alot of ups / downs 

df.head()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df.ExplosivityIndexMax, lags=200)
plt.show() #all positive. 

from statsmodels.tsa.arima_model import ARMA

#arma (1,0)
arma = df[['ExplosivityIndexMax']].astype(float)
model = ARMA(arma, (1, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=100)
plt.show 

#arma(2,0)
arma2 = df[['ExplosivityIndexMax']].astype(float)
model = ARMA(arma2, (2, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=100)
plt.show 

#ARIMA (2,0,2)

from statsmodels.tsa.arima_model import ARIMA

arima202 = df[['ExplosivityIndexMax']].astype(float)
model = ARIMA(arima202, (2, 0, 2)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=100)
plt.show 

df.ExplosivityIndexMax.diff(1).autocorr(1) #-0.46688

df.ExplosivityIndexMax.diff(1).plot()
plt.show()


#predictions of explosivity of volcanic eruptions 

model.plot_predict(1, 35) #ok for up to (1,35) (1,40 onwards returns error)
#(1,30) shows 1960 - 2003

model.plot_predict(10, 30) #1979-1995
model.plot_predict(10, 35) #1981-2003
                  
              
fig, ax = plt.subplots()
ax = df['1960'].plot(ax=ax)

fig = model.plot_predict(1, 35, ax=ax, plot_insample=False) 
#the predictive model doesn't seem to work. No error!  

#TODO - to do the code for splitting the data for training / test set 

df.head()

n = len(df.ExplosivityIndexMax)

train = df.ExplosivityIndexMax[:int(.75*n)]
test = df.ExplosivityIndexMax[int(.75*n):]

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

model = sm.tsa.ARIMA(train, (2, 0, 2)).fit()