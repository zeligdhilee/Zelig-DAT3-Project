# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:26:01 2017

@author: zelig
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#data exploration 
df = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\earthquakes.csv')
df.head()
len(df) #75849

df.date.value_counts() #correlation and no. of quakes between date and mag? The days with the more serious quakes have more counts. 
df.place.value_counts()
            
                    
cols = ['date', 'latitude', 'longitude', 'depth', 'mag', 'ampratio']

df.plot(kind='scatter', x='date',y='mag') #visualizing by date overall
#mostly between 5.5 - 7.5. A few outliers eg 1960, the 2004 Indian Tsunami, 2011 Japan tsunami
       
df.plot(kind='scatter', x='date',y='mag') #2010-2016
plt.xlim(20100101,20161231)

df.plot(kind='scatter', x='date',y='mag') #2000-2009
plt.xlim(20000101,20091231)

df.plot(kind='scatter', x='date',y='mag') #1990-1999
plt.xlim(19900101,19991231) 
       
df.plot(kind='scatter', x='date',y='mag') #1980-1989
plt.xlim(19900101,19991231) 

sns.factorplot(x='date',y='mag',data=df, kind='box')

df.plot(kind='scatter', x='date',y='ampratio')
df.plot(kind='scatter', x='date',y='depth') #not much of a trend
df.plot(kind='scatter',x='latitude',y='mag') #Three "peaks" nearer -60, -5, +30
df.plot(kind='scatter',x='longitude',y='mag')#Peaks nearer the International date line eg. where pacific plateline?
df.plot (kind='scatter',x='longitude',y='latitude')
df['mag'].plot() #magnitude over time
df['ampratio'].plot()
df['depth'].plot()

       
pd.scatter_matrix(df[['date','mag','depth']],alpha=0.2)
pd.scatter_matrix(df[['date','ampratio','depth']],alpha=0.2)
pd.scatter_matrix(df[['date','latitude','longitude']],alpha=0.2)
pd.scatter_matrix(df[['date','latitude','longitude','mag','depth']],alpha=0.2)
pd.scatter_matrix(df[['date','latitude','longitude','ampratio','depth']],alpha=0.2)

df.describe()
#mean depth of earthquake - 69.681607
#mean magnitude - 5.39. Seems that dataset is missing earthquakes below 5, as min is 5. 

#seaborn with regression line
sns.lmplot(x='date',y='mag', data=df, aspect=1.5, scatter_kws={'alpha':0.3})
sns.lmplot(x='date',y='ampratio', data=df, aspect=1.5, scatter_kws={'alpha':0.3})


#selected seaborn with regression line based on date range
sns.lmplot(x='date',y='mag', data=df, aspect=1.5, scatter_kws={'alpha':0.3})
plt.xlim(20100101,20161231)

sns.lmplot(x='date',y='mag', data=df, aspect=1.5, scatter_kws={'alpha':0.3})
plt.xlim(20000101,20091231)

sns.lmplot(x='date',y='mag', data=df, aspect=1.5, scatter_kws={'alpha':0.3})
plt.xlim(19900101,19991231)

#K-means clustering of earthquakes- will it replicate geographical trends?
X = df.drop(['id','place'], axis=1)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=1)
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
colors = np.array(['red', 'green', 'blue', 'yellow','orange','black','purple'])

plt.scatter(df.longitude, df.latitude, c=colors[df.cluster], s=50)
plt.scatter(centers.longitude, centers.latitude, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('longitude')
plt.ylabel('latitude')

#not sure which value most reflects geographical reality

#LinReg model - overall 

feature_cols = ['mag','depth']

X = df.date 
y = df[feature_cols]
X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)

print linreg.intercept_
print linreg.coef_ #-4.26336812e-07, -6.331536943-05
#hardly any correlation over time when compared to mag / depth 

#LinReg model - 2010 to 2016

feature_cols = ['mag','depth']
year2010to2016 = df[(df['date']>=20100101) & (df['date']<=20161231)]
X = year2010to2016.date 
y = year2010to2016[feature_cols]
X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)

print linreg.intercept_
print linreg.coef_ #9.66123569e-07, -3.5603160e-05

#Time series Analysis

df['date'] = pd.to_datetime(df['date'],format='%Y%m%d',errors='ignore')
df.set_index('date', inplace=True)
df.head()


df['mag'].autocorr(lag=1) #0.137528
df['depth'].autocorr(lag=1) #0.05604

df['mag'].autocorr(lag=30) #0.12211
df['depth'].autocorr(lag=30) #0.01964

df['mag'].autocorr(lag=365) #0.111375
df['depth'].autocorr(lag=365) #0.005654

df[['mag']].resample('D').mean().expanding().mean().head()  #
  
df['mag'].apply(['median','mean']).head() #error! 'list' object is not callable


from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(df.mag) #starts at 0.25 before it peaks down to zero 
autocorrelation_plot(df.depth)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df.mag, lags=100)
plt.show() #all positive, <0.2 

plot_acf(df.depth, lags=100)
plt.show() #all positive, very close to 0 
        
from statsmodels.tsa.arima_model import ARMA

#arma (1,0)
arma = df[['mag']].astype(float)
model = ARMA(arma, (1, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=50)
plt.show #For arma (1,0), data is relatively stationery between 0.1. First value is (0, 1.0)


#arma (2,0)
arma2 = df[['mag']].astype(float)
model = ARMA(arma2, (2, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=50)
plt.show

#arma (2,1)
arma2 = df[['mag']].astype(float)
model = ARMA(arma2, (2, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=50)
plt.show

#seems like arma(1,0) is sufficient given little fluctuations in autocorrelation values after 3

#TODO - to do the code for splitting the data for training / test set 
df.head()

n = len(df.mag)

train = df.mag[:int(.75*n)]
test = df.mag[int(.75*n):]


import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

model = sm.tsa.ARMA(train, (1, 0)).fit()

model.plot_predict(1,35) #The max is (1,35) 40 and above returns error. This is only for two months' worth of quakes in 1960 (Jan-Feb)
model.plot_predict(1,20) #For Jan 1960 only. 

fig, ax = plt.subplots()
ax = df['1960'].plot(ax=ax)

fig = model.plot_predict(1, 35, ax=ax, plot_insample=False) 
#whole of 1960                   
#Other years, the visualization doesn't seem to work 

