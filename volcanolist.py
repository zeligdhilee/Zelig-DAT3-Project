# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:31:15 2017

@author: zelig
"""
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline

volcanoeruption = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\volcanoeruption.csv')
volcanoeruption.head()
volcanoeruption.describe()

volcanolist = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\volcanolist.csv')
volcanolist.head()

volcanolist.LastKnownEruption.value_counts() #top 5 were in 2nd century AD

len(volcanolist.LastKnownEruption.value_counts()) #Eruptions occured in 353 different years 

volcanolist_eruption = pd.merge(volcanoeruption, volcanolist)
volcanolist_eruption.head()
len(volcanolist_eruption) 
#only returns 21 data points due to VolcanoNumber data in Volcanolist being inconsistent in terms of no.of digits 

cols = ['VolcanoNumber', 'VolcanoName',	 'ExplosivityIndexMax',	'StartDate',	
        'StartDateYear',	'StartDateMonth',	'StartDateDay',	'EndDate',	'EndDateYear',	
        'EndDateMonth',	'EndDateDay',	 'ContinuingEruption',	'LatitudeDecimal',	
        'LongitudeDecimal', 	'GeoLocation', 	'Activity_ID', 'Country',  'Primary Volcano Type',
        'Activity Evidence',	'LastKnownEruption',	'Region',	'Subregion',	'Latitude',	
        'Longitude',	'Elevation (m)',	'Dominant Rock Type',	'Tectonic Setting','SO2Mass','AssumedSO2Altitude']


volcanolist_eruption.plot(kind='scatter', x='Elevation (m)',y='ExplosivityIndexMax') 
#undersea volcanos tend to have zero explosvity index. 3 outliers. 
sns.lmplot(x='Elevation (m)',y='ExplosivityIndexMax', data=volcanolist_eruption, aspect=1.5, scatter_kws={'alpha':0.2})
#seaborn plot with regression line 
volcanolist_eruption.plot(kind='scatter', x='LastKnownEruption',y='ExplosivityIndexMax')
sns.lmplot(x='LastKnownEruption',y='ExplosivityIndexMax', data=volcanolist_eruption, aspect=1.5, scatter_kws={'alpha':0.2})
#seaborn plot with regression line, negative correlation after replacing "0" with "NaN" under LAstKnown Eruption, only 11 years worth 

volcanolist_eruption.Country.value_counts() #In descending - Indonesia, USA, Japan, Russia, PNG 
volcanolist_eruption.Country.value_counts().plot(kind='bar', title='Volcanoes distribution by Country')
len(volcanolist_eruption.Country.value_counts()) #48 countries 

volcanolist_eruption.Region.value_counts() 
#More meaningful to show this as opposed to countries as Volcanoes are distributed along plate boundaries
len(volcanolist_eruption.Region.value_counts()) #19 Regions 
volcanolist_eruption.Region.value_counts().plot(kind='bar', title='Volcano distribution by region')

len(volcanolist_eruption) #1985 eruptions
#There are 721 values with "NaN" - where we do not know the date of the eruption. 
#UPDATE - only 21 values left based on merged table after replacing 0 with NaN 

volcanolist_eruption.describe()

volcanolist_eruption.corr() #Correlation matrix
sns.heatmap(volcanolist_eruption.corr()) 
#As a whole there is no relation between elevation, lat / long, explosivity index with the various eruption. No trend it seems.


SO2emissions = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\SO2emissions.csv')
SO2emissions.head()
SO2emissions.describe() 
SO2emissions.SO2Mass.value_counts()
#SO2 Mass - 1.0 (33), 10.0 (28), 5.0 (25), 2.0 (21), 12.0 (14)
# Count - 395, Mean SO2 Mass emission - 146, Mean Assumed SO2 Altitude 
SO2emissions.plot(kind='scatter',x='Date Start', y='SO2Mass',title ='SO2 Emissions from Volcanic Eruptions, 1979-2015')
# SO2 emissions lastly near 0, with around 7 outliers (around 2%)
sns.lmplot(x='Date Start',y='SO2Mass', data=SO2emissions, aspect=1.5, scatter_kws={'alpha':0.2})
# Negative correlations - looks like decreasing trend of SO2 emissions over time 

#Compare SO2 emissions to eruptions     
SO2_eruption = pd.merge(SO2emissions,volcanoeruption)
SO2_eruption.head()
SO2_eruption.describe() #4688 rows
SO2_eruption.corr()
sns.heatmap(SO2_eruption.corr())
#Weak correlation between SO2 Mass and Assummed SO2 Altitude. 
#High negative correlation between longitude and elevation - means the higher volcanoes tend to be located in Western Hemisphere
#There doesn't seem to be a relationship between SO2 emissions and other geographic factors eg. elevation, latitude etc. 
#Not useful to use heatmap here! 

#LinReg - compare Elevation vs Explosivity x
X = volcanolist_eruption['Elevation (m)']
X = X.reshape(-1,1)
y = volcanolist_eruption.ExplosivityIndexMax

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)

print linreg.intercept_ #1.0394
print linreg.coef_ #0.000002053 - very weak relationship for elevation vs explosivity index

X = volcanolist_eruption.StartDate
X = X.reshape(-1,1)
y = volcanolist_eruption.ExplosivityIndexMax

linreg = LinearRegression()
linreg.fit(X,y)

print linreg.intercept_ #-44.505
print linreg.coef_ #2.290374843-06 - hardly any relationship over time - no real trend 

#LinReg probably doesn't work given the weak relationships 

#LogReg
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)

#Elevation vs Explosivity Index Max
X = volcanolist_eruption['Elevation (m)']
X = X.reshape(-1,1)
y = volcanolist_eruption.ExplosivityIndexMax
logreg.fit(X,y)
assorted_pred_class = logreg.predict(X)
assorted_pred_class
volcanolist_eruption.sort_values(by=X, inplace=True)
plt.scatter(X, y)
plt.plot(X, assorted_pred_class, color='red')
#the lines seems to be all over the place even after rearranging 
#perhaps not necessary to do LogReg after all since regression line has given some insights! 

#autocorrelation for SO2 emissions

SO2_eruption['SO2Mass'].autocorr(lag=1) #0.72147
SO2_eruption['SO2Mass'].autocorr(lag=30) #0.08537
SO2_eruption['SO2Mass'].autocorr(lag=365) #-0.001014

        
#K-Means for SO2 emissions 

SO2_eruption.head()
X = SO2_eruption.drop(['VolcanoNumber', 'VolcanoName', 'Country','Emission ID','Emission Detail ID', 'Platform','Date Start','EndDate','EndDateYear','ContinuingEruption','EndDateMonth','EndDateDay','GeoLocation','Activity_ID'], axis=1) 

from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, random_state=1)
km.fit(X)
km.labels_ 
SO2_eruption['cluster'] = km.labels_
SO2_eruption.sort('cluster')  
km.cluster_centers_
centers = SO2_eruption.groupby('cluster').mean()

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
            
# create a "colors" array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow','orange','black','purple','brown'])

plt.scatter(SO2_eruption.LongitudeDecimal, SO2_eruption.LatitudeDecimal, c=colors[SO2_eruption.cluster], s=50)
plt.scatter(centers.LongitudeDecimal, centers.LatitudeDecimal, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('longitude')
plt.ylabel('latitude')


#Time series for SO2 emissions 

SO2_eruption['Date Start'] = pd.to_datetime(SO2_eruption['Date Start'],format='%Y%m%d',errors='ignore')
SO2_eruption.head()
SO2_eruption.set_index('Date Start', inplace=True)

SO2_eruption[['SO2Mass']].resample('D').mean().expanding().mean().head()  #358

from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(SO2_eruption.ExplosivityIndexMax) 
#both positive / negative values, quite alot of ups / downs, stabilses at around 2000
autocorrelation_plot(SO2_eruption.SO2Mass)
#more stable compared to explosivity index   

SO2_eruption.head()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(SO2_eruption.SO2Mass, lags=100)
plt.show() #mostly between 0 to 0.2, reaches 0 around 60  

plot_acf(SO2_eruption.SO2Mass, lags=60)
plt.show() #mostly between 0 to 0.2, stabilises around 60  

plot_acf(SO2_eruption.SO2Mass, lags=50)
plt.show() #mostly between 0 to 0.2         
        
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error

#splitting of SO2 emissions data into training (25) + test (75) set
n = len(SO2_eruption.SO2Mass)

train = SO2_eruption.SO2Mass[:int(.75*n)]
test = SO2_eruption.SO2Mass[int(.75*n):]

#arma (1,0)
arma = SO2_eruption[['SO2Mass']].astype(float)
model = ARMA(arma, (1, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=50)
plt.show 
#one big negative outlier

#arma (2,0)
arma2 = SO2_eruption[['SO2Mass']].astype(float)
model = ARMA(arma2, (2, 0)).fit()
print model.summary()

model.resid.plot()
plot_acf(model.resid,lags=50)
plt.show 
#one big negative outlier 

#arima (2,0,2)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(SO2_eruption[['SO2Mass']], (2, 0, 2)).fit()
print model.summary() 

model.resid.plot()
plot_acf(model.resid,lags=50)
plt.show  #this time the autocorelation has no negative "outlier"

#predicting with arima(2,0,2)

model.plot_predict(1, 10) 
#don't understand why this returns error as compared to volcano explosivity and earthquake magnitude 

ig, ax = plt.subplots()
ax = SO2_eruption['2015'].plot(ax=ax)

fig = model.plot_predict(1, 50, ax=ax, plot_insample=False)

predictions = model.predict(
    '2012-01-05',
    '2016-03-30',
    dynamic=True,
)
mean_absolute_error(test, predictions)
model.summary()