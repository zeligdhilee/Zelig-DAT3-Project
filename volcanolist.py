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

volcanolist = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\volcanolist.csv')
volcanolist.head()

volcanolist_eruption = pd.merge(volcanoeruption, volcanolist)
volcanolist_eruption.head()

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
#seaborn plot with regression line, very small positive correlation 

volcanolist_eruption.Country.value_counts() #In descending - Indonesia, USA, Japan, Russia, PNG 
volcanolist_eruption.Country.value_counts().plot(kind='bar', title='Volcanoes distribution by Country')
len(volcanolist_eruption.Country.value_counts()) #48 countries 

volcanolist_eruption.Region.value_counts() 
#More meaningful to show this as opposed to countries as Volcanoes are distributed along plate boundaries
len(volcanolist_eruption.Region.value_counts()) #19 Regions 
volcanolist_eruption.Region.value_counts().plot(kind='bar', title='Volcano distribution by region')

len(volcanolist_eruption) #1985 eruptions
#There are 721 values with "0" - where we do not know the date of the eruption. 

volcanolist_eruption.describe()

volcanolist_eruption.corr() #Correlation matrix
sns.heatmap(volcanolist_eruption.corr()) 
#As a whole there is no relation between elevation, lat / long, explosivity index with the various eruption. No trend it seems.


SO2emissions = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\SO2emissions.csv')
SO2emissions.head()
SO2emissions.describe() 
# Count - 395, Mean SO2 Mass emission - 146, Mean Assumed SO2 Altitude 
SO2emissions.plot(kind='scatter',x='Date Start', y='SO2Mass',title ='SO2 Emissions from Volcanic Eruptions, 1979-2015')
# SO2 emissions lastly near 0, with around 7 outliers (around 2%)
sns.lmplot(x='Date Start',y='SO2Mass', data=SO2emissions, aspect=1.5, scatter_kws={'alpha':0.2})
# Negative correlations - looks like decreasing trend of SO2 emissions over time 

#Compare SO2 emissions to eruptions     
SO2_eruption = pd.merge(SO2emissions,volcanolist)
SO2_eruption.head()
SO2_eruption.describe()
SO2_eruption.corr()
sns.heatmap(SO2_eruption.corr())
#Weak correlation between SO2 Mass and Assummed SO2 Altitude. 
#High negative correlation between longitude and elevation - means the higher volcanoes tend to be located in Western Hemisphere
#There doesn't seem to be a relationship between SO2 emissions and other geographic factors eg. elevation, latitude etc. 

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




