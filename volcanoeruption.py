# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:09:22 2017

@author: zelig
"""

import pandas as pd
import mathplotlib.pyplot as plt

#data exploration 
df = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\volcanoeruption.csv')
df.head()
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
km = KMeans(n_clusters=7, random_state=1)
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

plt.scatter(df.LongitudeDecimal, df.LatitudeDecimal, c=colors[df.cluster], s=50)
plt.scatter(centers.LongitudeDecimal, centers.LatitudeDecimal, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('longitude')
plt.ylabel('latitude')

#not sure which value most reflects geographical reality