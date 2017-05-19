# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:26:01 2017

@author: zelig
"""

import pandas as pd
import mathplotlib.pyplot as plt

#data exploration 
df = pd.read_csv(r'C:\Users\zelig\Documents\GA Data Science DAT3\Project\dataset\earthquakes.csv')
df.head()

cols = ['date', 'latitude', 'longitude', 'depth', 'mag']

df.plot(kind='scatter', x='date',y='mag') #mostly between 5.5 - 7.5. A few outliers eg 1960, the 2004 Indian Tsunami, 2011 Japan tsunami
df.plot(kind='scatter', x='date',y='depth') #not much of a trend
df.plot(kind='scatter',x='latitude',y='mag') #Three "peaks" nearer -60, -5, +30
df.plot(kind='scatter',x='longitude',y='mag')#Peaks nearer the International date line eg. where pacific plateline?
df.plot (kind='scatter',x='longitude',y='latitude')
       
pd.scatter_matrix(df[['date','mag','depth']])
pd.scatter_matrix(df[['date','latitude','longitude']])

df.describe()
#mean depth of earthquake - 69.681607
#mean magnitude - 5.39. Seems that dataset is missing earthquakes below 5, as min is 5. 

len(df) #75849

#K-means clustering of earthquakes- will it replicate geographical trends?
X = df.drop(['id','place'], axis=1)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, random_state=1)
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