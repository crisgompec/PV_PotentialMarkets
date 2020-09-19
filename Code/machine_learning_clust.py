#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:35:44 2020

@author: peces
"""

import pandas as pd
import numpy as np
#import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default = "browser"

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/results/'

# Draw data
df = pd.read_csv(path + 'FinalResults2.csv')
df_irr_lmp = df.copy()

df_irr_lmp['size'] = df_irr_lmp['BC_ratio']**6


X = df[['Sun','std']].values 

# %% K-Means
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # la inercia es lo que hemos definido como sum(distancias ci,pj)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X) # OJO, FIT_PREDICT, NO FIT ASECAS. 
# y_means es vector asociado a cada dato de X, dando un numero que se asocia al 
# cluster al que pertenece!

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'High irradiance')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'High volatility')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Less sensitive')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'cyan', label = 'Centroids')
plt.title('Clusters of Load Zones')
plt.xlabel('Average Irradiation (W/m^2)')
plt.ylabel('Price Volatility ($/MWh)')
plt.legend()
plt.show()

# Including results in dataset 
df_irr_lmp = df.copy()
df_irr_lmp['y_kmeans'] = y_kmeans 
def colored(n):
    if n==0:
        color = 'Low inputs'
    elif n==1:
        color = 'High volatility markets'
    else:
        color = 'High irradiance areas'
    return color

df_irr_lmp['Clusters']= df_irr_lmp.apply(lambda x: colored(x['y_kmeans']),axis=1)

# %% From clasification - Final Plot

df_irr_lmp['size'] = df_irr_lmp['BC_ratio']**6

fig = px.scatter(df_irr_lmp, x="Sun", y="std", size='size',
           hover_name="State", color="Clusters", size_max=25, opacity=1)#, log_x=False, size_max=60)

fig.update_layout(
        title_text = 'B/C vs. Irradiance vs. Volatility',# <br>(Click legend to toggle traces)',#title_text = '2014 US city populations<br>(Click legend to toggle traces)', # 
        showlegend = True,
        xaxis_title="Irradiance (W/m^2)",
        yaxis_title = "Standard deviation (%)",
        shapes=[
            # filled Triangle
            dict(
                type="path",
                path=" M 140 0 L 140 25 L 245 0 Z",
                fillcolor="LightPink",
                line_color="Crimson",
                opacity=0.3,
                line_width=0,
            ),
            
            dict(
            type="path",
            path=" M 140 25 L 140 90 L 245 90 L 245 0 Z",
            fillcolor="PaleTurquoise",
            line_color="LightSeaGreen",
            opacity=0.3,
            line_width=0,
        ),
            dict(type="line", xref="x1", yref="y1",
            x0=140, y0=25, x1=245, y1=0, line_width=3, name="Logistic Regression Threshold"),
        ]
    )
# fig.add_trace(go.Scatter(x=[140,245], y=[25,0], name="Logistic Regression Threshold",
#                     line_shape='spline', line_width=0, size_max = 0.1))
            
fig.show()