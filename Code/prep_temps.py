#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:12:52 2020

@author: peces
"""

import pandas as pd
import numpy as np
import os
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

# First definitions
path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/temps/'
def get_nhour(month, day, hour):
    return (month-1)*30.416667*24 + (day-1)*24 + hour 
 
geographical_areas = os.listdir(path)
geographical_areas.remove('.DS_Store')
data_to_store = pd.DataFrame()

dataset = pd.read_csv(path + geographical_areas[3]) # 3 for example
dt_hourly = dataset[dataset['Minute']<15]
dt_hourly['N_hour'] = dt_hourly.apply(lambda x: get_nhour(x['Month'],x['Day'],x['Hour']),axis=1)
data_to_store['N_hour'] = dt_hourly['N_hour']

# Divide areas from stage 1 and stage 2
geographical_areas_1 = []
geographical_areas_2 = []
for area in geographical_areas:
    if area[0].isupper():
        geographical_areas_2.append(area)
    else:
       geographical_areas_1.append(area)     

# For loop to get all the location data available - NY
for area in geographical_areas_1:
    dataset = pd.read_csv(path + area)
    dt_hourly = dataset[dataset['Minute']<15]
    data_to_store[area] = dt_hourly['GHI']
    
# For loop to get all the location data available - REST
for area in geographical_areas_2:
    dataset = pd.read_csv(path + area, header = 2)
    dt_hourly = dataset[dataset['Minute']<15]
    data_to_store[area] = dt_hourly['GHI']

# Plot GHI 
fig = go.Figure()

for area in geographical_areas:
    fig.add_trace(go.Scatter(
        x=data_to_store['N_hour'],#dt_aux['Time Stamp'],#x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        y=data_to_store[area],
        name=area))#,


# Add figure title and legend
fig.update_layout(title_text="Irradiation 2018 in capital",showlegend=True)
# Set x-axis title
fig.update_xaxes(title_text="Hour")
# Set y-axis title
fig.update_yaxes(title_text="GHI (W/m^2)") #, range=[-250.0,1100.0])

fig.show()

for node in list(data_to_store.columns):
    print(data_to_store[node].describe())

# Save data extracted
data_to_store.to_csv('/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/filtered/irradiation.csv', index=False)



