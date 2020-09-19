#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:30:00 2020

@author: peces
"""

import pandas as pd
import numpy as np
import os
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/'
FILE_NAME = 'project-sunroof-state.csv'
SIZE_PANEL = 1.6368 #m^2/panel
KWH_TO_MWH = 1/1000
SIZE_NORM = 10000
EFFICIENCY = 0.2
N_HOURS = 8664.0
HOURS_YEAR = 8770
FACTOR_HOURS = N_HOURS/HOURS_YEAR
DC_AC_DERATE_FACTOR = 0.85


df = pd.read_csv(path + FILE_NAME)

df['Normalized'] = df['yearly_sunlight_kwh_f']/(SIZE_PANEL * df['number_of_panels_f']) * KWH_TO_MWH * SIZE_NORM / (df['percent_qualified']/100 )#* EFFICIENCY * df['percent_qualified']/100 
df_compare = df[['Normalized','region_name']]
df_generation = pd.read_csv(path+'results/25YearsGeneration.csv')

node_names = list(df_generation.columns)
dt_empty = pd.DataFrame(columns = node_names)

for node in node_names:
    for i in list(df_compare.index):
        if df_compare.loc[i,"region_name"].upper()==node:
            dt_empty.loc[0,node]=df_compare.loc[i,"Normalized"]
            print(df_compare.loc[i,"Normalized"])
            
#dt_empty.to_csv('/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/results/SunRoofNormalized.csv', index=False)

# %% 
dt_reloaded = pd.read_csv(path+'results/SunRoofNormalized2.csv')
for node in node_names:
    dt_reloaded.loc[1,node] = df_generation[node].mean()
    dt_reloaded.loc[2,node] = dt_reloaded.loc[1,node]/dt_reloaded.loc[0,node]*DC_AC_DERATE_FACTOR
    
    
# %% Objective: make a comparative map of mine and sunroofs project
list_to_drop = ['TX_AEN', 'TX_CPS', 'TX_HOUSTON', 'TX_LCRA', 'TX_NORTH',
       'TX_RAYBN', 'TX_SOUTH', 'TX_WEST', 'NY_CAPITL', 'NY_DUNWOD', 'NY_GENESE', 'NY_HQ',
       'NY_HUDSON_VALLEY', 'NY_MHKVL', 'NY_MILLWD', 'N.Y.C.',
       'NY_NORTH', 'NY_NPX', 'NY_OH', 'NY_PJM', 'NY_WEST',]


dt_trans = (dt_reloaded.drop(columns = list_to_drop).transpose()).copy()
dt_trans['Index'] = dt_trans.index
#dt_trans['Index'] = dt_trans.index
dt_trans[2] = dt_trans[2]-1
dt_trans['Increment'] = dt_trans[2] * 100
dt_trans['Increment']  = dt_trans['Increment'] #- dt_trans['Increment'].mean()
print(dt_trans['Increment'].mean())

fig = px.bar(dt_trans.sort_values(by=['Increment']), x="Index", y="Increment") #, color='time')
fig.update_layout(
        title_text = 'Porcentual Difference in Generation Estimation <br>Mean difference = '  + str(dt_trans['Increment'].mean()) + '%',#title_text = '2014 US city populations<br>(Click legend to toggle traces)', # 
        showlegend = True,
        xaxis_title="Hubs",
        yaxis_title = "Porcentual Difference (%)"
    )
fig.show()

# %%


#limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
limits = [(0,15),(15.01,25),(25.01,35),(35.01,45),(45.01,50)]
#colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
colors = ['#b4651f', '#b4911f', '#a0b41f', '#5bb41f', '#1fb48c' , '#871fb4']
cities = []
scale = 0.01
#sizeref = 2. * max(df['BC_ratio']) / (desired maximum marker size ** 2)

fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    #df_sub = df[lim[0]:lim[1]]
    df_sub = df[(df['BC_ratio']>lim[0]) & (df['BC_ratio']<lim[1])]
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_sub['lon'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = (df_sub['BC_ratio'])**8/scale,#size = df_sub['pop']/scale, # 
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.01,
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = 'Estimated B/C ratio',# <br>(Click legend to toggle traces)',#title_text = '2014 US city populations<br>(Click legend to toggle traces)', # 
        showlegend = True,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()





