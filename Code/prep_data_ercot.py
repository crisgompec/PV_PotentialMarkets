#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:13:49 2020

@author: peces
"""

import pandas as pd
import numpy as np
import os
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
pio.renderers.default = "browser"

path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/'
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


dataset = pd.read_excel(path+'ercot2018.xlsx')
HUBS = pd.unique(dataset['Settlement Point'])[6:] # We are interested in the loadzones
LMP_def = np.zeros([8,1])

for month in tqdm(MONTHS):
    dataset = pd.read_excel(path+'ercot2018.xlsx', sheet_name = month)
    lmp_aux = []
    
    for hub_name in HUBS:
        
        df_node = dataset[dataset['Settlement Point']==hub_name] 
        
        lmp_aux.append(list(df_node.iloc[:,4].values))
        
    LMP_def = np.append(LMP_def,np.array(lmp_aux),axis=1)
        
LMP_def = LMP_def[:,1:]

# %% Plot results

fig = go.Figure()

i=0
for n_node in HUBS:
    
    fig.add_trace(go.Scatter(
        #x=TIME,#dt_aux['Time Stamp'],#x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        y=LMP_def[i],
        name=n_node
    ))
    i = i + 1

# Add figure title and legend
fig.update_layout(title_text="LMPs in 2018 at ERCOT",showlegend=True)
# Set x-axis title
fig.update_xaxes(title_text="Hour")
# Set y-axis title
fig.update_yaxes(title_text="Price($/MW)", range=[-10.0,315.0])
# Set domain
#fig.update_layout(domain=[0.0, 1.0])

fig.show()

# %% Create useful df for basic stats
# column_names = [n[3:] for n in node_names]
HUBS = ['TX_'+HUBS[i].split('_')[1] for i in range(len(HUBS))]
filtered_df = pd.DataFrame(data = np.transpose(LMP_def), columns = HUBS)
#for cont in range(24):
#    filtered_df = filtered_df.drop([cont])

fig = go.Figure()
for node in HUBS:
    #print(filtered_df[node].describe())
    #print('\n')
    fig.add_trace(go.Box(y=filtered_df[node], name=node))


#fig.add_trace(go.Box(y=LMP[n_node], name='Sample A',
#                marker_color = 'indianred'))
#fig.add_trace(go.Box(y=y1, name = 'Sample B',
#                marker_color = 'lightseagreen'))

fig.show()

# %% Save data extracted
#filtered_df.to_csv('/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/filtered/data-ercot.csv', index=False)








