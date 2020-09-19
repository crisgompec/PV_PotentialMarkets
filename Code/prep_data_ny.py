#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21,  2020
@author: peces
@description: read data from NY ISO tables

"""
import pandas as pd
import numpy as np
import os
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
pio.renderers.default = "browser"

# Initializations 
path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/nyiso/'
os.chdir(path)
#dataset['LBMP ($/MWHr)'].describe()
n_nodes = 16 # 15 nodes + 0 init
n_hours = 24
LMP = np.empty([n_nodes, n_hours]) # I want 25 measures a day (24h)
TIME = pd.DataFrame()

# Data access
folders = os.listdir(path) # TO TRY: 
folders.remove('.DS_Store')

# Bucle de meses
for month in tqdm(sorted(folders)):

    files = os.listdir(path + month) # TO TRY: 
    #print('\nMONTH: ' + month + '\n')
    
    # Bucle de días
    for day in sorted(files):
        #print(day)
        
        try:
            dataset = pd.read_csv(path + month + '/' +day)
            columnas = dataset.columns
            node_names = pd.unique(dataset['Name'])
        except Except as e:
            print('Error while reading data from ' + day)
            
        
        # Bucle de nodos
        LMP_aux = np.empty([1, n_hours])
        for node in node_names:
            #print(node)
            # Filter by node and integer hour (dataset['Name']=='N.Y.C.')] & (dataset['Time Stamp'][:][14:16]=='00')
            #dt_aux = dataset[(dataset['Name']== node) & (dataset['Time Stamp'][:][14:19]=='00:00')]
            
            try:
                dt_aux = pd.DataFrame(columns=columnas)
                for i in range(0,len(dataset)):
                    if dataset['Name'][i]== node and (dataset['Time Stamp'][i][14:19] == '00:00'):
                        dt_aux = dt_aux.append(dataset[i:i+1])
                        #print(dataset['Time Stamp'][i])
                        
                LMP_aux = np.append(LMP_aux,[dt_aux['LBMP ($/MWHr)']], axis=0) # Axis 0 to add next row
            except ValueError:
                print('Value error at file ' + day)
                
        try:
            TIME = TIME.append(dt_aux['Time Stamp'],ignore_index=True)
            #print(LMP)
            #print(LMP_aux)
            LMP = np.append(LMP,LMP_aux,axis=1)
        except ValueError:
            print('Value error while appending in file ' + day)
    

#corrupt_nodes = ['CAPITL','WEST', 'H Q', 'CENTRL', 'NPX', 'HUD VL']
#for i in corrupt_nodes:
#    node_names.tolist().remove(i)

# Plot prices 
fig = go.Figure()

for n_node in range(1,len(node_names)+1):
    print(LMP[n_node].shape)
    fig.add_trace(go.Scatter(
        #x=TIME,#dt_aux['Time Stamp'],#x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        y=LMP[n_node],
        name=node_names[n_node-1]
    ))

# Add figure title and legend
fig.update_layout(title_text="LMPs in 2018 at NYISO",showlegend=True)
# Set x-axis title
fig.update_xaxes(title_text="Hour")
# Set y-axis title
fig.update_yaxes(title_text="Price($/MW)", range=[-250.0,1100.0])
# Set domain
#fig.update_layout(domain=[0.0, 1.0])

fig.show()

#Create useful df for basic stats
filtered_df = pd.DataFrame(np.transpose(LMP[1:,:]), columns = node_names)

fig = go.Figure()
for node in node_names:
    print(filtered_df[node].describe())
    print('\n')
    fig.add_trace(go.Box(y=filtered_df[node], name=node))


#fig.add_trace(go.Box(y=LMP[n_node], name='Sample A',
#                marker_color = 'indianred'))
#fig.add_trace(go.Box(y=y1, name = 'Sample B',
#                marker_color = 'lightseagreen'))

fig.show()

# Save data extracted
#filtered_df.to_csv('/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/filtered/data-ny.csv', index=False)



