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
path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/neiso/'
os.chdir(path)
#dataset['LBMP ($/MWHr)'].describe()
n_nodes = 9 # 8 nodes + 1 init
n_hours = 24
hours = [str(i) for i in range(1,25)]
for i in range(9):
    hours[i] = '0' + hours[i]
LMP = np.zeros([n_nodes, n_hours]) # I want 24 measures a day (24h)
TIME = pd.DataFrame()

# Data access
# Eliminate garbage row - ONLY APPLY ONCE
FLAG_ELIMINATE_ROWS = False
folders = os.listdir(path) # TO TRY: 
folders.remove('.DS_Store')


# Bucle de meses
for month in tqdm(sorted(folders)):
    print(month)
    files = os.listdir(path + month) # TO TRY: 
    #print('\nMONTH: ' + month + '\n')
    
    # Bucle de días
    for day in sorted(files):
        #print(day)
        
        
        try:
            dataset = pd.read_csv(path + month + '/' +day, header=4)
            dataset = dataset.drop(dataset.index[0])
            columnas = dataset.columns
            d_useful = dataset[dataset['Location Type'].values=='LOAD ZONE']
            node_names = pd.unique(d_useful['Location Name'])
        except ValueError as e:
            print('Error while reading data from ' + day)
            
        
        # Bucle de nodos
        LMP_aux = np.zeros([1, n_hours])
        #lmps = pd.DataFrame(columns=columnas[])
        for node in node_names:
                
                d_aux_lmp = d_useful[d_useful['Location Name']==node]
                lmp_node = np.transpose(d_aux_lmp['Locational Marginal Price'].values)
                lmp_node = np.array([lmp_node.astype(float)])
                #print(np.array([lmp_node.astype(float)]))
                # Fix lack of data
                try:
                    LMP_aux = np.append(LMP_aux,lmp_node,axis=0)
                except: 
                    while lmp_node.shape[1] < n_hours:
                        print('Entro en while')
                        print('Diferent size detected: ' + str(lmp_node.shape[1]))
                        print(lmp_node.shape[1] < n_hours)
                        lmp_node = np.array([np.append(lmp_node, np.mean(lmp_node))])
                        print('Appendizado correctamente')
                        
                    while lmp_node.shape[1] > n_hours:
                        print('Entro en 2 while: ' + day)
                        print(lmp_node)
                        lmp_node = np.array([np.delete(lmp_node, 2)])

                    LMP_aux = np.append(LMP_aux,lmp_node,axis=0)
                    
                    
                
        try:
            #TIME = TIME.append(dt_aux['Time Stamp'],ignore_index=True)
            #print(LMP)
            #print(LMP_aux)
            LMP = np.append(LMP,LMP_aux,axis=1)
        except ValueError as e:
            print(e)
            print(LMP_aux[1:])
            a = input()
    

# Plot prices 
fig = go.Figure()

i=1
for n_node in node_names:
    
    fig.add_trace(go.Scatter(
        #x=TIME,#dt_aux['Time Stamp'],#x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        y=LMP[i],
        name=n_node
    ))
    i = i + 1

# Add figure title and legend
fig.update_layout(title_text="LMPs in 2018 at NEISO",showlegend=True)
# Set x-axis title
fig.update_xaxes(title_text="Hour")
# Set y-axis title
fig.update_yaxes(title_text="Price($/MW)", range=[-50.0,300.0])
# Set domain
#fig.update_layout(domain=[0.0, 1.0])

fig.show()

#Create useful df for basic stats
column_names = [n[3:] for n in node_names]
filtered_df = pd.DataFrame(data = np.transpose(LMP[1:,:]), columns = column_names)
for cont in range(24):
    filtered_df = filtered_df.drop([cont])

fig = go.Figure()
for node in column_names:
    print(filtered_df[node].describe())
    print('\n')
    fig.add_trace(go.Box(y=filtered_df[node], name=node))


#fig.add_trace(go.Box(y=LMP[n_node], name='Sample A',
#                marker_color = 'indianred'))
#fig.add_trace(go.Box(y=y1, name = 'Sample B',
#                marker_color = 'lightseagreen'))

fig.show()

# Save data extracted
filtered_df.to_csv('/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/filtered/data-ne.csv', index=False)



