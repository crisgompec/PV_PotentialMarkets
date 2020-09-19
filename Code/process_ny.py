#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:23:46 2020

@author: peces
@description: Process all data related to NY
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

# First definitions
path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/filtered/'
GHI_NAME = 'irradiation.csv'
LMP_NAME = 'data-ny-f.csv'
N_HOURS = 8712
EFFICIENCY = 0.2
SIZE_PV = 10000# In m^2
LCOE_NY = 3.5e-2 * 1000 #In $/MWh
WACC = 0.0075
DEVIATION_LMP = 0.1
DEVIATION_GENERATION = 0.1


# Read inyectable quantity of MW, according to location - assumes efficiency 20% and 1 hectarea - in MW
iny_power = (pd.read_csv(path + GHI_NAME)).multiply(SIZE_PV*EFFICIENCY*1e-6)
iny_power['N_hour'] = iny_power['N_hour'].multiply(1/(SIZE_PV*EFFICIENCY*1e-6))
iny_power = iny_power[iny_power.columns[0:9]]

# Read hourly marginal prices files - in $/MWh
lmp_ny = pd.read_csv(path + LMP_NAME)
node_names = lmp_ny.columns

# Create dictionary to assign nodes with geographical areas
dict_nodes_areas = {
    'NY_CAPITL':'capital.csv',
    'NY_CENTRL': 'central.csv',
    'NY_DUNWOD':'south.csv',
    'NY_GENESE': 'genesse.csv',
    'NY_HQ':'mo_valley.csv',
    'NY_HUDSON_VALLEY': 'south.csv',
    'NY_LONG_ISLAND':'nyc.csv',
    'NY_MILLWD': 'south.csv',
    'N.Y.C.':'nyc.csv',
    'NY_NORTH': 'north.csv',
    'NY_NPX':'capital.csv',
    'NY_OH': 'mo_valley.csv',
    'NY_PJM':'central.csv',
    'NY_WEST': 'west.csv',
    'NY_MHK VL': 'mo_valley.csv'}


# Create dataframe of simulated earnings - 363 days - 8712 hours
#earnings =  np.multiply(lmp_ny.loc[:8712,['CAPITL']],iny_power.loc[:8711,[dict_nodes_areas['CAPITL']]])
#earnings = pd.DataFrame(columns=node_names)
#frames = []

ear_aux = np.zeros([1, N_HOURS])

for node in node_names:
    #earnings[node] =  np.multiply(lmp_ny.loc[:8712,[node]],iny_power.loc[:8711,[dict_nodes_areas[node]]])
    ear_aux = np.append(ear_aux, (np.multiply(lmp_ny.loc[:N_HOURS,[node]],iny_power.loc[:N_HOURS-1,[dict_nodes_areas[node]]].values).transpose()),axis=0)


earnings = pd.DataFrame(np.transpose(ear_aux[1:]), columns = node_names) # in $
earnings[earnings < 0] = 0

# Plot yearly earnings
fig = go.Figure()
for node in node_names:
    fig.add_trace(go.Scatter(
        #x=TIME,#dt_aux['Time Stamp'],#x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        y=earnings[node],
        name=node
    ))
# Add figure title and legend
fig.update_layout(title_text="Earnings Areas of NY ISO",showlegend=True)
# Set x-axis title
fig.update_xaxes(title_text="Hour")
# Set y-axis title
fig.update_yaxes(title_text="Earnings($/h)", range=[-250.0,1100.0])
# Set domain
#fig.update_layout(domain=[0.0, 1.0])
fig.show()
    
# Now we integrate the earnings along the year simulation
total_yearly_earnings = np.zeros([len(node_names),1])
total_yearly_generation = np.zeros([len(node_names),1])
for i in range(len(node_names)):
    total_yearly_earnings[i] =  np.trapz(earnings[node_names[i]]) # $
    total_yearly_generation[i] = np.trapz(iny_power[dict_nodes_areas[node_names[i]]]) # MWh




# Cost estimation
# Information from NREL: In NY, the LCOE for 2018 is 4.2cents/KWh
# Associated to node_names
costs_yearly = LCOE_NY * total_yearly_generation
profit_yearly = total_yearly_earnings-costs_yearly


# Simulation for 25 years
def randomize(value,deviation):
    return value*(1+np.random.normal(0,deviation))

total_yearly_earnings_25 = np.zeros([len(node_names),25])
total_yearly_generation_25 = np.zeros([len(node_names),25])
for i in tqdm(range(25)):
    
    ear_aux = np.zeros([1, N_HOURS])
    lmp_ny_aux = np.zeros([1, N_HOURS])
    iny_power_aux = np.zeros([1, N_HOURS])
    for node in node_names:
        lmp_ny_aux[0,:] = lmp_ny.apply(lambda x: randomize(x[node],DEVIATION_LMP),axis=1)
        iny_power_aux[0,:] = (iny_power.loc[:N_HOURS-1,[dict_nodes_areas[node]]]).apply(lambda x: randomize(x[dict_nodes_areas[node]],DEVIATION_GENERATION),axis=1)
        ear_aux = np.append(ear_aux, (np.multiply(lmp_ny_aux,iny_power_aux)),axis=0)
    
    earnings = pd.DataFrame(np.transpose(ear_aux[1:]), columns = node_names) # in $
    earnings[earnings < 0] = 0
    
    # Now we integrate the earnings along the year simulation
    total_yearly_earnings = np.zeros([len(node_names)])
    total_yearly_generation = np.zeros([len(node_names)])
    for j in range(len(node_names)):
        
        total_yearly_earnings[j] =  np.trapz(earnings[node_names[j]])/(1+WACC)**i # $
        total_yearly_generation[j] = np.trapz(iny_power[dict_nodes_areas[node_names[j]]]) # MWh
        
    #print('year ' + str(i+1) + ': ' )
    #print(total_yearly_earnings)
    
    total_yearly_earnings_25[:,i] = total_yearly_earnings
    total_yearly_generation_25[:,i] = total_yearly_generation
    

benefit_cost = np.empty([len(node_names)])
for i in range(len(node_names)):
    benefit_cost[i] = sum(total_yearly_earnings_25[i])/(sum(total_yearly_generation_25[i])*LCOE_NY)
    
    
    
    
    
    
    
    
    
    
    
    