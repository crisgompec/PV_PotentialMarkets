#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:01:59 2020

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

HUB_NAMES = ['MINN.HUB', 'MICHIGAN.HUB', 'ILLINOIS.HUB', 'INDIANA.HUB', 'ARKANSAS.HUB', 'MS.HUB', 'TEXAS.HUB', 'LOUISIANA.HUB', 'AMMO.CALLAWAY1', 'ALTW.DAEC', 'WPS.WESTON3', 'GRE.COALC2_AC' ,'OTP.BIGSTON1','MDU.LEWIS1']
STATE_NAMES = ['MINNESOTA', 'MICHIGAN', 'ILLINOIS', 'INDIANA', 'ARKANSAS', 'MISSISIPI', 'TEXAS', 'LOUISIANA', 'MISOURI', 'IOWA', 'WISCONSIN', 'N-DAKOTA', 'S-DAKOTA','MONTANA']
dict_hubs = {HUB_NAMES[i] : STATE_NAMES[i] for i in range(len(HUB_NAMES))}

path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/miso/'
FILES = os.listdir(path)
FILES.remove('.DS_Store')

dict_dtype_data = {'HE'+str(i) : float for i in range(1,25)}

# %% Eliminate issue of commas + (-) signs
# with open(path + sorted(FILES)[3], 'r') as infile, \
#       open(path + sorted(FILES)[3].split('.')[0] + '_fix.csv', 'w') as outfile:
#     data = infile.read()
#     data = data.replace('"', "")
#     data = data.replace(',1,', ",1")
#     data = data.replace(',-1,', ",-1")
#     data = data.replace('-1-','-1.0,')
#     data = data.replace('1-','1.0,')
#     outfile.write(data)
#     infile.close()
#     outfile.close()


# %% Tests
# cont_node = 0
# lmps_node = np.array([])
# lmps_node = np.append(lmps_node,np.array([[1,2]]))
# lmps_node = np.append(lmps_node,np.array([[1,2]]))
# l_4 = np.array([[1,2,4,4]])
# l_fin = np.append(l_4,np.array([np.transpose(lmps_node)]),axis=0)
LMP_def = np.zeros([14,1])

#FILES = ['4quarter2018_fix.csv']
for csv_file in tqdm(sorted(FILES)):
    LMPs_quarter = []
    dataset = pd.read_csv(path+csv_file, dtype=dict_dtype_data)
    
    for hub_name in HUB_NAMES:
        
        df_node = dataset[dataset['NODE']==hub_name] 
        df_node = df_node[df_node['VALUE']=='LMP']
        
        lmp_aux = []
        
        for idx in range(len(df_node.index)):
            #lmp_aux = np.append(lmp_aux, df_node.iloc[idx,4:].values)
            lmp_aux.append(list(df_node.iloc[idx,4:].values))
        
        lmp_aux = np.asarray(lmp_aux).flatten()
        LMPs_quarter.append(lmp_aux) #= np.append(np.array([LMPs_quarter]),np.array([lmp_aux]),axis=0)

    LMP_def = np.append(LMP_def,np.array(LMPs_quarter),axis=1)

LMP_def = LMP_def[:,1:]

# %% Display and save data
fig = go.Figure()

i=0
for n_node in STATE_NAMES:
    
    fig.add_trace(go.Scatter(
        #x=TIME,#dt_aux['Time Stamp'],#x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
        y=LMP_def[i],
        name=n_node
    ))
    i = i + 1

# Add figure title and legend
fig.update_layout(title_text="LMPs in 2018 at miso",showlegend=True)
# Set x-axis title
fig.update_xaxes(title_text="Hour")
# Set y-axis title
fig.update_yaxes(title_text="Price($/MW)", range=[-150.0,500.0])
# Set domain
#fig.update_layout(domain=[0.0, 1.0])

fig.show()


# %% Create useful df for basic stats
# column_names = [n[3:] for n in node_names]
filtered_df = pd.DataFrame(data = np.transpose(LMP_def), columns = STATE_NAMES)
#for cont in range(24):
#    filtered_df = filtered_df.drop([cont])

fig = go.Figure()
for node in STATE_NAMES:
    print(filtered_df[node].describe())
    print('\n')
    fig.add_trace(go.Box(y=filtered_df[node], name=node))


#fig.add_trace(go.Box(y=LMP[n_node], name='Sample A',
#                marker_color = 'indianred'))
#fig.add_trace(go.Box(y=y1, name = 'Sample B',
#                marker_color = 'lightseagreen'))

fig.show()

# %% Save data extracted
filtered_df.to_csv('/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/filtered/data-miso.csv', index=False)



# %% Deal with final result - la ultima parece que no es float, sino string!
# LMP_yearly = []
# LMP_yearly = [item for l in LMP_final for item in LMP_final]


# LMP_def = []
# for season in range(len(FILES)):
#     LMP = []
#     LMP_aux = []
#     LMP_aux_2 = []
    
#     for i in range(int(len(LMP_yearly)/len(FILES))):
#         LMP.append(LMP_yearly[i])
    
    
#     for i in range(len(LMP)): # Para cada season
#         for j in range(len(LMP[i])): # Para cada nudo
#             for lmp in range(len(LMP[i][j])): # Para cada lmp
#                 LMP_aux_2.append(LMP[i][j][lmp])
#             LMP_aux.append(LMP_aux_2)
                
#     #LMP_aux = [item for i in LMP for l in LMP for item in LMP] # Flatten each season
#     LMP_def.append(LMP_aux)

# LMP_np = np.array(LMP_def)




