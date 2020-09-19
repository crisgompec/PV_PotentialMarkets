#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:48:03 2020

@author: peces
@description: Use processed data from ISOs and irradiation to determine benefits
"""
import pandas as pd
import numpy as np


path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/'
irradiance_names = ['central.csv', 'capital.csv']

ny_data = pd.DataFrame(path + 'filtered/data-ny-f.csv')

for irr_name in irradiance_names:
    data_irradiation = pd.DataFrame(path + irr_name)
    
