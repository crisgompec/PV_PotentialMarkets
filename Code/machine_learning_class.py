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
from matplotlib.colors import ListedColormap
pio.renderers.default = "browser"

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


path = '/Users/peces/OneDrive - Georgia Institute of Technology/Winter2020/Electriciy_Markets/Project/data/results/'

# Draw data
df = pd.read_csv(path + 'FinalResults2.csv')
df_irr_lmp = df.copy()

df_irr_lmp['size'] = df_irr_lmp['BC_ratio']**6
perfomance = []


# %% Read data 
# Importing the dataset
X = df[['Sun','std']].values #df.iloc[:, [2, 12]].values
y = df['BC_ratio'].values

for i in range(len(y)):
    #print(y[i])
    if y[i]>=1:
        y[i] = 1
    else:
        y[i] = 0
y = y.astype(int)

perfo_1= 0
perfo_2= 0
perfo_3= 0
perfo_4 = 0
perfo_5 = 0.0

for i in tqdm(range(1000)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    perfomance = []
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    m1 = np.mean(X_train[:,0])
    d1 = np.std(X_train[:,0])
    m2 = np.mean(X_train[:,1])
    d2 = np.std(X_train[:,1])
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    

    # %% Logistic Regression
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Evaluating perfomance
    cm_logreg = confusion_matrix(y_test, y_pred)
    
    perfomance.append(np.trace(cm_logreg)/np.sum(cm_logreg))
    
    # %% K-neareast Neighbors
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm_knn = confusion_matrix(y_test, y_pred)
    perfomance.append(np.trace(cm_knn)/np.sum(cm_knn))
    
    # %% Support Vector Machine
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'sigmoid', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm_svm = confusion_matrix(y_test, y_pred)
    perfomance.append(np.trace(cm_svm)/np.sum(cm_svm))
    
    # %% Naive-Bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm_nb = confusion_matrix(y_test, y_pred)
    perfomance.append(np.trace(cm_nb)/np.sum(cm_nb))
    
    # %% # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred)
    perfomance.append(np.trace(cm_rf).astype(float)/np.sum(cm_rf))
    
    # %% Visualizing results
    #print(perfomance)
    
    perfo_1 = perfo_1 +perfomance[0]
    perfo_2= perfo_2+perfomance[1]
    perfo_3= perfo_3+perfomance[2]
    perfo_4= perfo_4+perfomance[3]
    perfo_5 = perfo_5 + perfomance[4]
    
    # Visualising the Training set results
    
    
    # X_set, y_set = X_train, y_train
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    #                       np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    # plt.contourf(X1*d1+m1, X2*d2+m2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #               alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    
    # plt.xlim(X1.min()*d1+m1, X1.max()*d1+m1)
    # plt.ylim(X2.min()*d2+m2, X2.max()*d2+m2)
    
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c = ListedColormap(('red', 'green'))(i), label = j)
        
    # plt.title('Logistic Regression (Training set)')
    # plt.xlabel('std')
    # plt.ylabel('Irradiance')
    # plt.legend()
    # plt.show()
    
    # # Visualising the Test set results
    # X_set, y_set = X_test, y_test
    # #X_set, y_set = X, y
    
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    #                       np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    # sc
    # a = plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #               alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # #X_set = X_set*d+m
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c = ListedColormap(('red', 'green'))(i), label = j)
    # plt.title('Logistic Regression (Test set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()

# %% 
# x_aux = sc.transform(X)
# x_pred = np.zeros([1,2])
# y_predicted = np.zeros([len(X[:,0]),len(X[:,1])])
# #y_predicted[i,j] = classifier.predict()

# for i in range(len(X[:,0])):
#     for j in range(len(X[:,1])):
#         x_pred[0,0] = x_aux[i,0]
#         x_pred[0,1] = x_aux[j,1]
#         y_predicted[i,j] = classifier.predict(x_pred)
#         print(classifier.predict(x_pred))
        
# #y_predicted = classifier.predict(sc.transform(X))
# #y_predicted = classifier.predict(np.array([X[:,0].ravel(), X[:,1].ravel()]).T).reshape(X[:,0].shape)
# fig = go.Figure(data =
#     go.Contour(
#         z=y_predicted,
#         x=X[:,0],
#         y=X[:,1],
#         colorscale='Hot',
#         contours=dict(
#             start=0,
#             end=1.2,
#             size=1000,
#         ),
#     ))

# fig.show()


# # %% 

# import plotly.graph_objects as go

# fig = go.Figure()

# # Create scatter trace of text labels
# fig.add_trace(go.Scatter(
#     x=[2, 1, 8, 8],
#     y=[0.25, 9, 2, 6],
#     text=["Filled Triangle",
#           "Filled Polygon",
#           "Quadratic Bezier Curves",
#           "Cubic Bezier Curves"],
#     mode="text",
# ))

# # Update axes properties
# fig.update_xaxes(
#     range=[0, 9],
#     zeroline=False,
# )

# fig.update_yaxes(
#     range=[0, 11],
#     zeroline=False,
# )

# # Add shapes
# fig.update_layout(
#     shapes=[
#         # Quadratic Bezier Curves
#         dict(
#             type="path",
#             path="M 4,4 Q 6,0 8,4",
#             line_color="RoyalBlue",
#         ),
#         # Cubic Bezier Curves
#         dict(
#             type="path",
#             path="M 1,4 C 2,8 6,4 8,8",
#             line_color="MediumPurple",
#         ),
#         # filled Triangle
#         dict(
#             type="path",
#             path=" M 1 1 L 1 3 L 4 1 Z",
#             fillcolor="LightPink",
#             line_color="Crimson",
#         ),
#         # filled Polygon
#         dict(
#             type="path",
#             path=" M 3,7 L2,8 L2,9 L3,10, L4,10 L5,9 L5,8 L4,7 Z",
#             fillcolor="PaleTurquoise",
#             line_color="LightSeaGreen",
#         ),
#     ]
# )

# fig.show()


# # %% 
# df_irr_lmp = df.copy()
# df_irr_lmp['size'] = df_irr_lmp['BC_ratio']**6

# fig = px.scatter(df_irr_lmp, x="Sun", y="std", size='size',
#            hover_name="State", color_continuous_scale=px.colors.sequential.Viridis, size_max=12, opacity=1)#, log_x=False, size_max=60)

# fig.update_layout(
#         title_text = 'B/C vs. Irradiance vs. Volatility',# <br>(Click legend to toggle traces)',#title_text = '2014 US city populations<br>(Click legend to toggle traces)', # 
#         showlegend = True,
#         xaxis_title="Irradiance (W/m^2)",
#         yaxis_title = "Standard deviation (%)",
#         shapes=[
#             # filled Triangle
#             dict(
#                 type="path",
#                 path=" M 140 0 L 140 25 L 245 0 Z",
#                 fillcolor="LightPink",
#                 line_color="Crimson",
#                 opacity=0.5,
#                 line_width=0,
#             ),
            
#             dict(
#             type="path",
#             path=" M 140 25 L 140 90 L 245 90 L 245 0 Z",
#             fillcolor="PaleTurquoise",
#             line_color="LightSeaGreen",
#             opacity=0.5,
#             line_width=0,
#         ),
#             dict(type="line", xref="x1", yref="y1",
#             x0=140, y0=25, x1=245, y1=0, line_width=3),
#         ]
#     )
            
# fig.show()