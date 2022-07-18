#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:41:37 2022

@author: arthur
"""

import numpy as np
import pandas as pd
import os
from opts import parser
args = parser.parse_args()

root_path=os.getcwd()

task_list=['multi','bekVSall','melVSall','nevVSall','bekVSmel','bekVSnev','melVSnev']

for task in task_list:
    # Display to collect the user input
    args.store_name = '_'.join(['isic2018', 'efficientb5', task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
    
    df_data=pd.read_csv(root_path+'/results/'+args.store_name+'/test_history.csv')
    df_data=df_data.drop(columns='Unnamed: 0')
    
    print(f'Statistique for {task}')
    print('------------------------------------------------------------')
    print('[mean bacc %.2f] ,[std bacc %.2f] ,  [mean auc %.2f],  [std auc %.2f]' % (np.mean(df_data.bacc),np.std(df_data.bacc),np.mean(df_data.auc),np.std(df_data.auc)))
    print('------------------------------------------------------------')

idx_best=7
# correspond to 
best_thresh=[0.1,0.5]
pip_perform=pd.DataFrame()
for i in range(10):
    df_data=pd.read_csv(root_path+'/stats/stats_pipelines_predictions_run'+str(i)+'.csv')
    df_data=df_data.drop(columns='Unnamed: 0')
    row=df_data.iloc[[idx_best]]
    pip_perform=pd.concat([pip_perform,row])

print(f'Statistique for with threshold: {best_thresh}')
print('------------------------------------------------------------')
print('[mean bacc %.2f] ,[std bacc %.2f] ,  [mean auc %.2f],  [std auc %.2f]' % (np.mean(pip_perform.BACC),np.std(pip_perform.BACC),np.mean(pip_perform.Mean_AUC),np.std(pip_perform.Mean_AUC)))
print('------------------------------------------------------------')
print('------------------------------------------------------------')
print('[MEL_auc %.2f] ,[std %.2f] ,  [BEK auc %.2f],  [std %.2f],  [NEV auc %.2f],  [std %.2f]' % (np.mean(pip_perform.MEL_AUC),np.std(pip_perform.MEL_AUC),np.mean(pip_perform.BEK_AUC),np.std(pip_perform.BEK_AUC),np.mean(pip_perform.NEV_AUC),np.std(pip_perform.NEV_AUC)))
print('------------------------------------------------------------')
