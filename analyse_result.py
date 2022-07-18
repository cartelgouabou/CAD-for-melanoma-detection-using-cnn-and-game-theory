#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 01:34:57 2022

@author: arthur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.ion()   # interactive mode
from tqdm import tqdm
import os
import torch

import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from data_utils import CustomImageDataset

# sklearn libraries
from sklearn.metrics import classification_report,confusion_matrix, balanced_accuracy_score, roc_auc_score,roc_curve
import warnings
from opts import parser
args = parser.parse_args()
device = torch.device("cpu")

root_path=os.getcwd()



if args.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''

args.run = input("Which run do you want to execute?")
# Display to collect the user input
print('--You wrote--')
print(args.run)

os.system(root_path+'/'+"generate_prediction_scores.py")
os.system(root_path+'/'+"pipelines_predictions.py")

args.weighting_type='CS'
args.loss_type='CE'
args.task='multi'



thresh_list=[[0.05,0.2],[0.05,0.3],[0.05,0.4],[0.05,0.5],[0.1,0.2],[0.1,0.3],[0.1,0.4],[0.1,0.5],[0.2,0.3],[0.2,0.4],[0.2,0.5],[0.3,0.4],[0.3,0.5]]


stat_result=pd.DataFrame(columns=['u1','u2','BACC','Mean_AUC','BEK_AUC','MEL_AUC','NEV_AUC'])
for u1,u2 in thresh_list:
    df_result=pd.read_csv(root_path+'/pipelines/'+f'pipelines_predictions_{args.run}_u1_'+str(u1)+'_u2_'+str(u2)+'.csv')
    df_result=df_result.drop(columns='Unnamed: 0')
    
    
    
    y_true = np.array(df_result.label_idx)
    y_true_2D= F.one_hot(torch.from_numpy(y_true), num_classes=3).cpu().numpy()
    y_score=[]
    for i in range(len(df_result)):
        y_score.append([df_result.BEK_sco[i],df_result.MEL_sco[i],df_result.NEV_sco[i]])
    y_score=np.array(y_score)
    y_pred = np.array([np.argmax(y_score[p]) for p in range(len(y_score))])
    
    test_bacc = balanced_accuracy_score(y_true, y_pred)
    test_auc=roc_auc_score(y_true_2D, y_score)
    
    
    print(f'pipelines_predictions_{args.run}_u1_'+str(u1)+'_u2_'+str(u2))
    print('------------------------------------------------------------')
    print('[test bacc %.5f] ,  [test auc %.5f]' % (test_bacc,test_auc))
    print('------------------------------------------------------------')
    plot_labels = ['BEK','MEL','NEV']
    report = classification_report(y_true, y_pred, target_names=plot_labels)
    print(report)
    
    fpr = dict()
    tpr=dict()
    thd=dict()
    roc_auc=dict()
    for i in range(3):      
        fpr[i], tpr[i], thd[i] = roc_curve(y_true_2D[:,i], y_score[:,i])
        roc_auc[i]=roc_auc_score(y_true_2D[:,i], y_score[:,i],average='weighted')
    
        
    
    stat_result.loc[len(stat_result)]=[u1,u2,test_bacc,test_auc,roc_auc[0],roc_auc[1],roc_auc[2]]

folder='stats'
save_folder_path=root_path+'/'+folder+'/'
if not os.path.exists(save_folder_path):
    print('creating ' + folder + ' folder where to save prediction')
    os.mkdir(folder)
    
stat_result.to_csv(root_path+'/stats/'+f'stats_pipelines_predictions_{args.run}'+'.csv')
