#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:33:09 2022

@author: arthur
"""
import os
import numpy as np
import pandas as pd
from opts import parser
args = parser.parse_args()

root_path=os.getcwd()
args.run = input("Which run do you want to execute for pipelines_pred?")
# Display to collect the user input
print('--You wrote--')
print(args.run)


#args.run='run0'
df_result=pd.read_csv(root_path+'/pipelines/'+f'val_set_all_predictions_{args.run}.csv')
df_result=df_result.drop(columns='Unnamed: 0')



thresh_list=[[0.05,0.2],[0.05,0.3],[0.05,0.4],[0.05,0.5],[0.1,0.2],[0.1,0.3],[0.1,0.4],[0.1,0.5],[0.2,0.3],[0.2,0.4],[0.2,0.5],[0.3,0.4],[0.3,0.5]]
n=2

class_dict={0:'B_M',1:'M_M',2:'N_M'}
u_dict={0:'B_conf',1:'M_conf',2:'N_conf'}

for u1,u2 in thresh_list:
    df_result_pip=pd.DataFrame(columns=['BEK_sco','MEL_sco','NEV_sco','Pred_conf'])
    for i in range(len(df_result)):
        multi_class_score=[df_result.B_M[i],df_result.M_M[i],df_result.N_M[i]]
        idx_PredClass_M3=np.argmax(multi_class_score)
        Pred_Prob_M3=df_result[class_dict[idx_PredClass_M3]][i]
        if (df_result[u_dict[idx_PredClass_M3]][i]<=u1):
            df_result_pip.loc[i] = [df_result.B_M[i],df_result.M_M[i],df_result.N_M[i],df_result[u_dict[idx_PredClass_M3]][i]]
        elif   (df_result[u_dict[idx_PredClass_M3]][i]<=u2):
            sort_ind=(-np.array(multi_class_score)).argsort()[:n]
            if (0 in sort_ind) & (1 in sort_ind):
                df_result_pip.loc[i] = [df_result.B_BM[i],df_result.M_BM[i],0.0,df_result[u_dict[idx_PredClass_M3]][i]]
            elif (0 in sort_ind) & (2 in sort_ind):
                df_result_pip.loc[i] = [df_result.B_BN[i],0.0,df_result.N_BN[i],df_result[u_dict[idx_PredClass_M3]][i]]
            else:
                df_result_pip.loc[i] = [0.0,df_result.M_MN[i],df_result.N_MN[i],df_result[u_dict[idx_PredClass_M3]][i]]
        else:
            bek=np.max([df_result.B_M[i],df_result.B_BR[i],df_result.B_BM[i],df_result.B_BN[i]])
            mel=np.max([df_result.M_M[i],df_result.M_MR[i],df_result.M_BM[i],df_result.M_MN[i]])
            nev=np.max([df_result.N_M[i],df_result.N_NR[i],df_result.N_BN[i],df_result.N_MN[i]])
            df_result_pip.loc[i] = [bek/np.sum([bek,mel,nev]),mel/np.sum([bek,mel,nev]),nev/np.sum([bek,mel,nev]),df_result[u_dict[idx_PredClass_M3]][i]]
         
    
    df_result_pip['path']=df_result.path
    df_result_pip['label']=df_result.label
    df_result_pip['image_id']=df_result.image_id
    df_result_pip['label_idx'] = pd.Categorical(df_result['label']).codes
    df_result_pip.to_csv(root_path+'/pipelines/'+f'val_set_pipelines_predictions_{args.run}_u1_'+str(u1)+'_u2_'+str(u2)+'.csv')
    
   
