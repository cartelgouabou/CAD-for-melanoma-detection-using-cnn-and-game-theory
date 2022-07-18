#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:40:46 2022

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
import argparse
# '''
from opts import parser
args = parser.parse_args()
device = torch.device("cpu")

root_path=os.getcwd()



if args.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''
args.run = input("Which run do you want to execute for generate_prediction?")
# Display to collect the user input
print('--You wrote--')
print(args.run)
#args.run='run0'

args.weighting_type='CS'
args.loss_type='CE'
args.task='multi'

df_test=pd.read_csv(root_path+'/base/'+args.task+'/df_val.csv')
df_test=df_test.drop(columns='Unnamed: 0')



for i in range(len(df_test)):
    df_test.path[i]=root_path+df_test.path[i][45:]


# cls_num_list= pd.read_csv(root_path+'/base/cls_num_list_pd.csv')
# cls_num_list=cls_num_list.drop(columns='Unnamed: 0')
# cls_num_list=cls_num_list['0']
# cls_num_list=cls_num_list.tolist()

model=torch.load(root_path+'/model/efficientNetB5_multi.pth')
print('[INFO]: Set all trainable layers to TRUE...')

input_size = 456

# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.59451115, 0.59429777, 0.59418184],
                                                            [0.14208533, 0.18548788, 0.20363748])])


# Same for the test set:
test_set = CustomImageDataset(df_test, transform=val_transform)
test_loader = DataLoader(test_set, batch_size=args.batch_size//2, shuffle=False, num_workers=args.workers)

df_result=pd.DataFrame(columns=['path','image_id','label','label_idx','B_M','M_M','N_M','B_BR','M_MR','N_NR','B_BM','M_BM','B_BN','N_BN','M_MN','N_MN'])
df_result.path=df_test.path
df_result.label=df_test.label
df_result.image_id=df_test.image_id
df_result.label_idx = pd.Categorical(df_result['label']).codes

# run_list=[]
# for run in range(args.num_runs):
#     run_list.append('run'+str(run))
#run_list=['run0']





args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])


#Multiclasse        
print ('Evaluating the multiclass model')
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        prediction = outputs.max(1, keepdim=True)[1]
        y_true.append(labels.cpu().numpy())
        y_pred.append(prediction.detach().cpu().numpy())
        y_score.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_true_2D= F.one_hot(torch.from_numpy(y_true), num_classes=3).cpu().numpy()
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    test_bacc = balanced_accuracy_score(y_true, y_pred)
    test_auc=roc_auc_score(y_true_2D, y_score)
print(f'Multiclass prediciont model on {args.run}')
print('------------------------------------------------------------')
print('[test bacc %.5f] ,  [test auc %.5f]' % (test_bacc,test_auc))
print('------------------------------------------------------------')
plot_labels = ['BEK','MEL','NEV']
report = classification_report(y_true, y_pred, target_names=plot_labels)
print(report)
y_score_multi_pd=pd.DataFrame(y_score,columns=['BEK','MEL','NEV'])
    
df_result.B_M=y_score_multi_pd.BEK
df_result.M_M=y_score_multi_pd.MEL
df_result.N_M=y_score_multi_pd.NEV

#Binary
model=torch.load(root_path+'/model/efficientNetB5_bin.pth')

#task bekVSall
args.task='bekVSall'
args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
       
print ('Evaluating the model for task: '+args.task)
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        y_score.append(probs.detach().cpu().numpy())
    y_score = np.concatenate(y_score)
y_score_bVa_pd=pd.DataFrame(y_score,columns=['AUT','BEK'])
df_result.B_BR=y_score_bVa_pd.BEK

#task melVSall
args.task='melVSall'
args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
      
print ('Evaluating the model for task: '+args.task)
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        y_score.append(probs.detach().cpu().numpy())
    y_score = np.concatenate(y_score)
y_score_mVa_pd=pd.DataFrame(y_score,columns=['AUT','MEL'])
df_result.M_MR=y_score_mVa_pd.MEL

#task nevVSall
args.task='nevVSall'
args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
      
print ('Evaluating the model for task: '+args.task)
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        y_score.append(probs.detach().cpu().numpy())
    y_score = np.concatenate(y_score)
y_score_nVa_pd=pd.DataFrame(y_score,columns=['AUT','NEV'])
df_result.N_NR=y_score_nVa_pd.NEV

#task bekVSmel
args.task='bekVSmel'
args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
      
print ('Evaluating the model for task: '+args.task)
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        y_score.append(probs.detach().cpu().numpy())
    y_score = np.concatenate(y_score)
y_score_bVm_pd=pd.DataFrame(y_score,columns=['BEK','MEL'])
df_result.B_BM=y_score_bVm_pd.BEK
df_result.M_BM=y_score_bVm_pd.MEL

#task bekVSnev
args.task='bekVSnev'
args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
      
print ('Evaluating the model for task: '+args.task)
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        y_score.append(probs.detach().cpu().numpy())
    y_score = np.concatenate(y_score)
y_score_bVn_pd=pd.DataFrame(y_score,columns=['BEK','NEV'])
df_result.B_BN=y_score_bVn_pd.BEK
df_result.N_BN=y_score_bVn_pd.NEV


#task melVSnev
args.task='melVSnev'
args.store_name = '_'.join(['isic2018', 'efficientb5', args.task,args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
      
print ('Evaluating the model for task: '+args.task)
model.load_state_dict(torch.load(root_path+args.model_path+args.store_name+'/'+f'best_model_{args.run}.pth'))
model.to(device)
model.eval()
y_true = []
y_pred = []
y_score=[]
with torch.no_grad():
    for _, data in enumerate(test_loader):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        probs = torch.softmax(outputs,dim=1)
        y_score.append(probs.detach().cpu().numpy())
    y_score = np.concatenate(y_score)
y_score_mVn_pd=pd.DataFrame(y_score,columns=['MEL','NEV'])
df_result.M_MN=y_score_mVn_pd.MEL
df_result.N_MN=y_score_mVn_pd.NEV

df_result['B_conf']=np.abs(df_result.B_M-df_result.B_BR)
df_result['M_conf']=np.abs(df_result.M_M-df_result.M_MR)
df_result['N_conf']=np.abs(df_result.N_M-df_result.N_NR)

folder='pipelines'
save_folder_path=root_path+'/'+folder+'/'
if not os.path.exists(save_folder_path):
    print('creating ' + folder + ' folder where to save prediction')
    os.mkdir(folder)
    
df_result.to_csv(root_path+'/pipelines/'+f'val_set_all_predictions_{args.run}.csv')




