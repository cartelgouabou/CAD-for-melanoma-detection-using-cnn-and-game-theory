#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 07:54:36 2022

@author: arthur
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.ion()   # interactive mode
from tqdm import tqdm
import os
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import  compute_img_mean_std,prepare_folders
from data_utils import get_data_multi,CustomImageDataset,get_data_all,get_data_one
from train import train, validate,backup
from visualize import plot_confusion_matrix,learning_curve
from torch.optim.lr_scheduler import CyclicLR
# sklearn libraries
from sklearn.metrics import classification_report
import warnings
from tensorboardX import SummaryWriter
from opts import parser
args = parser.parse_args()
device = torch.device("cpu")

root_path=os.getcwd()



if args.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''

task=['multi','bekVSall','melVSall','nevVSall','bekVSmel','bekVSnev','melVSnev']

task='multi'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_multi(root_path+args.dataset_dir)

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')

task='bekVSall'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_all(root_path+args.dataset_dir,name_class1='BEK',name_class2='AUT')

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')


task='melVSall'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_all(root_path+args.dataset_dir,name_class1='MEL',name_class2='AUT')

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')

task='nevVSall'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_all(root_path+args.dataset_dir,name_class1='NEV',name_class2='AUT')

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')

task='bekVSmel'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_all(root_path+args.dataset_dir,name_class1='BEK',name_class2='MEL')

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')

task='bekVSnev'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_all(root_path+args.dataset_dir,name_class1='BEK',name_class2='NEV')

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')

task='melVSnev'
args.dataset_dir='/dataset/'+task+'/'
df_train, df_test, df_val, cls_num_list = get_data_all(root_path+args.dataset_dir,name_class1='MEL',name_class2='NEV')

df_train.to_csv(root_path+'/base/'+task+'/df_train.csv')
df_test.to_csv(root_path+'/base/'+task+'/df_test.csv')
df_val.to_csv(root_path+'/base/'+task+'/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/'+task+'/cls_num_list_pd.csv')

model = models.efficientnet_b3(pretrained=True)
model.classifier[1]=nn.Linear(1536, 3)
torch.save(model,root_path+'/model/efficientNetB3_multi.pth')

model = models.efficientnet_b3(pretrained=True)
model.classifier[1]=nn.Linear(1536, 2)
torch.save(model,root_path+'/model/efficientNetB3_bin.pth')