#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:12:33 2022

@author: arthur
"""
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from glob import glob
import numpy as np
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df['path'][idx])
        label = torch.tensor(int(self.df['label_idx'][idx]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




def get_data_multi(dataset_dir='./dataset/isic/'):
    all_train_image_path = glob(os.path.join(dataset_dir+'TRAIN/', '*', '*.jpg'))
    all_val_image_path = glob(os.path.join(dataset_dir+'VALID/', '*', '*.jpg'))
    all_test_image_path = glob(os.path.join(dataset_dir+'TEST/', '*', '*.jpg'))
    if    len(all_train_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_train=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_train_image_path)):
        df_data_train=df_data_train.append(pd.DataFrame({'path':[all_train_image_path[i]],
                                 'image_id':[os.path.basename(all_train_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_train_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_train['label_idx'] = pd.Categorical(df_data_train['label']).codes
    train_filenames_bek=np.array(df_data_train[df_data_train.label=='BEK'].image_id)
    train_filenames_mel=np.array(df_data_train[df_data_train.label=='MEL'].image_id)
    train_filenames_nev=np.array(df_data_train[df_data_train.label=='NEV'].image_id)

        
    if    len(all_val_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_val=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_val_image_path)):
        df_data_val=df_data_val.append(pd.DataFrame({'path':[all_val_image_path[i]],
                                 'image_id':[os.path.basename(all_val_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_val_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_val['label_idx'] = pd.Categorical(df_data_val['label']).codes
    val_filenames_bek=np.array(df_data_val[df_data_val.label=='BEK'].image_id)
    val_filenames_mel=np.array(df_data_val[df_data_val.label=='MEL'].image_id)
    val_filenames_nev=np.array(df_data_val[df_data_val.label=='NEV'].image_id)

    if    len(all_test_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_test=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_test_image_path)):
        df_data_test=df_data_test.append(pd.DataFrame({'path':[all_test_image_path[i]],
                                 'image_id':[os.path.basename(all_test_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_test_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_test['label_idx'] = pd.Categorical(df_data_test['label']).codes
    test_filenames_bek=np.array(df_data_test[df_data_test.label=='BEK'].image_id)
    test_filenames_mel=np.array(df_data_test[df_data_test.label=='MEL'].image_id)
    test_filenames_nev=np.array(df_data_test[df_data_test.label=='NEV'].image_id)

    global train_filenames 
    global test_filenames 
    global val_filenames

    train_filenames=np.concatenate((train_filenames_bek,train_filenames_mel,train_filenames_nev))
    test_filenames=np.concatenate((test_filenames_bek,test_filenames_mel,test_filenames_nev))
    val_filenames=np.concatenate((val_filenames_bek,val_filenames_mel,val_filenames_nev))
    train_class_num_list=[len(train_filenames_bek),len(train_filenames_mel),len(train_filenames_nev)]
    # This function identifies if an image is part of the train or test or val set.


    return df_data_train, df_data_test, df_data_val, train_class_num_list

def get_data_all(dataset_dir='./dataset/isic/',name_class1='MEL',name_class2='AUT'):
    all_train_image_path = glob(os.path.join(dataset_dir+'TRAIN/', '*', '*.jpg'))
    all_val_image_path = glob(os.path.join(dataset_dir+'VALID/', '*', '*.jpg'))
    all_test_image_path = glob(os.path.join(dataset_dir+'TEST/', '*', '*.jpg'))
    if    len(all_train_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_train=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_train_image_path)):
        df_data_train=df_data_train.append(pd.DataFrame({'path':[all_train_image_path[i]],
                                 'image_id':[os.path.basename(all_train_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_train_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_train['label_idx'] = pd.Categorical(df_data_train['label']).codes
    train_filenames_class1=np.array(df_data_train[df_data_train.label==name_class1].image_id)
    train_filenames_class2=np.array(df_data_train[df_data_train.label==name_class2].image_id)


        
    if    len(all_val_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_val=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_val_image_path)):
        df_data_val=df_data_val.append(pd.DataFrame({'path':[all_val_image_path[i]],
                                 'image_id':[os.path.basename(all_val_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_val_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_val['label_idx'] = pd.Categorical(df_data_val['label']).codes
    val_filenames_class1=np.array(df_data_val[df_data_val.label==name_class1].image_id)
    val_filenames_class2=np.array(df_data_val[df_data_val.label==name_class2].image_id)
   

    if    len(all_test_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_test=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_test_image_path)):
        df_data_test=df_data_test.append(pd.DataFrame({'path':[all_test_image_path[i]],
                                 'image_id':[os.path.basename(all_test_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_test_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_test['label_idx'] = pd.Categorical(df_data_test['label']).codes
    test_filenames_class1=np.array(df_data_test[df_data_test.label==name_class1].image_id)
    test_filenames_class2=np.array(df_data_test[df_data_test.label==name_class2].image_id)

    global train_filenames 
    global test_filenames 
    global val_filenames

    train_filenames=np.concatenate((train_filenames_class1,train_filenames_class2))
    test_filenames=np.concatenate((test_filenames_class1,test_filenames_class2))
    val_filenames=np.concatenate((val_filenames_class1,val_filenames_class2))
    train_class_num_list=[len(train_filenames_class1),len(train_filenames_class2)]
    # This function identifies if an image is part of the train or test or val set.


    return df_data_train, df_data_test, df_data_val, train_class_num_list

def get_data_one(dataset_dir='./dataset/isic/',name_class1='MEL',name_class2='NEV'):
    all_train_image_path = glob(os.path.join(dataset_dir+'TRAIN/', '*', '*.jpg'))
    all_val_image_path = glob(os.path.join(dataset_dir+'VALID/', '*', '*.jpg'))
    all_test_image_path = glob(os.path.join(dataset_dir+'TEST/', '*', '*.jpg'))
    if    len(all_train_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_train=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_train_image_path)):
        df_data_train=df_data_train.append(pd.DataFrame({'path':[all_train_image_path[i]],
                                 'image_id':[os.path.basename(all_train_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_train_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_train['label_idx'] = pd.Categorical(df_data_train['label']).codes
    train_filenames_class1=np.array(df_data_train[df_data_train.label==name_class1].image_id)
    train_filenames_class2=np.array(df_data_train[df_data_train.label==name_class2].image_id)


        
    if    len(all_val_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_val=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_val_image_path)):
        df_data_val=df_data_val.append(pd.DataFrame({'path':[all_val_image_path[i]],
                                 'image_id':[os.path.basename(all_val_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_val_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_val['label_idx'] = pd.Categorical(df_data_val['label']).codes
    val_filenames_class1=np.array(df_data_val[df_data_val.label==name_class1].image_id)
    val_filenames_class2=np.array(df_data_val[df_data_val.label==name_class2].image_id)
   

    if    len(all_test_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data_test=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_test_image_path)):
        df_data_test=df_data_test.append(pd.DataFrame({'path':[all_test_image_path[i]],
                                 'image_id':[os.path.basename(all_test_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_test_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data_test['label_idx'] = pd.Categorical(df_data_test['label']).codes
    test_filenames_class1=np.array(df_data_test[df_data_test.label==name_class1].image_id)
    test_filenames_class2=np.array(df_data_test[df_data_test.label==name_class2].image_id)

    global train_filenames 
    global test_filenames 
    global val_filenames

    train_filenames=np.concatenate((train_filenames_class1,train_filenames_class2))
    test_filenames=np.concatenate((test_filenames_class1,test_filenames_class2))
    val_filenames=np.concatenate((val_filenames_class1,val_filenames_class2))
    train_class_num_list=[len(train_filenames_class1),len(train_filenames_class2)]
    # This function identifies if an image is part of the train or test or val set.


    return df_data_train, df_data_test, df_data_val, train_class_num_list