#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 11:26:46 2022

@author: arthur
"""

import argparse
from os.path import join

import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
from heatmap import demo1

root_path=os.getcwd()

root_path=root_path[:-9]
parser = argparse.ArgumentParser(description='Heatmaps')

parser.add_argument('--prev_id', type=int, help='continuous from previous id')

args = parser.parse_args()

target_layer = "features"
arch = "efficientNetB3_bin"
topk = 2
cuda = True
# isic_path = "../ISIC2018_Task3_Training_Input/"
isic_path = root_path+"/dataset/multi/TEST/"
#n_run = args.n_run
n_run='run5'
output_dir =  root_path+"/heat_maps/images/"
# output_dir = "results/"

all_scores = pd.read_csv(root_path+'/pipelines/all_predictions_run5.csv')
all_scores=all_scores.drop(columns='Unnamed: 0')


true_mel_names = all_scores[all_scores['label_idx'] == 1]['image_id'].values

#Medium confident of melanoma hesitating between MEL and BEK
les_noms = all_scores[(all_scores['M_M'] > 0.3) & (all_scores['B_M'] < 0.3) & (all_scores['M_conf'] > 0.1) & (all_scores['M_conf'] < 0.4) & (all_scores['M_BM'] > 0.7)]['image_id'].values
les_noms = [nom for nom in les_noms if nom in true_mel_names]
print(len(les_noms))


import glob
files=[n for n in glob.glob('/media/arthur/Data/PROJET/journal/heat_maps/images/medium/selected/*.jpg')]

les_noms2=[f[-16:] for f in files]

print_path = root_path+"/heat_maps/images/medium/"


image_paths = [isic_path + '/MEL/'+le_nom for le_nom in les_noms2]


print("Calcul des heatmaps...")
print('curent task:')
print('confident')

task='multi'
arch = "efficientNetB3_multi"
prev_id=args.prev_id
for i in range(41, 45):
# for i in range(len(image_paths)):
    print(f'current image: {i}')
    output_path = print_path + task + "/"
    limage = plt.imread(image_paths[i])
    plt.imsave(output_path + image_paths[i].split('/')[-1], limage)
    demo1((image_paths[i],), target_layer, arch, topk, output_path, cuda, task)

#exit(0)


