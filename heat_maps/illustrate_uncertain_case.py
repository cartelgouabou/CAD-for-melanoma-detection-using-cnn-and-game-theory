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


#Uncertain Melanoma
les_noms = all_scores[(all_scores['B_M'] < 0.5) & (all_scores['M_M'] < 0.5) & (all_scores['N_M'] < 0.5)  ]['image_id'].values
les_noms = [nom for nom in les_noms if nom in true_mel_names]
print(len(les_noms))

print_path = root_path+"/heat_maps/images/uncertain/"


image_paths = [isic_path + '/MEL/'+le_nom for le_nom in les_noms]

#Generate and save score for sampled images
for nom in les_noms:
    pred_scores_sample=all_scores[all_scores.image_id==nom]
    pred_scores_sample.to_csv(print_path+f'pred_scores_for_img_{nom}.csv')




print("Calcul des heatmaps...")
print('curent task:')
print('uncertain')

task='multi'
arch = "efficientNetB3_multi"
prev_id=args.prev_id
for i in range(0, 5):
# for i in range(len(image_paths)):
    print(f'current image: {i}')
    output_path = print_path + task + "/"
    limage = plt.imread(image_paths[i])
    plt.imsave(output_path + image_paths[i].split('/')[-1], limage)
    demo1((image_paths[i],), target_layer, arch, topk, output_path, cuda, task)

#exit(0)


#binary tasks


print("Calcul des heatmaps...")

print('curent task:')
les_modeles=('bekVSmel','bekVSnev','melVSnev')
prev_id=args.prev_id
for i in range(2, 4):
#for i in range(len(image_paths)):
    print(f'current image: {i}')
    for task in les_modeles:
        output_path = print_path + task + "/"
        limage = plt.imread(image_paths[i])
        plt.imsave(output_path + image_paths[i].split('/')[-1], limage)
        demo1((image_paths[i],), target_layer, arch, topk, output_path, cuda, task)

