#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:57:23 2022

@author: arthur
"""

import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_recall_curve, \
    PrecisionRecallDisplay, roc_curve, auc

import matplotlib.pyplot as plt

def plot_average_curve(y_test, predictions, color, name, alpha=1):
    all_fpr = []
    all_tpr = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)
    for pred in predictions:
        # fpr, tpr, thresh = roc_curve(y_test[:, 1], pred[:, 1])
        fpr, tpr, thresh = roc_curve(y_test, pred)
        aucs.append(auc(fpr, tpr))
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        all_fpr.append(fpr)
        all_tpr.append(tpr)
    tprs = np.array(all_tpr)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, color, label=f'{name} area={np.mean(aucs):.2f}')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=color, alpha=alpha / 2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()


def get_mean_auc(y_test, predictions):
    aucs = []
    for pred in predictions:
        fpr, tpr, thresh = roc_curve(y_test, pred)
        aucs.append(auc(fpr, tpr))
    return np.mean(aucs)


def get_ecart_auc(y_test, predictions):
    aucs = []
    for pred in predictions:
        fpr, tpr, thresh = roc_curve(y_test, pred)
        aucs.append(auc(fpr, tpr))
    return np.std(aucs)


if __name__ == "__main__":

    #Indicate the best value obtained for alpha1 and alpha2 
    best_a1=0.1
    best_a2=0.5


    root_path=os.getcwd()

    csv_path = "pipelines_predictions_run0_u1_"+str(best_a1)+"_u2_"+str(best_a2)+".csv"
    # csv_path = "D:\Documents\ISIC2018_Task3_Training_GroundTruth.csv"

    df_data = pd.read_csv(root_path+'/pipelines/'+csv_path)
    base_dir = "../final/"

    all_preds = []
    all_bacc = []
    y_test2=np.array(df_data.label_idx)
    y_test= F.one_hot(torch.from_numpy(y_test2), num_classes=3).cpu().numpy()
    for run in range(10):
        csv_path = "pipelines_predictions_run"+str(run)+"_u1_"+str(best_a1)+"_u2_"+str(best_a2)+".csv"
        df_data = pd.read_csv(root_path+'/pipelines/'+csv_path)
        final_score=np.array(df_data[['BEK_sco','MEL_sco','NEV_sco']])
        all_preds.append(final_score)

    print(f"\n\nUtilisation du seuil trouvé sur la base de validation [alpha1,alpha2] : {[best_a1,best_a2]}\n")

   
    bacc = [balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)) for pred in all_preds]
    print("BACC")
    print("Mean = " + str(np.mean(bacc)))
    print("std = " + str(np.std(bacc)))

    y_test = np.asarray(y_test).astype(float)
    all_preds = np.asarray(all_preds).astype(float)
    colors = ('green', 'red', 'blue')
    labels = ('BEK', 'MEL', 'NEV')
    les_aucs = []
    les_ecart_auc = []
    for i in range(3):
        plot_average_curve(y_test[:, i], all_preds[:, :, i], color=colors[i], name=labels[i], alpha=0.5)
        les_aucs.append(get_mean_auc(y_test[:, i], all_preds[:, :, i]))
        les_ecart_auc.append(get_ecart_auc(y_test[:, i], all_preds[:, :, i]))

    plt.plot([], [], ' ', label=f"Mean AUC = {np.mean(les_aucs):.2f}")
    plt.legend()
    plt.show()
    print(np.mean(les_aucs))
    print(np.mean(les_ecart_auc))
    print(les_ecart_auc)