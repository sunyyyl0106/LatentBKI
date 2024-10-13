import os
import pandas as pd
import numpy as np
import sys
import json
import time
import seaborn as sns
import clip
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    # path of data
    data_dir = '/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset'
    sequence_name = 'gTV8FGcVJC9_1'
    gt_dir = os.path.join(data_dir, f'{sequence_name}/semantic')
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

    target_dimension = 64
    if len(sys.argv) > 1:
        target_dimension = int(sys.argv[1])
        print(f"Received argument: {target_dimension}")
    else:
        print("No arguments provided")

    num_classes = 40

    pca_pred_dir = os.path.join(data_dir, f'{sequence_name}/ipca_pred_{target_dimension}')
    pca_miou_dir = os.path.join(data_dir, f'{sequence_name}/ipca_miou_{target_dimension}/miou.txt')
    #pca_pred_dir = os.path.join(data_dir, f'{sequence_name}/lseg_pred')
    #pca_miou_dir = os.path.join(data_dir, f'{sequence_name}/ipca_miou_{target_dimension}/miou.txt')
    
    iou_list = []
    for cls in range(num_classes):
        intersection = 0
        union = 0

        for i, gt_file in enumerate(gt_files):
            gd_pred = np.load(os.path.join(gt_dir, gt_file))
            root, _ = os.path.splitext(os.path.join(pca_pred_dir, gt_file))
            pca_pred = torch.load(root + '.pt')

            pca_pred = pca_pred.to('cpu').numpy()

            gt_mask = (gd_pred == cls)
            pred_mask = (pca_pred == cls)

            
            # Calculate intersection and union
            intersection += np.logical_and(gt_mask, pred_mask).sum()
            union += np.logical_or(gt_mask, pred_mask).sum()
            #print("intersections: ", intersection)
            #print("union: ", union)
            #time.sleep(2)
            
        if union == 0:
            # Avoid division by zero, consider IoU for this class as 1 if both are empty
            iou = 1 if intersection == 0 else 0
        else:
            iou = intersection / union
        iou_list.append(iou)
        print(cls)
        
    # Calculate mean IoU
    mean_iou = np.mean(iou_list)
    with open(pca_miou_dir, mode='a') as file:
        file.write(f"{mean_iou}\n")
       


if __name__ == "__main__":
    main()
