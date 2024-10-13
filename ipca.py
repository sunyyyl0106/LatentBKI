import os
import sys
import pandas as pd
import numpy as np
import json
import time
import seaborn as sns 
import clip
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


from PCAonGPU.gpu_pca import IncrementalPCAonGPU


def mean_iou(ground_truth, prediction, num_classes):
    iou_list = []
    
    for cls in range(num_classes):
        # Create binary masks for the current class
        gt_mask = (ground_truth == cls)
        pred_mask = (prediction == cls)
        
        # Calculate intersection and union
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        
        if union == 0:
            # Avoid division by zero, consider IoU for this class as 1 if both are empty
            iou = 1 if intersection == 0 else 0
        else:
            iou = intersection / union
        
        iou_list.append(iou)
    
    # Calculate mean IoU
    mean_iou = np.mean(iou_list)
    return mean_iou

def write_miou(ground_truth, prediction, num_classes, file_path):
    with open(file_path, mode='a') as file:
        miou = mean_iou(ground_truth, prediction, num_classes)
        file.write(f"{miou}\n")



def get_sim_mat(batch_size, bp_data, label_features):
    num_batches = (bp_data.size(0) + batch_size - 1) // batch_size
    similarity_matrices = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, bp_data.size(0))
        batch_bp_data = bp_data[start_idx:end_idx]
        similarity_matrix = F.cosine_similarity(batch_bp_data.unsqueeze(1), label_features.unsqueeze(0), dim=2)
        similarity_matrices.append(similarity_matrix)

    similarity_matrix = torch.cat(similarity_matrices, dim=0)
    print("similarity_matrix.shape: ", similarity_matrix.shape)
    return similarity_matrix


def get_flattened_data(lseg_dir, npy_file, device):
    data = np.load(os.path.join(lseg_dir, npy_file))
    print("data.shape: ", data.shape)
    
    data = data.squeeze(0)
    print("data.shape: ", data.shape)

    flattened_data = data.reshape(data.shape[0], -1)
    print("flattened_data.shape: ", flattened_data.shape)

    flattened_data = flattened_data.T
    print("flattened_data.shape: ", flattened_data.shape)

    flattened_data = torch.tensor(flattened_data).to(device)

    return flattened_data
    
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print("device: ", device)
    model, _ = clip.load("ViT-B/32", device)


    # initialize ipca
    target_dimension = 128
    if len(sys.argv) > 1:
        target_dimension = int(sys.argv[1])
        print(f"Received argument: {target_dimension}")
    else:
        print("No arguments provided")
    ipca = IncrementalPCAonGPU(n_components=target_dimension)

    # path of data
    data_dir = '/workspace/sdh1/vlmaps_data_dir/vlmaps_dataset'
    data_dir2 = '/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset'
    sequence_name = 'gTV8FGcVJC9_1'
    #sequence_dir = os.path.join(data_dir, sequence_name)
    lseg_dir = os.path.join(data_dir2, f'{sequence_name}/lseg_feature')
    #pred_dir = os.path.join(data_dir, f'{sequence_name}/lseg_pred')
    npy_files = sorted([f for f in os.listdir(lseg_dir) if f.endswith('.npy')])

    pca_feature_dir = os.path.join(data_dir, f'{sequence_name}/ipca_feature_{target_dimension}')
    pca_pred_dir = os.path.join(data_dir, f'{sequence_name}/ipca_pred_{target_dimension}')
    pca_miou_dir = os.path.join(data_dir, f'{sequence_name}/ipca_miou_{target_dimension}')
    pca_save_dir = os.path.join(data_dir, f'{sequence_name}/ipca_{target_dimension}')
    pca_save_path = os.path.join(pca_save_dir, 'ipca.pkl')
    os.makedirs(pca_feature_dir, exist_ok=True)
    os.makedirs(pca_pred_dir, exist_ok=True)
    os.makedirs(pca_miou_dir, exist_ok=True)
    os.makedirs(pca_save_dir, exist_ok=True)
    # initialize labels
    labels = ["void","wall","floor","chair",
            "door","table","picture","cabinet",
            "cushion","window","sofa","bed",
            "curtain","chest_of_drawers","plant",
            "sink","stairs","ceiling","toilet",
            "stool","towel","mirror","tv_monitor",
            "shower","column","bathtub","counter",
            "fireplace","lighting","beam","railing",
            "shelving","blinds","gym_equipment",
            "seating","board_panel","furniture",
            "appliances","clothes","objects",]
    print("len(labels): ", len(labels))
    label_token = clip.tokenize(labels).to(device)
    print("label_token.shape: ", label_token.shape)
    with torch.no_grad():
        label_features = model.encode_text(label_token)
        print("label_features.shape: ", label_features.shape)
    h = 720
    w = 1080
    
    for i, npy_file in enumerate(npy_files):

        print(i, " start")
        flattened_data = get_flattened_data(lseg_dir, npy_file, device)
        ipca.partial_fit(flattened_data)
        print("partial fit ", i)

    with open(pca_save_path, 'wb') as file:
            pickle.dump(ipca, file)
            print("IPCA model saved.")

    for i, npy_file in enumerate(npy_files):

        feature_path = os.path.join(pca_feature_dir, f'{i:06d}.pt')
        pred_path = os.path.join(pca_pred_dir, f'{i:06d}.pt')
    
        print(i, "start")
        flattened_data = get_flattened_data(lseg_dir, npy_file, device)
        pca_data = ipca.transform(flattened_data)

        # back project to 512 dimension
        bp_data = ipca.inverse_transform(pca_data)
        print('bp_data.shape: ', bp_data.shape)

        # save pca features
        pca_data = pca_data.transpose(0, 1)
        print('pca_data.shape: ', pca_data.shape)
        pca_data = pca_data.reshape(pca_data.shape[0], h, w)
        print('pca_data.shape: ', pca_data.shape)
        pca_data = pca_data.unsqueeze(0)
        print('pca_data.shape: ', pca_data.shape)
        if not os.path.exists(feature_path):
            torch.save(pca_data, feature_path)
            print("save pca_data")
        
        similarity_matrix = get_sim_mat(1000, bp_data, label_features)

        print("similarity_matrix.shape: ", similarity_matrix.shape)
        prediction_probs = F.softmax(similarity_matrix, dim=1)  
        print("prediction_probs.shape: ", prediction_probs.shape)
        predictions = torch.argmax(prediction_probs, dim=1) 
        print("predictions.shape: ", predictions.shape)
        predictions = predictions.reshape(h, w)
        print("predictions.shape: ", predictions.shape)

        # save pca predictions
        if not os.path.exists(pred_path):
            torch.save(predictions, pred_path)
            print("save predictions")
        
        print(i, "finished")

    
if __name__ == "__main__":
    main()
