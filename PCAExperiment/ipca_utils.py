import os
import torch
import numpy as np
import torch.nn.functional as F

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
    return similarity_matrix


def get_flattened_data(pix_feats, device):
    data = pix_feats
    data = data.squeeze(0)
    flattened_data = data.reshape(data.shape[0], -1)
    flattened_data = flattened_data.T
    flattened_data = torch.tensor(flattened_data).to(device)
    return flattened_data


def get_miou(num_classes, val_seqs, data_dir, save_dir, target_dimension):
    iou_list = []
    for cls in range(num_classes):
        intersection = 0
        union = 0

        for seq in val_seqs:

            gt_dir = os.path.join(data_dir, f'{seq}/semantic')
            gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

            for i, gt_file in enumerate(gt_files):

                gt_pred = np.load(os.path.join(gt_dir, gt_file))

                pca_pred = np.load(os.path.join(save_dir, target_dimension, 
                                                f'{seq}_pred', gt_file))
                gt_mask = (gt_pred == cls)
                pred_mask = (pca_pred == cls)

                # Calculate intersection and union
                intersection += np.logical_and(gt_mask, pred_mask).sum()
                union += np.logical_or(gt_mask, pred_mask).sum()
    
        iou = intersection / (union + +1e-6)

        iou_list.append(iou)
        print("class: ", cls)
        
    # Calculate mean IoU
    return np.mean(iou_list)

def get_accuracy(val_seqs, data_dir, save_dir, target_dimension):
    correct_predictions = 0
    total_predictions = 0
    for seq in val_seqs:

        gt_dir = os.path.join(data_dir, f'{seq}/semantic')
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

        for i, gt_file in enumerate(gt_files):

            gt_pred = np.load(os.path.join(gt_dir, gt_file))

            pca_pred = np.load(os.path.join(save_dir, target_dimension, 
                                            f'{seq}_pred', gt_file))

            correct_predictions += (gt_pred == pca_pred).sum()
            total_predictions += gt_pred.size
        
    return correct_predictions / total_predictions






