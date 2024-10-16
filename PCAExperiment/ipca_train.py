import os
import sys
sys.path.append("..")

import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import yaml

from PCAonGPU.gpu_pca import IncrementalPCAonGPU
from ipca_utils import get_sim_mat, get_flattened_data
from dataset import HabitatDataset
from Models.Lseg.lseg_utils import init_lseg, get_lseg_feat
    
def main():
    config_file = os.path.join(os.getcwd(), "ipca.yaml")
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            target_dimension = config["target_dimension"]
            labels = config["labels"]
            h = config["image_height"]
            w = config["image_width"]
            save_dir = config["save_dir"]
            data_dir = config["mp3d_data_dir"]
        except yaml.YAMLError as exc:
            print(exc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device)

    save_dir = os.path.join(save_dir, target_dimension)
    os.makedirs(save_dir, exist_ok=True)

    ipca = IncrementalPCAonGPU(n_components=int(target_dimension))

    train_sequences = HabitatDataset(dataset_dir=data_dir, data_split="train")
    val_sequences = HabitatDataset(dataset_dir=data_dir, data_split="val")
    # init lseg model
    (lseg_model, lseg_transform, crop_size, 
     base_size, norm_mean, norm_std, clip_feat_dim) = init_lseg(device)
    
    # train
    for seq_i, sequence in enumerate(tqdm(train_sequences, desc="Training")):
        # if pca file exists
        pca_save_path = os.path.join(save_dir, f'ipca_{seq_i}.pkl')
        if os.path.exists(pca_save_path):
            print(pca_save_path, "exists, continue")
            with open(pca_save_path, 'rb') as file:
                ipca = pickle.load(file)
            print(f'load ipca_{seq_i}.pkl')
            continue

        for frame_i, frame in enumerate(tqdm(sequence, desc=sequence.seq_name)):
            pix_feats = get_lseg_feat(
                lseg_model, frame['rgb'], labels,
                lseg_transform, device, crop_size,
                base_size, norm_mean, norm_std, vis=False
            ) 
            flattened_data = get_flattened_data(pix_feats, device)
            ipca.partial_fit(flattened_data)
        
        with open(pca_save_path, 'wb') as file:
            pickle.dump(ipca, file)
            print(f'ipca_{seq_i}.pkl saved.')

    label_token = clip.tokenize(labels).to(device)
    with torch.no_grad():
        label_features = model.encode_text(label_token)

    # validation
    for seq_i, sequence in enumerate(tqdm(val_sequences, desc="Validation")):

        for frame_i, frame in enumerate(tqdm(sequence, desc=sequence.seq_name)):
            pred_dir = os.path.join(save_dir, f'{sequence.seq_name}_pred')
            os.makedirs(pred_dir, exist_ok=True)
        
            pred_path = os.path.join(pred_dir, f'{frame_i:06d}.npy')
            if os.path.exists(pred_path):
                continue

            pix_feats = get_lseg_feat(
                lseg_model, frame['rgb'], labels,
                lseg_transform, device, crop_size,
                base_size, norm_mean, norm_std, vis=False
            ) 
            flattened_data = get_flattened_data(pix_feats, device)

            pca_data = ipca.transform(flattened_data)

            # back project to 512 dimension
            bp_data = ipca.inverse_transform(pca_data)
            
            similarity_matrix = get_sim_mat(1000, bp_data, label_features)

            prediction_probs = F.softmax(similarity_matrix, dim=1)  
            predictions = torch.argmax(prediction_probs, dim=1) 
            predictions = predictions.reshape(h, w)

            # save predictions
            predictions = predictions.to('cpu').numpy()
            np.save(pred_path, predictions)

       
if __name__ == "__main__":
    main()
