import torch
import importlib
import numpy as np

from Data.KITTI_SPVCNN import KITTI_SPVCNN_config

class SPVCNN_Module():
    def __init__(self, device=("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.config = KITTI_SPVCNN_config()
        self.config = self.config.config
        
        model_file = importlib.import_module('TwoDPASS.network.' + self.config['model_params']['model_architecture'])
        my_model = model_file.get_model(self.config) ######## get model ############
        my_model = my_model.load_from_checkpoint(self.config.checkpoint, config=self.config, strict=(not self.config.pretrain2d))
        my_model = my_model.eval()
        
        self.my_model = my_model.to(self.device)
            
        
    def encoding_feature(self, data_dict) -> torch.Tensor:
        '''
        Input: 
            rgb: image torch tensor
        Output:
            features: per pixel features in the same shape
        '''
        with torch.no_grad():
            features = self.my_model.encode_points(data_dict, self.device)
        return features
    
    def decoding_feature(self, features, category_features = None) -> torch.Tensor:
        '''
        Input:
            features: (N, C), features of N elements
            Category_features: (M, C), M is the number of categories you have
        Output:
            semantic_probs: (N, C), category logits for each element
        '''
        with torch.no_grad():
            logits = self.my_model.decode_points(features, self.device)
        return logits # TODO: return category instead?

        