import torch
import clip
import pickle as pk
import os

from Models.Lseg.lseg_utils import get_lseg_feat, init_lseg
from PCAonGPU.gpu_pca.pca_module import IncrementalPCAonGPU

class Lseg_module():
    def __init__(self, pca_path = None, device=("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        (self.lseg_model, self.lseg_transform, 
         self.crop_size, self.base_size, self.norm_mean, 
         self.norm_std, self.clip_feat_dim) = init_lseg(self.device)
        self.pca = None 
        if pca_path is not None:
            if os.path.basename(pca_path).split('.')[-1] == 'pkl':
                self.pca = pk.load(open(pca_path,'rb'))
            elif os.path.basename(pca_path).split('.')[-1] == 'pt':
                self.pca = IncrementalPCAonGPU(device=self.device)
                self.pca.load_vars(pca_path)
            
        
    def encoding_feature(self, rgb : torch.Tensor) -> torch.Tensor:
        '''
        Input: 
            rgb: image torch tensor
        Output:
            features: per pixel features in the same shape
        '''
        labels = ["example"]
        pix_feats = get_lseg_feat(
            self.lseg_model, rgb.to('cpu').numpy(), 
            labels, self.lseg_transform, self.device, 
            self.crop_size, rgb.shape[1], self.norm_mean, 
            self.norm_std, vis=False
        )
        pix_feats = torch.tensor(pix_feats).to(self.device)
        return pix_feats
    
    def decoding_feature(self, features : torch.Tensor, category_features : torch.Tensor) -> torch.Tensor:
        '''
        Input:
            features: (N, C), features of N elements
            Category_features: (M, C), M is the number of categories you have
        Output:
            semantic_probs: (N, C), category probability for each element
        '''
        similarity_matrix = (features / features.norm(dim=-1,keepdim=True)) @ (category_features / (category_features.norm(dim=-1,keepdim=True))).T
        # similarity_matrix = features @ category_features.T
        # similarity_matrix = torch.nn.functional.cosine_similarity(features.unsqueeze(1), category_features,)
        # print(similarity_matrix.shape)
        # semantic_probs = similarity_matrix.softmax(dim=-1) # convert to probability
        return similarity_matrix # TODO: return category instead?
        
    def words_to_clip(self, word_list) -> torch.Tensor:
        text = clip.tokenize(word_list).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=1, keepdim=True)
        return text_features
    
    def down_sampling(self, features) -> torch.Tensor:
        '''
        Input:
            features: (N, C), features of N elements with C dimensions of the feature vector
            pca: predefined IncrementalPCAonGPU(n_components=D) object 
        Output:
            tf_features: (N, D), features of N elements with D dimensions of the feature vector
        '''
        return self.pca.transform(features)
    
    def backproject_to_clip(self, features) -> torch.Tensor:
        '''
        Input:
            features: (N, D), features of N elements with D dimensions of the feature vector
            pca: predefined IncrementalPCAonGPU(n_components=D) object 
        Output:
            tf_features: (N, C), features of N elements with C dimensions of the feature vector
        '''
        return self.pca.inverse_transform(features)

        