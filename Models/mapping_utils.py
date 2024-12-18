# This file contains classes for local and global offline mapping (not running semantic prediction)
import torch
import numpy as np
from tqdm import tqdm
from Models.LatentBKI import LatentBKI
from pytorch3d.ops import knn_points

# Save grid in CPU memory, load to GPU when needed for update step
class GlobalMapContinuous(LatentBKI):
    def __init__(self, grid_size, min_bound, max_bound, filter_size, ell, category_feature, 
                 decode = None, pca_upsample = None, num_classes=42, latent_dim=512, device="cpu", datatype=torch.float32, delete_time=10, use_relative_pose=True, pseduo_discrete=True):
        super().__init__(grid_size, min_bound, max_bound, max_dist=ell, filter_size=filter_size,
                 num_classes=num_classes, latent_dim=latent_dim, device=device, datatype=datatype, pseduo_discrete=pseduo_discrete)
        self.reset_grid()
        self.delete_time = delete_time
        self.category_feature = category_feature
        self.decode = decode
        self.pca_upsample = pca_upsample
        self.use_relative_pose = use_relative_pose
        self.sum_void_voxel = 0

    def reset_grid(self):
        self.global_map = None
        self.map_times = None
        self.initial_pose = None
        self.translation_discretized = np.zeros(3)
        self.points_rotation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.points_translation = torch.zeros(3, dtype=self.dtype, device=self.device)

    def inside_mask(self, min_bounds, max_bounds):
        inside = torch.all((self.global_map[:, :3] >= min_bounds) & (self.global_map[:, :3] < max_bounds), axis=1)
        return inside

    def get_local_map(self, min_bound=None, max_bound=None):
        # Fetch local map from CPU (anything not seen is prior)
        local_map = self.initialize_grid() # NOTE: N * (x, y, z, c)
        inside_mask = None
        if min_bound is None:
            min_bound = self.min_bound
        if max_bound is None:
            max_bound = self.max_bound
        local_min_bound = min_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        local_max_bound = max_bound + torch.from_numpy(self.voxel_translation).to(self.device)
        if self.global_map is not None:
            inside_mask = self.inside_mask(local_min_bound.detach().cpu(), local_max_bound.detach().cpu()) # NOTE: mask the select existing global grid within the local map boundry
            allocated_map = self.global_map[inside_mask].clone().to(device=self.device, dtype=self.dtype) # NOTE: grid thats already exsists in the global map
            grid_map = self.grid_ind(allocated_map, min_bound=local_min_bound, max_bound=local_max_bound) # TODO: why this step?
            grid_indices = grid_map[:, :3].to(torch.long) # NOTE: store grid int(x, y, z)
            local_map[0][grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, 3:self.latent_dim+3] # mean
            local_map[1][grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, self.latent_dim+3:2*self.latent_dim+3] # variance
            local_map[2][grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], :] = allocated_map[:, -1].view(-1,1) # confidence
        return local_map, local_min_bound, local_max_bound, inside_mask
    
    def camera_to_global(self, points):
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        return torch.matmul(global_pose[:3, :3], points.T).T + global_pose[:3, 3] # NOTE: global (x,y,z)
    
    def discretize_to_centroid(self, points):
        grid_inds = torch.floor(points / self.voxel_sizes) # assume globle voxel, min starts at 0
        grid_centroids = grid_inds * self.voxel_sizes + self.voxel_sizes/2
        return grid_centroids
    
    # Propagate map given a transformation matrix
    def propagate(self, pose):
        # Was just initialized
        if self.initial_pose is None:
            self.initial_pose = pose
            self.inv_rotation = np.linalg.inv(self.initial_pose[:3, :3])
            
        # we set the first frame as initial pose, assuem at origin
        # the global pose will be the relative pose to the origin
        # global_translation = self.inv_rotation @ (pose[:3, 3] - self.initial_pose[:3, 3])
        if self.use_relative_pose:
            global_translation = pose[:3, 3] - self.initial_pose[:3, 3]
            global_rotation = pose[:3, :3] @ self.inv_rotation 
        else:
            global_translation = pose[:3, 3]
            global_rotation = pose[:3, :3]
            
        self.global_pose = np.zeros((4,4), dtype=np.float32)
        self.global_pose[:3,3] = global_translation
        self.global_pose[:3,:3] = global_rotation
        self.global_pose[3,3] = 1
        
        # Relative transformation between origin and current point
        relative_translation = self.global_pose[:3, 3] 
        # To select voxels from memory, find the nearest voxel
        voxel_sizes = self.voxel_sizes.detach().cpu().numpy()
        self.voxel_translation = np.round(relative_translation / voxel_sizes) * voxel_sizes
        if self.use_relative_pose:
            self.nearest_voxel = self.initial_pose[:3, 3] + self.voxel_translation
        else:
            self.nearest_voxel = self.voxel_translation

    # Uses saved weights instead of generating a filter
    def update_map(self, semantic_preds):
        semantic_preds = semantic_preds.to(self.dtype) # NOTE: (n, 3+1) pose + semantic class, imagine number of semantic class of cube (xyz) stack on top of each other
        local_map, local_min_bound, local_max_bound, inside_mask = self.get_local_map() # NOTE: local map contain exisiting global boundary (501, 501, 27, 20)

        # Rotate the point cloud and translate to global frame
        global_pose = torch.from_numpy(self.global_pose).to(self.device)
        semantic_preds[:, :3] = torch.matmul(global_pose[:3, :3], semantic_preds[:, :3].T).T + global_pose[:3, 3] # NOTE: global (x,y,z)

        # Update local map
        with torch.no_grad():
            local_map = self.forward(local_map, semantic_preds, min_bound=local_min_bound, max_bound=local_max_bound, iterative=True)        

        # Find updated cells
        effective_cells = (local_map[2] > 0).reshape(-1,) # all the cells that has been updated in the local region (including cells updated in previous frame but not this frame)
        updated_cells = effective_cells

        updated_centroids = self.centroids[updated_cells, :] + torch.from_numpy(self.voxel_translation).to(self.device) # NOTE: change from local to global, (n, 3)
        updated_mean = local_map[0].view(-1,self.latent_dim)[updated_cells]
        updated_variance = local_map[1].view(-1,self.latent_dim)[updated_cells]
        updated_confidence = local_map[2].view(-1, 1)[updated_cells]
        new_cells = torch.cat((updated_centroids, updated_mean, updated_variance, updated_confidence), dim=1) # NOTE: only contain updated cells position and feature, (n, 3 + features)
        visited_times = torch.zeros(new_cells.shape[0], 1).detach().cpu().numpy()
        
        # If empty
        if self.global_map is None:
            self.global_map = new_cells.detach().cpu()
            self.map_times = visited_times
        else:
            # Replace local cells
            outside_mask = ~ inside_mask
            # Add new cells
            self.global_map = torch.vstack((self.global_map[outside_mask, :], new_cells.detach().cpu()))
            self.map_times = np.vstack((self.map_times[outside_mask, :], visited_times))
            
        # Garbage Collection
        # self.garbage_collection() # remove old global map depends on timestamp
        return self.global_map

    def garbage_collection(self):
        self.map_times += 1
        # Remove cells with T > self.delete_time
        recent_mask = self.map_times < self.delete_time
        recent_mask = np.squeeze(recent_mask)
        self.map_times = self.map_times[recent_mask, :]
        self.global_map = self.global_map[recent_mask, :]
    
    def label_points_iterative(self, points, batch_size = 100000, with_variance = True):
        # Preprocessing map
        if with_variance:
            predictions = torch.empty(0, self.global_map.shape[1]-3+1 ,dtype=torch.float) # features + variance + category (N, features_size)
        else:
            predictions = torch.empty(0, 1,dtype=torch.float) # category (N,)
            ### NOTE: change order of computation, for mean, compute the logits first ###
            map_feature = self.global_map[:,3:3+self.latent_dim]
            if self.pca_upsample:
                map_feature = self.pca_upsample(map_feature)
            if self.category_feature is not None:
                # mp3d
                self.global_map = torch.hstack((self.global_map[:,:3].to(self.device), self.decode(map_feature.to(self.device), self.category_feature.to(self.device)))) # (N, 3 + num_classes)
            else:
                # kitti
                self.global_map = torch.hstack((self.global_map[:,:3].to(self.device), self.decode(map_feature.to(self.device), self.category_feature))) 
                    
        self.sum_void_voxel = 0
        N = points.shape[0]
        for start in tqdm(range(0, N, batch_size)):
            end = min(start+batch_size, N)
            batch_points = points[start:end]
            batch_pred = self.label_points(batch_points, with_variance)
                
            predictions = torch.vstack((predictions, batch_pred.cpu()))
            
            # TODO: testing purpose 
            # if start > batch_size * 20:
            #     print("[TEST] Break after 20 batch")
            #     break
           # TODO: testing purpose 
        
        print("Point that does not fall in voxel:", self.sum_void_voxel/N, f"[{self.sum_void_voxel}/{N}]")
        return predictions

    def label_points(self, points, with_variance = False):
        '''
        Input:
            points: (N, 3) unlabeld points in global frame
        Output: 

        '''
        K = 1
        N = points.shape[0]
        
        ### NOTE: change order of computation, for mean, compute the logits first ###
        points = self.discretize_to_centroid(points)
        
        # find the neighbors for each of the points, (N, K, 3)
        nn_results = knn_points(points.unsqueeze(0), self.global_map[:,:3].to(self.device).unsqueeze(0), K=K, return_sorted=False)
        idx = nn_results.idx[0].detach().cpu()
        dists = torch.sqrt(nn_results.dists[0].detach().cpu()) # knn returns squared distance
        
        # points that not fall inside voxel
        far_dist_mask = (dists > self.voxel_sizes[0].item()/2).squeeze(1)
        self.sum_void_voxel += far_dist_mask.sum().item()

        if with_variance:
            nearest_labeled_pc = self.global_map[idx, :].to(self.device) # try label to be feature + variance + confidence
            # transform to postiror predictive distribution
            confidence = nearest_labeled_pc[:,:,-1].reshape(N,K,1)
            nearest_labeled_pc[:, :, 3+self.latent_dim:3+2*self.latent_dim] = (confidence + 1) / (confidence * confidence) * nearest_labeled_pc[:, :, 3+self.latent_dim:3+2*self.latent_dim] # convert variance
            nearest_labeled_pc[far_dist_mask, :, 3:3+2*self.latent_dim] = 0 # points not fall in voxel has 0 features and variance and confidence
        else:
            nearest_labeled_pc = self.global_map[idx, :].to(self.device) # return, (N, 3+40), which is categoricall logits
        
        pred_features_variance = nearest_labeled_pc[:,:,3:]
        pred_features_variance = pred_features_variance.squeeze(1)
        pred_features_variance[far_dist_mask, 3:] = 0 
            
        # return pred_features, mask
        if with_variance:
            pred_features = pred_features_variance[:,:self.latent_dim] 
        else: 
            # NOTE: if not with_variance, here is categorical logits, eg. (N, 40) 
            pred_features = pred_features_variance
            
        # backproject to clip if sample is small
        if self.pca_upsample and with_variance:
            pred_features = self.pca_upsample(pred_features)
        
        # decode features into probability for each class
        if with_variance:
            labels = self.decode(pred_features, self.category_feature)
            labels = labels.softmax(dim=-1)
        else:
            labels = pred_features.softmax(dim=-1)
        
        predictions = torch.argmax(labels, dim=-1) #(N, )
        predictions[far_dist_mask] = -1 # points that falls in the void
        
        if with_variance:
            predictions = torch.hstack((pred_features_variance, predictions.reshape(-1,1).to(torch.float))) # feature, variance, confidence, category
        else:
            predictions = predictions.reshape(-1,1)
        
        return predictions