
import clip
import os
import torch
import yaml
import rospy
import numpy as np
from tqdm import tqdm

from Models.Lseg.Lseg_module import Lseg_module
from torch.distributions.studentT import StudentT
# from pyquaternion import Quaternion
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

class OpenQuerier():
    def __init__(self, latent_map_path, latent_size, device, pca_path, grid_params, threshold) -> None:
        self.device = device
        self.seg_module = Lseg_module(pca_path=pca_path, device=self.device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.latent_map = np.load(latent_map_path)
        
        # Flip from OpenGL coordinate to x forward coordinate
        # q_xforward = Quaternion([0.5, 0.5, -0.5, -0.5,  ])
        # self.latent_map[:,:3] = (q_xforward.rotation_matrix @ self.latent_map[:,:3].T).T
        self.latent_map[:,:3] = self.latent_map[:,:3]
        self.latent_map = torch.tensor(self.latent_map)
        
        self.latent_size = latent_size
        self.max_dim = grid_params["max_bound"]
        self.min_dim = grid_params["min_bound"]
        self.grid_dims = grid_params["grid_size"]
        self.threshold = threshold
        
        self.heatmap_publisher = rospy.Publisher("/Open_Query/Heatmap",MarkerArray, queue_size=10)
        self.uncertainty_publisher = rospy.Publisher("/Open_Query/Uncertainty",MarkerArray, queue_size=10)
    
    def sampling_for_variance(self, t_v, t_mean, t_variance, category_features, batch_size = 1000, sample_size = 30):
        N = t_v.shape[0]
        logits_variances_list = torch.empty(0,1)

        for start in tqdm(range(0, N, batch_size)):
            end = min(start+batch_size, N)
            
            distribution = StudentT(df=t_v[start:end], loc=t_mean[start:end], scale=t_variance[start:end])
            sampled_features = distribution.sample(torch.zeros(sample_size).shape).permute(1,0,2) # (B, m, latent_size)

            # decode into 40 categories
            category_logits = (self.seg_module.backproject_to_clip(sampled_features.to(self.device)) @ category_features.T).cpu() # (B, m, # of category)

            # calculate variance in logits space
            difference_square = (category_logits - category_logits.mean(dim=1, keepdim=True)).pow(2)
            logits_variance = (difference_square / (sample_size - 1)).sum(dim=1) # (B, # of category)
            # print(logits_variance.shape)
            
            if difference_square.sum(dim=1).isinf().any():
                raise
                        
            # add to list
            logits_variances_list = torch.vstack((logits_variances_list, logits_variance.reshape(-1,1)))
            
            # clean cache if needed
            torch.cuda.empty_cache()
            
        return logits_variances_list

    def sample_uncertainty(self, category_features):
        t_v = self.latent_map[:,-1].reshape(-1,1)
        wishart_variance = self.latent_map[:, 3+64:3+64*2]
        t_variance = (t_v + 1) / (t_v * t_v) * wishart_variance
        t_mean = self.latent_map[:, 3:3+64]
        # take confidence > 2, t distribution variance will only be effective when > 2, otherwise undefined
        mask = (t_v > 2).reshape(-1)
        t_variance = t_variance[mask]
        t_mean = t_mean[mask]
        t_v = t_v[mask]
        xyz = self.latent_map[:,:3][mask]
        
        logits_variances_list = self.sampling_for_variance(t_v, t_mean, t_variance, category_features, sample_size=30, batch_size=10000)
        # global_map_variance = torch.hstack((xyz, per_voxel_logits_variance.reshape(-1,1)))
        return xyz, logits_variances_list.reshape(-1,1)
    
    def heatmap_to_marker(self, xyz, score, ns):
        
        score -= torch.min(score)
        score /= torch.max(score)
        markerArray = MarkerArray()
        
        # only publish map that's greater than threshold
        # score_mask = (score > 0.8).reshape(-1,)
        # xyz = xyz[score_mask]
        # score = score[score_mask]
        
        print("Creating ros message")
        marker = Marker()
        marker.id = 2
        marker.ns = ns
        marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
        marker.type = marker.CUBE_LIST
        marker.action = marker.ADD
        marker.header.stamp = rospy.Time.now()

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1

        marker.scale.x = (self.max_dim[0] - self.min_dim[0]) / self.grid_dims[0]
        marker.scale.y = (self.max_dim[1] - self.min_dim[1]) / self.grid_dims[1]
        marker.scale.z = (self.max_dim[2] - self.min_dim[2]) / self.grid_dims[2]

        for i in range(xyz.shape[0]):
            
            point = Point32()
            color = ColorRGBA()
            point.x = xyz[i, 0]
            point.y = xyz[i, 1]
            point.z = xyz[i, 2]
            var = 2 * score[i].squeeze()
            color.r = max(0, var - 1)
            color.b = max(0, 1 - var) 
            color.g = 1 - color.r - color.b
            color.a = 1.0
            
            if ns == "Open_Query_Heatmap":
                cmap = plt.cm.get_cmap('plasma', 11) 
            else:
                cmap = plt.cm.get_cmap('viridis', 11) 
            listed_cmap = ListedColormap(cmap(np.arange(11)))
            
            var = score[i].squeeze()
            idx = int(var / 0.1)
            color.r, color.g, color.b, color.a = listed_cmap(idx)
        
            marker.points.append(point)
            marker.colors.append(color)
        
        markerArray.markers.append(marker)
        return markerArray
    
    def query(self, str, with_uncertainty = False):
        text = clip.tokenize(str).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            clip_text = text_features / text_features.norm(dim=1, keepdim=True)
            clip_text = clip_text.to(torch.float32)
        
        if self.latent_size < 512:
            latent_feature = self.seg_module.backproject_to_clip(self.latent_map[:,3:3+self.latent_size]) # (n, 512)
        else:
            latent_feature = self.latent_map[:,3:3+self.latent_size]
        
        # compute similarity
        latent_feature /= latent_feature.norm(dim=1, keepdim=True)
        score = (latent_feature @ clip_text.T).cpu() # (-1 to 1)
        print(score.min(), score.max())
        score[score <= self.threshold] = self.threshold # cut off value for better visual
        
        # publish heat map
        query_result = self.heatmap_to_marker(self.latent_map[:,:3], score, "Open_Query_Heatmap")
        print("Published heatmap!")
        self.heatmap_publisher.publish(query_result)

        if with_uncertainty:
            xyz, uncertainty = self.sample_uncertainty(clip_text)
            # crop out too high uncertatinty for visualization
            sorted_uncertainty = sorted(uncertainty)
            value = sorted_uncertainty[int(len(sorted_uncertainty) * 0.95)] # ascending order
            uncertainty[uncertainty > value] = value
            # crop out too high uncertatinty for visualization
            uncertainty_result = self.heatmap_to_marker(xyz, uncertainty.cpu(), "Open_Query_Uncertainty")
            print("Published uncertainty!")
            self.uncertainty_publisher.publish(uncertainty_result)
            
def main():
    # TODO: modify the model and path to the map you want to query
    MODEL_NAME = "LatentBKI_realworld"
    latent_map_path = "/Users/multyxu/Desktop/Programming/LatentBKI/Results/real_world/my_house_long/global_map_latent.npy"
    threshold = 0.8
    device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else 'cpu'))
    
    model_params_file = os.path.join(os.getcwd(), "Config", MODEL_NAME + ".yaml")
    with open(model_params_file, "r") as stream:
        try:
            model_params = yaml.safe_load(stream)
            dataset = model_params["dataset"]
            GRID_PARAMS = model_params["grid_params"]
        except yaml.YAMLError as exc:
            print(exc)
            
    data_params_file = os.path.join(os.getcwd(), "Config", dataset + ".yaml")
    with open(data_params_file, "r") as stream:
        try:
            data_params = yaml.safe_load(stream)
            FEATURE_SIZE = data_params["feature_size"]
            PCA_PATH = data_params['pca_path']
        except yaml.YAMLError as exc:
            print(exc)
    
    # PCA_PATH = '/Users/multyxu/Desktop/Programming/LatentBKI/Results/real_world/64_state_dict.pt' # manually set on macbook
    
    print("Init querier...")
    querier = OpenQuerier(latent_map_path, FEATURE_SIZE, device, PCA_PATH, GRID_PARAMS, threshold)
    rospy.init_node('Open_vocabulary_demo', anonymous=True)
    
    while not rospy.is_shutdown():
        word = input("What's te word you want to query? (enter 'q' to quit) ")
        if word == 'q':
            print("Ending query session...")
            break
        with_uncertainty = input("With Uncertainty? (True or False, enter 'q' to quit)")
        if with_uncertainty == "True":
            with_uncertainty = True
        else:
            with_uncertainty = False
        if with_uncertainty == 'q':
            print("Ending query session...")
            break
        print("Querying for:", word, "With uncertainty = ",with_uncertainty)
        querier.query(word, with_uncertainty)
        rospy.sleep(1)

if __name__ == '__main__':
    main()
    
    