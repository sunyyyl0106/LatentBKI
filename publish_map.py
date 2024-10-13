import rospy
import yaml
import os
import numpy as np
import torch
import roslib; roslib.load_manifest('visualization_marker_tutorials')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA

# view variance map
latent_map = np.load("/Users/multyxu/Desktop/Programming/LatentBKI/Results/5L/global_map_latent.npy")
category_map = np.load("/Users/multyxu/Desktop/Programming/LatentBKI/Results/5L/global_map.npy")

# sampled variance map
# latent_map = torch.load("/Users/multyxu/Desktop/Programming/LatentBKI/Results/5L/global_map_variance.pt").numpy()
# category_map = torch.load("/Users/multyxu/Desktop/Programming/LatentBKI/Results/5L/global_map_category.pt").numpy()

def create_marker(marker_ns, xyz, value, min_dim, max_dim, grid_dims, colors):
    '''
    xyz: (N, 3)
    value: (N, )
    '''
    marker = Marker()
    marker.id = 1
    marker.ns = "Global_Semantic_Map" #
    marker.ns = marker_ns
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

    for i in range(xyz.shape[0]):
        point = Point32()
        color = ColorRGBA()
        point.x = xyz[i, 0]
        point.y = xyz[i, 1]
        point.z = xyz[i, 2]
        color.r, color.g, color.b, color.a = colors(value[i])

        marker.points.append(point)
        marker.colors.append(color)
    
    return marker

def create_semantic_marker(global_map, num_class, min_dim, max_dim, grid_dims):
    '''
    global_map: (N, 3+1+1)
    '''
    markerArray = MarkerArray()
    
    cmap = plt.cm.get_cmap('jet', num_class) 
    listed_cmap = ListedColormap(cmap(np.arange(num_class)))
    
    print("Creating marker for semantic with map shape of", global_map.shape)
    semantic_labels = global_map[:,3].astype(np.int32).reshape(-1,)
    centroids = global_map[:, :3]
    
    marker = create_marker("Global_Semantic_Map", centroids, semantic_labels, min_dim, max_dim, grid_dims, listed_cmap)
    markerArray.markers.append(marker)
    
    return markerArray  

def create_variance_marker(global_map, min_dim, max_dim, grid_dims):
    markerArray = MarkerArray()
    
    cmap = plt.cm.get_cmap('viridis', 11) 
    listed_cmap = ListedColormap(cmap(np.arange(11)))
    
    print("Creating marker for variance with map shape of", global_map.shape)
    
    xyz = global_map[:,:3]
    variance = global_map[:, 3]
    variance -= np.min(variance)
    variance /= np.max(variance) # make it between 0-1
    variance = (variance / 0.1).astype(np.int32)
    
    marker = create_marker("Global_Variance_Map", xyz, variance, min_dim, max_dim, grid_dims, listed_cmap)
    markerArray.markers.append(marker)
    
    return markerArray

############## main script ################

# TODO: change these
MODEL_NAME = "LatentBKI_default"
SAMPLEED = True # view sampled variance map
D_OP = True # view variance in latent space
MAP_DIR = "/Users/multyxu/Desktop/Programming/LatentBKI/Results/gT"
# TODO: change these

publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
rospy.init_node('register')

# load map from memory
if SAMPLEED:
    # TODO: need to preprocess the variance map
    latent_map = torch.load(os.path.join(MAP_DIR,"global_map_variance.pt")).numpy()
    category_map = torch.load(os.path.join(MAP_DIR,"global_map_category.pt")).numpy()
else:
    latent_map = np.load(os.path.join(MAP_DIR,"global_map_latent.npy"))
    category_map = np.load(os.path.join(MAP_DIR,"global_map.npy"))

model_params_file = os.path.join(os.getcwd(), "Config", MODEL_NAME + ".yaml")
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
        DATASET = model_params["dataset"]
        GRID_PARAMS = model_params["grid_params"]
        MIN_BOUND = np.array(GRID_PARAMS["min_bound"])
        MAX_BOUND = np.array(GRID_PARAMS["max_bound"])
        GRID_SIZE = np.array(GRID_PARAMS["grid_size"])
    except yaml.YAMLError as exc:
        print(exc)
        
data_params_file = os.path.join(os.getcwd(), "Config", DATASET + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
        FEATURE_SIZE = data_params["feature_size"]
        PCA_PATH = data_params['pca_path']
        NUM_CLASS = data_params['num_classes']
    except yaml.YAMLError as exc:
        print(exc)

# mask out ceiling
if DATASET == "mp3d":
    mask = category_map[:,3] != 17 # in MP3D, 17 is ceiling
    latent_map = latent_map[mask]
    category_map = category_map[mask]

# process variance
if SAMPLEED:
    xyz = latent_map[:,:3]
    variance = latent_map[:,4].reshape(-1,1)
else:
    # convert into Multivariate Student-t Distribution per voxel
    pred_confidence = latent_map[:,-1].reshape(-1,1)
    pred_variance = latent_map[:,3+64: 3+64*2]
    # mean remain the same
    t_variance = (pred_confidence + 1) / (pred_confidence * pred_confidence) * pred_variance
    t_mean = latent_map[:, 3:3+64]
    t_v = pred_confidence
    
    confidence_mask = (t_v > 2).reshape(-1) # variance is effective when v > 2?
    t_variance = t_variance[confidence_mask]
    t_mean = t_mean[confidence_mask]
    t_v = t_v[confidence_mask]
    t_xyz = latent_map[:,:3][confidence_mask]
    
    if D_OP:
        t_variance_norm = np.sum(np.log(t_variance), axis=-1, keepdims=True) / t_variance.shape[-1]
        t_variance_norm = np.exp(t_variance_norm)
    else:
        # E-OP
        t_variance_norm = np.max(np.abs(t_variance), axis=-1, keepdims=True)
    
    xyz = t_xyz
    variance = t_variance_norm
    
print(variance.max(), variance.min(), variance.mean())

# cut off points for better visuals
sorted_uncertainty = sorted(variance)
value = sorted_uncertainty[int(len(sorted_uncertainty) * 0.95)] # ascending order
variance[variance > value] = value
variance_map = np.hstack((xyz, variance))

semantic_marker = create_semantic_marker(category_map, NUM_CLASS, MIN_BOUND, MAX_BOUND, GRID_SIZE)
variance_marker = create_variance_marker(variance_map, MIN_BOUND, MAX_BOUND, GRID_SIZE)

while not rospy.is_shutdown():
    publisher.publish(semantic_marker)
    publisher.publish(variance_marker)
    rospy.sleep(3)