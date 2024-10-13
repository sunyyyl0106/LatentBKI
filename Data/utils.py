import os
import pdb
from matplotlib import markers
# import rospy # comment out for initial trial becuase no ros in container
import numpy as np
import time
import os
import pdb
import torch
# comment out for initial trial becuase no ros in container
# from visualization_msgs.msg import *
# from geometry_msgs.msg import Point32
# from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R


def get_sim_cam_mat_with_fov(h, w, fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat

def depth2pc(depth, fov=90, intr_mat=None, min_depth=0.1, max_depth=10):
    """
    Return Nx3 array and the mask of valid points in [min_depth, max_depth].
    """

    h, w = depth.shape

    cam_mat = intr_mat
    if intr_mat is None:
        cam_mat = get_sim_cam_mat_with_fov(h, w, fov)
    # cam_mat[:2, 2] = 0
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = x.reshape((1, -1))[:, :] + 0.5
    y = y.reshape((1, -1))[:, :] + 0.5
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask = pc[2, :] > min_depth

    mask = np.logical_and(mask, pc[2, :] < max_depth)
    # pc = pc[:, mask]
    pc = pc.T.astype(np.float32)
    mask = mask.T
    return pc, mask
    
# def qvec2rotmat(qvec):
#     # input (x, y, z, w)
#     w = qvec[3]
#     qvec[1:] = qvec[:3]
#     qvec[0] = w
#     # (w, x, y, z)
#     return np.array(
#         [
#             [
#                 1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
#                 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
#                 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
#             ],
#             [
#                 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
#                 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
#                 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
#             ],
#             [
#                 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
#                 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
#                 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
#             ],
#         ]
#     )
        
def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=21):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection[cls] = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union[cls] = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection[cls]
    return intersection, union

def points_to_voxels_torch(voxel_grid, points, min_bound, grid_dims, voxel_sizes):
    voxels = torch.floor((points - min_bound) / voxel_sizes).to(dtype=torch.int)
    # Clamp to account for any floating point errors
    maxes = (grid_dims - 1).reshape(1, 3)
    mins = torch.zeros_like(maxes)
    voxels = torch.clip(voxels, mins, maxes).to(dtype=torch.long)

    voxel_grid = voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return voxel_grid


# Remap colors to np array 0 to 1
def remap_colors(colors):
    # color
    colors_temp = np.zeros((len(colors), 3))
    for i in range(len(colors)):
        colors_temp[i, :] = colors[i]
    colors = colors_temp.astype("int")
    colors = colors / 255.0
    return colors


def publish_voxels(map_object, min_dim, max_dim, grid_dims, colors, next_map):
    next_map.markers.clear()
    marker = Marker()
    marker.id = 0
    marker.ns = "Global_Semantic_Map"
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

    semantic_labels = map_object.global_map[:,3:]
    centroids = map_object.global_map[:, :3]

    # Threshold here
    total_probs = np.sum(semantic_labels, axis=-1, keepdims=False)
    not_prior = total_probs > 1
    semantic_labels = semantic_labels[not_prior, :]
    centroids = centroids[not_prior, :]

    semantic_labels = np.argmax(semantic_labels, axis=-1)
    semantic_labels = semantic_labels.reshape(-1, 1)

    for i in range(semantic_labels.shape[0]):
        pred = semantic_labels[i]
        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]
        color.r, color.g, color.b = colors[pred].squeeze()

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)

    next_map.markers.append(marker)
    return next_map


def publish_local_map(labeled_grid, centroids, grid_params, colors, next_map):
    max_dim = grid_params["max_bound"]
    min_dim = grid_params["min_bound"]
    grid_dims = grid_params["grid_size"]

    next_map.markers.clear()
    marker = Marker()
    marker.id = 0
    marker.ns = "Local Semantic Map"
    marker.header.frame_id = "map"
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

    X, Y, Z, C = labeled_grid.shape
    semantic_labels = labeled_grid.view(-1, C).detach().cpu().numpy()
    centroids = centroids.detach().cpu().numpy()

    semantic_sums = np.sum(semantic_labels, axis=-1, keepdims=False)
    valid_mask = semantic_sums >= 1

    semantic_labels = semantic_labels[valid_mask, :]
    centroids = centroids[valid_mask, :]

    semantic_labels = np.argmax(semantic_labels / np.sum(semantic_labels, axis=-1, keepdims=True), axis=-1)
    semantic_labels = semantic_labels.reshape(-1, 1)

    for i in range(semantic_labels.shape[0]):
        pred = semantic_labels[i]
        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]
        color.r, color.g, color.b = colors[pred].squeeze()

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)

    next_map.markers.append(marker)
    return next_map

####################################################################

def base_pos2grid_id_3d(x_base, y_base, z_base):
    gs = 1000
    cs = 0.05
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    h = int(z_base / cs)
    return [row, col, h]

def out_of_range(row: int, col: int, height: int) -> bool:
        gs = 1000
        camera_height = 1.5
        cs = 0.05
        vh = int(camera_height / cs)
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0


def transform_pc(pc, pose):
    """
    pose: the pose of the camera coordinate where the pc is in
    pc: (3, N)
    """
    # pose_inv = np.linalg.inv(pose)

    pc_homo = np.vstack([pc, np.ones((1, pc.shape[1]))])

    pc_global_homo = pose @ pc_homo

    return pc_global_homo[:3, :]



def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf


def get_pc_transform(base_posevec, base_pose0):
    camera_height = 1.5
    base2cam_rot = [1, 0, 0, 0, -1, 0, 0, 0, -1]
    base_forward_axis = [0, 0, -1]
    base_left_axis = [-1, 0, 0]
    base_up_axis = [0, 1, 0]
    
    base2cam_tf = np.eye(4)
    base2cam_tf[:3, :3] = np.array([base2cam_rot]).reshape((3, 3))
    base2cam_tf[1, 3] = camera_height
    # transform the base coordinate such that x is forward, y is leftward, z is upward
    base_transform = np.eye(4)
    base_transform[0, :3] = base_forward_axis
    base_transform[1, :3] = base_left_axis
    base_transform[2, :3] = base_up_axis

    init_base_tf = (
            base_transform @ cvt_pose_vec2tf(base_pose0) @ np.linalg.inv(base_transform)
    )
    inv_init_base_tf = np.linalg.inv(init_base_tf)
    habitat_base_pose = cvt_pose_vec2tf(base_posevec)
    base_pose = base_transform @ habitat_base_pose @ np.linalg.inv(base_transform)
    tf = inv_init_base_tf @ base_pose
    pc_transform = tf @ base_transform @ base2cam_tf
    return pc_transform


####################################################################
