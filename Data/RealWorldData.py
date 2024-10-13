import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData

class RealWorldData(Dataset):
    """Matterport3D Dataset for Neural BKI project
    
    Access to the processed data, currently the predicted labels are the same as the ground truth
    """

    def __init__(self,
                grid_params,
                intrinsics,
                segmentation_encode = None,
                pca_downsample = None,
                feature_dir = 'lseg_feature',
                directory="/home/jason/kitti",
                device='cuda',
                latent_size = 512, 
                down_sample_feature = False, 
                sequences = [],
                subsample_points=0.5,
                 ):
        
        self._grid_size = grid_params['grid_size']
        self.coor_ranges = grid_params['min_bound'] + grid_params['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)
        
        # segmentation module
        self.segmentation_encode = segmentation_encode
        self.pca_downsample = pca_downsample
        
        self.feature_dir = feature_dir
        self._directory = directory
        self.device = device
        self.latent_size = latent_size
        self.down_sample_feature = down_sample_feature
        self.subsample_points = subsample_points

        # if raw:
        self._rgb_list = []
        self._depth_list = []
        self._conf_list = []
        # TODO: Change according to the output from Record3D software
        self._intrinsic_matrix = np.array(intrinsics).reshape(3,3) 
        # else:
        self._velodyne_list = []
        self._pred_list = [] # feature/categorical predictions
        self._poses = np.empty((0,7))

        self._seqs = sequences
        self._frames_list = [] # store the name of the frames
        self._num_frames_per_scene = [] # number of frames per sequence 
        self._scene_id = [] # repeat the scene id/name for the number of frames of times

        for seq in self._seqs:
            # each sequence must have rgb data
            rgb_dir = os.path.join(self._directory, seq, 'rgb')
            rgb_list = glob(os.path.join(rgb_dir, "*"))
            rgb_list.sort()
            frames_list = [os.path.basename(path).split('.')[0] for path in rgb_list] # contain all frame number of the sequence eg 000000,  000019, etc.
            self._frames_list.extend(frames_list)
            
            # sequence statistics
            self._num_frames_per_scene.append(len(frames_list))
            self._scene_id += [seq] * len(frames_list)
            
            # point cloud
            velodyne_dir = os.path.join(self._directory, seq, 'point_cloud')
            if os.path.exists(velodyne_dir):
                velodyne_list = glob(os.path.join(velodyne_dir, "*"))
                velodyne_list.sort()
                self._velodyne_list.extend(velodyne_list)
            else:
                depth_dir = os.path.join(self._directory, seq, 'depth')
                self._depth_list.extend([os.path.join(depth_dir, str(frame).zfill(6)+'.npy') for frame in frames_list]) 
            
            # depth confident
            conf_dir = os.path.join(self._directory, seq, 'conf')
            self._conf_list.extend([os.path.join(conf_dir, str(frame).zfill(6)+'.npy') for frame in frames_list]) 
            
            # per pixel feature
            preds_dir = os.path.join(self._directory, seq, self.feature_dir)
            if os.path.exists(preds_dir):
                pred_list = glob(os.path.join(preds_dir,"*"))
                pred_list.sort()
                self._pred_list.extend(pred_list)
            else:
                self._rgb_list.extend(rgb_list)
            
            pose = np.loadtxt(os.path.join(self._directory, seq, 'poses.txt'))[:(len(frames_list))] # xyz + quaternion
            self._poses = np.vstack((self._poses, pose))

    def collate_fn(self, data):
        points_batch = [bi[0] for bi in data]
        label_batch = [bi[1] for bi in data]
        gt_label_batch = [bi[2] for bi in data]
        return points_batch, label_batch, gt_label_batch

    def get_pose(self, frame_id):
        q_pose = self._poses[frame_id,:]
        xyz = q_pose[:3]
        q = q_pose[3:] # (x, y, z, w)
        
        pos_quat_vec = np.array([q[1], q[2], q[3], q[0]])
        rot = R.from_quat(pos_quat_vec.flatten()).as_matrix()
        
        pose = np.zeros((4,4))
        pose[3,3] = 1
        pose[:3,:3] = rot
        pose[:3,3] = xyz.reshape(-1)
        
        CT = np.array([[0,0,-1,0],
                        [-1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1]])
        
        pose = CT @ pose @ np.linalg.inv(CT)
        global_pose = pose.astype(np.float32)
        
        return global_pose

    def create_point_cloud_depth(self, depth, conf, intrinsics):
        fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        depth_shape = depth.shape
        [x_d, y_d] = np.meshgrid(range(0, depth_shape[1]), range(0, depth_shape[0]))
        x3 = np.divide(np.multiply((x_d-cx), depth), fx)
        y3 = np.divide(np.multiply((y_d-cy), depth), fy)
        z3 = depth

        coord =  np.stack((x3, y3, z3), axis=2)
        
        valid_depth = ~np.isnan(depth.reshape(-1,))
        conf_mask = conf.reshape(-1) >= 2
        valid_depth = np.logical_and(valid_depth, conf_mask)
        
        return coord.reshape(-1,3)[valid_depth], valid_depth

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_per_scene)

    def __getitem__(self, idx):
        return self.get_test_item(idx)
    
    def get_test_item(self, idx):
        '''
        Load one data point of everything
        pose: in global (rviz) frame x(forwad), y (left), z (up)
        point_cloud: defined in terms of camera position, but switched to rviz coordinate
        '''
        scene_name = self._scene_id[idx]
        scene_id = scene_name #int(scene_name)  # Scene ID
        frame_id = int(self._frames_list[idx])  # Frame ID in current scene ID

        pose = self.get_pose(idx)
        gt_labels = np.empty(0)
        
        if len(self._velodyne_list) != 0:
            # read ply file
            plydata = PlyData.read(self._velodyne_list[idx])
            points = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        else:
            with open(self._depth_list[idx], "rb") as f:
                depth_map = np.load(f)
            conf_map = np.load(self._conf_list[idx])
            points, valid_depth = self.create_point_cloud_depth(depth_map, conf_map, self._intrinsic_matrix)
            
        if len(self._pred_list) != 0:
            pred_labels = np.load(self._pred_list[idx]).astype(np.float32)
        else:
            img = read_image(self._rgb_list[idx]) 
            img = img[:3] # images may have alpha channel
            img = img.permute(1,2,0)
            pred_labels = self.segmentation_encode(img) 
            pred_labels = pred_labels.detach().cpu().numpy()
            
        pred_labels = pred_labels.squeeze(0).transpose(1,2,0)
        pred_labels = pred_labels.reshape(-1,pred_labels.shape[-1])
        pred_labels = pred_labels[valid_depth]
            
        # change points coordinate from camera to rviz
        # camera x (right), y (down), z (forward)
        # mp3d global pose x (right), y(up), z(back)
        # rviz x (forward), y (left), z (up)
        ### TODO: Debug purpose commet out ###
        points_x = points[:,0]
        points_y = points[:,1]
        points_z = points[:,2]
        points = np.vstack((points_z, -points_x, -points_y)).T
        
        ### filter point outside the grid ###
        grid_point_mask = np.all( (points < self.max_bound) & (points >= self.min_bound), axis=1)
        points = points[grid_point_mask]
        pred_labels = pred_labels[grid_point_mask]
        
        # filter out points that's too close to camera 
        close_point_mask = points[:,0] > 0.1 # z forward
        points = points[close_point_mask]
        pred_labels = pred_labels[close_point_mask]
        
        if self.down_sample_feature:
            pred_labels = torch.from_numpy(pred_labels)
            pred_labels = self.pca_downsample(pred_labels)
            pred_labels = pred_labels.detach().cpu().numpy().reshape(-1, self.latent_size)
        pred_labels = pred_labels.reshape(-1, self.latent_size) # dummy way

        return torch.from_numpy(pose), torch.from_numpy(points), torch.from_numpy(pred_labels), torch.from_numpy(gt_labels), scene_id, frame_id