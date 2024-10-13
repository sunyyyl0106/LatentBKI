import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from scipy.spatial.transform import Rotation as R
from Data.utils import depth2pc

class MP3D(Dataset):
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
                num_classes = 42,
                latent_size = 512, 
                down_sample_feature = False, 
                sequences = [],
                subsample_points=0.5,
                grid_mask=True,
                raw=False, # wether to sotre data in [rgb, depth] or [point cloud, feagures] 
                 ):
        if raw and segmentation_encode == None:
            raise ValueError("If want to load raw rgb data, must specify the segmentation ecnoding funciton")
        
        self.raw = raw
        self.grid_mask = grid_mask

        self._grid_size = grid_params['grid_size']
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        self.coor_ranges = grid_params['min_bound'] + grid_params['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)
        
        self.segmentation_encode = segmentation_encode
        self.pca_downsample = pca_downsample
        
        self.feature_dir = feature_dir
        self._directory = directory
        self.device = device
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.down_sample_feature = down_sample_feature
        self.sequences = sequences
        self.subsample_points = subsample_points

        if raw:
            self._rgb_list = []
            self._depth_list = []
            self._intrinsic_matrix = np.array(intrinsics).reshape(3,3)
        else:
            self._velodyne_list = []
            self._pred_list = [] # feature/categorical predictions
        
        self._label_list = [] # categorical semantic labels
        self._poses = np.empty((0,7))
        
        self._seqs = self.sequences
        self._frames_list = [] # store the name of the frames
        self._num_frames_per_scene = [] # number of frames per sequence 
        self._scene_id = [] # repeat the scene id/name for the number of frames of times
        self._base_pose0_list = [] # repeat the first base pose of each scene for the number of frames of times

        for seq in self._seqs:
            label_dir = os.path.join(self._directory, seq, 'semantic')
            label_path_list = sorted(os.listdir(label_dir)) # contain all file path to the frame labels
            ##### debug purpose #####
            # label_path_list = label_path_list[50:60] # only take the first ten sequence
            ##### debug purpose #####
            frames_list = [os.path.splitext(filename)[0] for filename in label_path_list] # contain all frame number of the sequence eg 000000,  000019, etc.
            self._frames_list.extend(frames_list)
            
            # sequence statistics
            self._num_frames_per_scene.append(len(frames_list))
            self._scene_id += [seq] * len(frames_list)
            
            # categorical label for each point/pixel
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.npy') for frame in frames_list])
            
            if raw:
                rgb_dir = os.path.join(self._directory, seq, 'rgb')
                depth_dir = os.path.join(self._directory, seq, 'depth')
                self._rgb_list.extend([os.path.join(rgb_dir, str(frame).zfill(6)+'.png') for frame in frames_list])
                self._depth_list.extend([os.path.join(depth_dir, str(frame).zfill(6)+'.npy') for frame in frames_list]) 
            else:
                velodyne_dir = os.path.join(self._directory, seq, 'point_cloud')
                preds_dir = os.path.join(self._directory, seq, self.feature_dir)
                self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.npy') for frame in frames_list])
                self._pred_list.extend([os.path.join(preds_dir, str(frame).zfill(6)+'.npy') for frame in frames_list]) 
            
            pose = np.loadtxt(os.path.join(self._directory, seq, 'poses.txt'))[:(len(frames_list))] # xyz + quaternion
            self._base_pose0_list += [pose[0]] * len(frames_list)
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
        
        ### coordinate transformation matrix, openGL to x forward, z upward ###
        pose = np.zeros((4, 4))
        pos_quat_vec = np.array([q[0], q[1], q[2], q[3]])
        pose[:3, :3] = R.from_quat(pos_quat_vec.flatten()).as_matrix()
        pose[:3, 3] = xyz.reshape(1,3)
        pose[3, 3] = 1
        
        CT = np.array([[0,0,-1,0],
                        [-1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1]])
        
        pose = CT @ pose @ np.linalg.inv(CT)
        global_pose = pose.astype(np.float32)
        ### coordinate transformation matrix, openGL to x forward, z upward ###
        
        return global_pose

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_per_scene)

    def __getitem__(self, idx):
        self.get_test_item(idx)

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
        gt_labels = np.load(self._label_list[idx]).astype(np.uint8).reshape((-1, 1))
        
        if self.raw:
            img = read_image(self._rgb_list[idx]) #.to(torch.float) / 255 
            img = img.permute(1,2,0)
            pred_labels = self.segmentation_encode(img) #self.segmentation_model.encoding_feature(img)
            pred_labels = pred_labels.detach().cpu().numpy()
            with open(self._depth_list[idx], "rb") as f:
                depth_map = np.load(f)
            # depth_map = np.load(self._depth_list[idx])
            points, _ = depth2pc(depth_map, intr_mat=self._intrinsic_matrix, min_depth=0.1, max_depth=6) 
        else:
            points =  np.load(self._velodyne_list[idx]).astype(np.float32).reshape(-1,3)[:, :3]
            pred_labels = np.load(self._pred_list[idx]).astype(np.float32)
        
        pred_labels = pred_labels.squeeze(0).transpose(1,2,0)
        pred_labels = pred_labels.reshape(-1,pred_labels.shape[-1])
            
        # change points coordinate from camera to rviz
        # camera x (right), y (down), z (forward)
        # rviz x (forward), y (left), z (up)
        points_x = points[:,0]
        points_y = points[:,1]
        points_z = points[:,2]
        points = np.vstack((points_z, -points_x, -points_y)).T 
        
        ### TODO: filter point outside the grid ###
        if self.grid_mask:
            grid_point_mask = np.all( (points < self.max_bound) & (points >= self.min_bound), axis=1)
            points = points[grid_point_mask]
            gt_labels = gt_labels[grid_point_mask]
            pred_labels = pred_labels[grid_point_mask]
            
            # filter out points that's too close to camera 
            close_point_mask = points[:,0] > 0.2
            points = points[close_point_mask]
            gt_labels = gt_labels[close_point_mask]
            pred_labels = pred_labels[close_point_mask]
        ### TODO: filter point outside the grid ###
        
        if self.down_sample_feature:
            pred_labels = torch.from_numpy(pred_labels)
            pred_labels = self.pca_downsample(pred_labels)
            pred_labels = pred_labels.detach().cpu().numpy().reshape(-1, self.latent_size)
        pred_labels = pred_labels.reshape(-1, self.latent_size) # dummy way

        return torch.from_numpy(pose), torch.from_numpy(points), torch.from_numpy(pred_labels), torch.from_numpy(gt_labels), scene_id, frame_id
