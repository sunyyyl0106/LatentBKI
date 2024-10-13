import torch
import yaml
import importlib

from easydict import EasyDict

from torch.utils.data import Dataset
from TwoDPASS.dataloader.dataset import get_model_class, get_collate_class
from TwoDPASS.dataloader.pc_dataset import get_pc_model_class

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    args = {}
    args['config_path'] = 'TwoDPASS/config/SPVCNN-semantickitti.yaml'
    args['seed'] = 0
    args['gpu'] = (0,)
    
    # training
    args['log_dir'] = 'default'                             
    args['monitor'] = 'val/mIoU'                            
    args['stop_patience'] = 50                              
    args['save_top_k'] = 1                                  
    args['check_val_every_n_epoch'] = 1                     
    args['SWA'] = False                                     
    args['baseline_only'] = False            
    # testing               
    args['test'] = True                                     
    args['fine_tune'] = False                               
    args['pretrain2d'] = False                              
    args['num_vote'] = 1                                    
    args['submit_to_server'] = False                        
    args['checkpoint'] = 'TwoDPASS/pretrained/SPVCNN/best_model.ckpt'
    # debug
    args['debug'] = False
    
    config = load_yaml(args['config_path'])
    config.update(args)  # override the configuration using the value in args

    # voting test
    if args['test']:
        config['dataset_params']['val_data_loader']['batch_size'] = args['num_vote']
    if args['num_vote'] > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if args['debug']:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)

class KITTI_SPVCNN_config():
    def __init__(self) -> None:
        self.config = parse_config()

class KITTI_SPVCNN(Dataset):
    def __init__(self, device, grid_params, grid_mask=True) -> None:
        super().__init__()
        self.device = device
        self.grid_mask = grid_mask
        self._grid_size = grid_params['grid_size']
        self.coor_ranges = grid_params['min_bound'] + grid_params['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = torch.tensor(self.coor_ranges[:3])
        self.max_bound = torch.tensor(self.coor_ranges[3:])
        self.config = parse_config()
        self.init_dataset()
        self.init_model()
    
    def init_dataset(self):
        pc_dataset = get_pc_model_class(self.config['dataset_params']['pc_dataset_type'])
        dataset_type = get_model_class(self.config['dataset_params']['dataset_type'])
        val_config = self.config['dataset_params']['val_data_loader']
        val_pt_dataset = pc_dataset(self.config, data_path=val_config['data_path'], imageset='val', num_vote=val_config["batch_size"])
        
        self.kitti_dataset = dataset_type(val_pt_dataset, self.config, val_config, num_vote=val_config["batch_size"])
        self.collate_fn = get_collate_class(self.config['dataset_params']['collate_type'])
        
    def init_model(self):
        model_file = importlib.import_module('TwoDPASS.network.' + self.config['model_params']['model_architecture'])
        my_model = model_file.get_model(self.config) ######## get model ############
        my_model = my_model.load_from_checkpoint(self.config.checkpoint, config=self.config, strict=(not self.config.pretrain2d))
        my_model = my_model.eval()
        
        self.my_model = my_model.to(self.device)
    
    def __len__(self):
        return len(self.kitti_dataset)
    
    def __getitem__(self, idx):
        return self.get_test_item(idx)
    
    def get_test_item(self, idx):
        data_dict = self.collate_fn([self.kitti_dataset[idx]])
        with torch.no_grad():
            features = self.my_model.encode_points(data_dict, self.device)
            
        points = data_dict['points'].F
        gt_labels = data_dict['targets_mapped'].F.reshape(-1,1)
        
        # only take points in the grid
        if self.grid_mask:
            grid_point_mask = torch.all( (points < self.max_bound) & (points >= self.min_bound), axis=1)
            points = points[grid_point_mask]
            gt_labels = gt_labels[grid_point_mask]
            features = features[grid_point_mask]
        
        return data_dict['global_pose'][0], points, features, gt_labels, data_dict['scene_id'][0], data_dict['frame_id'][0]
        