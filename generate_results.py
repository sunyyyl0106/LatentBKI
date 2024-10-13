import os
import time
import yaml
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
from tqdm import tqdm
import torch
import clip

# Custom Imports
from Data.utils import *
from Models.LatentBKI import *
from Models.mapping_utils import *
from Data.MP3D import MP3D
from Data.RealWorldData import RealWorldData
from Data.KITTI_SPVCNN import KITTI_SPVCNN
from Models.Lseg.Lseg_module import Lseg_module
from Models.SPVCNN.SPVCNN_module import SPVCNN_Module

# result same as train
class Results():
    def __init__(self) -> None:
        self.num_correct = 0
        self.num_total = 0
        self.all_intersections = 0
        self.all_unions = 0

        self.num_correct_seg = 0
        self.num_total_seg = 0
        self.all_intersections_seg = 0
        self.all_unions_seg = 0
        
def save_map():
    print("Saving Map ...")
    features = map_object.global_map[:,3:FEATURE_SIZE+3].to(device)
    if DOWN_SAMPLE_FEATURE:
        features = back_project_fn(features)
    labels = torch.argmax(map_object.decode(features, map_object.category_feature), dim=1, keepdim=True)
    confidence = map_object.global_map[:,-1].reshape(-1,1).to(dtype=torch.float32)
    global_map = torch.cat((map_object.global_map[:,:3], labels.to(torch.float32).cpu(), confidence), dim=1)
    global_map = global_map.numpy()
    print("Map size:", global_map.shape)
    np.save(os.path.join(SAVE_MAP_PATH, "global_map.npy"), global_map)
    np.save(os.path.join(SAVE_MAP_PATH, "global_map_latent.npy"), map_object.global_map)

def inference(unlabeld_pc_torch_list, pred_labels_list, gt_labels_list, map_object, results, SAVE_MAP_PATH, with_variance):
    # first save the points
    torch.save(unlabeld_pc_torch_list, os.path.join(SAVE_MAP_PATH, "unlabeld_pc_torch_list.pt"))
    torch.save(pred_labels_list, os.path.join(SAVE_MAP_PATH, "pred_labels_list.pt"))
    torch.save(gt_labels_list, os.path.join(SAVE_MAP_PATH, "gt_labels_list.pt"))

    print(f"Inference {last_scene} ... ")
    unlabeld_pc_torch_list = unlabeld_pc_torch_list.to(device=device, non_blocking=True)
    pred_labels_list = pred_labels_list.to(device=device, non_blocking=True)
    gt_labels_list = gt_labels_list.to(device=device, non_blocking=True)
    print(gt_labels_list.shape)
    features = map_object.label_points_iterative(unlabeld_pc_torch_list, with_variance=with_variance)
    torch.save(features.cpu(), os.path.join(SAVE_MAP_PATH, "predcited_features.pt")) # save predicted features, variance, confidence         
    
    category_pred = features[:, -1].to(torch.int64).to(device)
    
    for i in range(map_object.num_classes):
        gt_i = gt_labels_list == i
        pred_bki_i = category_pred == i
        pred_seg_i = pred_labels_list == i

        sequence_class[i] += torch.sum(gt_i)
        sequence_int_bki[i] += torch.sum(gt_i & pred_bki_i)
        sequence_int_seg[i] += torch.sum(gt_i & pred_seg_i)
        sequence_un_bki[i] += torch.sum(gt_i | pred_bki_i)
        sequence_un_seg[i] += torch.sum(gt_i | pred_seg_i)
            
    # accuracy
    correct = torch.sum(category_pred == gt_labels_list).item()
    total = gt_labels_list.shape[0]
    results.num_correct += correct
    results.num_total += total
    
    # miou
    inter, union = iou_one_frame(category_pred, gt_labels_list, n_classes=NUM_CLASSES)
    union += 1e-6
    results.all_intersections += inter
    results.all_unions += union
    
    # accuracy_seg
    # TODO: remove ignore lables?
    correct_seg = torch.sum(pred_labels_list == gt_labels_list).item()
    total_seg = gt_labels_list.shape[0]
    results.num_correct_seg += correct_seg
    results.num_total_seg += total_seg
    
    # miou_seg
    inter_seg, union_seg = iou_one_frame(pred_labels_list, gt_labels_list, n_classes=NUM_CLASSES)
    union_seg += 1e-6
    results.all_intersections_seg += inter_seg
    results.all_unions_seg += union_seg
    
    # save statistics
    print(f"{last_scene} stats:")
    seq_intersections = inter[union > 0]
    seq_unions = union[union > 0]
    seq_miou = np.mean(seq_intersections / seq_unions)
    print(f'Average map accuracy: {correct/total}')
    print(f'Map miou: {seq_miou}')
    
    seq_intersections_seg = inter_seg[union_seg > 0]
    seq_unions_seg = union_seg[union_seg > 0]
    seq_miou_seg = np.mean(seq_intersections_seg / seq_unions_seg)
    print(f'Average segmentation network accuracy: {correct_seg/total_seg}')
    print(f'Segmentation network miou: {seq_miou_seg}')
    print("")
    
    with open(os.path.join(SAVE_MAP_PATH, 'result.txt'), 'w') as file:
        file.write(f"{last_scene} stats:\n")
        seq_intersections = inter[union > 0]
        seq_unions = union[union > 0]
        seq_miou = np.mean(seq_intersections / seq_unions)
        file.write(f'Average map accuracy: {correct/total}\n')
        file.write(f'Map miou: {seq_miou}\n')
        
        seq_intersections_seg = inter_seg[union_seg > 0]
        seq_unions_seg = union_seg[union_seg > 0]
        seq_miou_seg = np.mean(seq_intersections_seg / seq_unions_seg)
        file.write(f'Average segmentation network accuracy: {correct_seg/total_seg}\n')
        file.write(f'Segmentation network miou: {seq_miou_seg}\n')


########################## main script ############################

# MODEL_NAME = "LatentBKI_default"
# MODEL_NAME = "LatentBKI_realworld"
# MODEL_NAME = "LatentBKI_vlmap"
MODEL_NAME = "LatentBKI_kitti"

print("Model is:", MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", device)

# Model Parameters
model_params_file = os.path.join(os.getcwd(), "Config", MODEL_NAME + ".yaml")
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
        DATASET = model_params["dataset"]
        MEAS_RESULT = model_params["meas_result"]
        SAVE_MAP = model_params["save_map"]
        ELL = model_params["ell"]
        WITH_VARIANCE = model_params['with_variance']
        USE_RELATIVE_POSE = model_params['use_relative_pose']
        PSEDUO_DISCRETE = model_params['pseduo_discrete']
        FILTER_SIZE = model_params["filter_size"]
        GRID_PARAMS = model_params["grid_params"]
    except yaml.YAMLError as exc:
        print(exc)

# Data Parameters
data_params_file = os.path.join(os.getcwd(), "Config", DATASET + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
        NUM_CLASSES = data_params["num_classes"]
        DATA_DIR = data_params["data_dir"]
        CATEGORY = data_params["category"]
        FEATURE_DIR = data_params["feature_dir"]
        FEATURE_SIZE = data_params["feature_size"]
        RAW_DATA = data_params['raw_data']
        DOWN_SAMPLE_FEATURE = data_params["down_sample_feature"]
        PCA_PATH = data_params['pca_path']
        GRID_MASK = data_params['grid_mask']
        SUBSAMPLE = data_params['subsample_points']
        SEQUENCES = data_params['sequences']
        INTRINSIC = data_params['intrinsic']
    except yaml.YAMLError as exc:
        print(exc)
        
# construct save directory 
cell_size = (GRID_PARAMS['max_bound'][0] - GRID_PARAMS['min_bound'][0]) / GRID_PARAMS['grid_size'][0]
SAVE_FOLDER = f"{MODEL_NAME}_{DATASET}_{FILTER_SIZE}_{ELL}_{FEATURE_SIZE}_{cell_size}_{SUBSAMPLE}"
RESULT_SAVE = os.path.join("Results", SAVE_FOLDER)

if SAVE_MAP:
    if not os.path.exists(RESULT_SAVE):
        os.makedirs(RESULT_SAVE)
    else:
        SAVE_FOLDER += time.strftime("_%Y-%m-%d_%H-%M-%S")
        RESULT_SAVE = os.path.join("Results", SAVE_FOLDER)
    print(f"Save to {RESULT_SAVE}")

print("Measure Result:", MEAS_RESULT)
print("Save Map :", SAVE_MAP)
print("grid_mask:", GRID_MASK)
print("Pseudo discrete: ", PSEDUO_DISCRETE)
print("with variance inference:", WITH_VARIANCE)
print("Subsampling input points:", SUBSAMPLE)

# PCA feature reduction functions
down_sampling_fn = None
back_project_fn = None
CATEGORY_CLIP = torch.empty(0, device=device)
# Create segmentation module
if DATASET != 'semantic_kitti':
    # clip features
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(CATEGORY).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = text_features.to(torch.float32)
        CATEGORY_CLIP = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"category_clip size: {CATEGORY_CLIP.shape}")

    # lseg module
    seg_module = Lseg_module(pca_path=PCA_PATH, device=device)
    if DOWN_SAMPLE_FEATURE:
        down_sampling_fn = seg_module.down_sampling
        back_project_fn = seg_module.backproject_to_clip
else:
    print("before SPVCNN_Module")
    seg_module = SPVCNN_Module(device)
    print("after SPVCNN_Module")

# Load data set
if DATASET == "mp3d":
    test_ds = MP3D(
        GRID_PARAMS, 
        INTRINSIC,
        segmentation_encode=seg_module.encoding_feature,
        pca_downsample=down_sampling_fn,
        feature_dir=FEATURE_DIR,
        directory=DATA_DIR, 
        device=device,
        latent_size=FEATURE_SIZE, 
        down_sample_feature=DOWN_SAMPLE_FEATURE,
        sequences=SEQUENCES,
        raw=RAW_DATA,
        grid_mask=GRID_MASK
    )
elif DATASET == 'realworld':
    test_ds = RealWorldData(
        GRID_PARAMS, 
        INTRINSIC,
        segmentation_encode=seg_module.encoding_feature,
        pca_downsample=down_sampling_fn,
        feature_dir=FEATURE_DIR,
        directory=DATA_DIR, 
        device=device,
        latent_size=FEATURE_SIZE, 
        down_sample_feature=DOWN_SAMPLE_FEATURE,
        sequences=SEQUENCES,
    )
elif DATASET == 'semantic_kitti':
    test_ds = KITTI_SPVCNN(device=device, grid_params=GRID_PARAMS)
else:
    raise ValueError("Invalid Dataset")
    
# Create map object
map_object = GlobalMapContinuous(
    torch.tensor([int(p) for p in GRID_PARAMS['grid_size']], dtype=torch.long).to(device),  # Grid size
    torch.tensor(GRID_PARAMS['min_bound']).to(device),  # Lower bound
    torch.tensor(GRID_PARAMS['max_bound']).to(device),  # Upper bound
    FILTER_SIZE, # Filter size
    decode=seg_module.decoding_feature,
    pca_upsample=back_project_fn,
    ell=ELL,
    category_feature=CATEGORY_CLIP,
    num_classes=NUM_CLASSES,
    latent_dim=FEATURE_SIZE,
    device=device, # Device
    use_relative_pose=USE_RELATIVE_POSE,
    pseduo_discrete = PSEDUO_DISCRETE
)

# result statistics
results = Results()
sequence_class = torch.zeros(map_object.num_classes, device=device)
sequence_int_bki = torch.zeros(map_object.num_classes, device=device)
sequence_int_seg = torch.zeros(map_object.num_classes, device=device)
sequence_un_bki = torch.zeros(map_object.num_classes, device=device)
sequence_un_seg = torch.zeros(map_object.num_classes, device=device)
total_t = 0.0

# Iteratively loop through each scan
last_scene = None
last_frame_id = None
seq_dir = None
frame_num = 0
unlabeld_pc_torch_list = torch.empty(0,3)
pred_labels_list = torch.empty(0)
gt_labels_list = torch.empty(0)

# for idx in tqdm(range(len(test_ds))):
for idx in tqdm(range(0, 10, 1)):
# for idx in tqdm([0,50]):
    with torch.no_grad():
        # Load data
        pose, points, pred_labels, gt_labels, scene_id, frame_id = test_ds.get_test_item(idx)
        
        # NOTE: scene_id, frame_id is the id will be processed, curent_id is the id been processed in last iteration
        # Reset and mearues result if new subsequence
        if scene_id != last_scene: #or (frame_id - 1) != last_frame_id:
            if MEAS_RESULT and map_object.global_map is not None and DATASET != "realworld":
                # save map
                if SAVE_MAP:
                    SAVE_MAP_PATH = os.path.join(RESULT_SAVE, last_scene)
                    if not os.path.exists(SAVE_MAP_PATH):
                        os.makedirs(SAVE_MAP_PATH)
                    save_map()
                
                # inference
                inference(unlabeld_pc_torch_list, pred_labels_list, gt_labels_list, map_object, results, SAVE_MAP_PATH, WITH_VARIANCE)
    
                # reset unlabeled pc
                unlabeld_pc_torch_list = torch.empty(0,3)
                pred_labels_list = torch.empty(0)
                gt_labels_list = torch.empty(0)
                
            map_object.reset_grid()
        
        # Update pose if not
        start_t = time.time()
        map_object.propagate(pose)
        
        # Add points to map
        labeled_pc_torch = torch.hstack((points.to(device), pred_labels.to(device)))
        
        # NOTE: subsample ranomd points for heldout calculation 
        if DATASET != 'realworld':
            # additional processing for comparison with VLMap
            if MODEL_NAME == "LatentBKI_vlmap":
                # subsample 1% input points
                depth_sample_rate = 100
                np.random.seed(42)
                shuffle_mask = np.arange(labeled_pc_torch.shape[0])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::depth_sample_rate]
                labeled_pc_torch = labeled_pc_torch[shuffle_mask, :]
                gt_labels = gt_labels[shuffle_mask, :]
                pred_labels = pred_labels[shuffle_mask, :]
                
                # out of range points filter out our pose
                pc_global = map_object.camera_to_global(labeled_pc_torch[:,:3]).clone().cpu().numpy()
                rows_cols_heights = np.array([base_pos2grid_id_3d(p[0], p[1], p[2]+1.5) for p in pc_global]) # add camera height with 1.5 on z axis
            
                out_of_range_mask = np.array([out_of_range(row, col, height) for row, col, height in rows_cols_heights])
                labeled_pc_torch = labeled_pc_torch[~out_of_range_mask, :]
                pc_global = pc_global[~out_of_range_mask, :]
                gt_labels = gt_labels[~out_of_range_mask, :]
                pred_labels = pred_labels[~out_of_range_mask, :]
                
                # close camera points
                mask = labeled_pc_torch[:, 0] > 0.1
                mask = torch.logical_and(mask, labeled_pc_torch[:, 0] < 6)
                labeled_pc_torch = labeled_pc_torch[mask, :]
                
                # eval & input the same
                gt_labels = gt_labels[mask, :]
                pred_labels = pred_labels[mask, :]
                unlabeld_pc_torch = labeled_pc_torch[:,:3].clone() 
            else:
                # heldout points
                np.random.seed(42) # each data point has a diffrent random seed preventing generate same random index
                point_num = labeled_pc_torch.shape[0]
                sampled_index = np.random.choice(point_num, int(0.2*point_num), replace=False)
                heldout_mask = np.full(point_num, False)
                heldout_mask[sampled_index] = True
                
                # mask for heldout points, eval points set
                gt_labels = gt_labels[heldout_mask, :]
                pred_labels = pred_labels[heldout_mask, :]
                unlabeld_pc_torch = labeled_pc_torch[heldout_mask, :3]
                labeled_pc_torch = labeled_pc_torch[~heldout_mask, :]
                
                # testing to use fewer points for update 
                if SUBSAMPLE < 1 and SUBSAMPLE > 0:
                    # use subsample points here
                    point_num = labeled_pc_torch.shape[0]
                    sampled_index = np.random.choice(point_num, int(SUBSAMPLE*point_num), replace=False)
                    mask = np.full(point_num, False)
                    mask[sampled_index] = True
                    labeled_pc_torch = labeled_pc_torch[mask, :]
                
                # TODO: input eval the same #
                # gt_labels = gt_labels[mask, :]
                # pred_labels = pred_labels[mask, :]
                # unlabeld_pc_torch = labeled_pc_torch[:, :3].clone()
                # TODO: input eval the same #
            
        # update map using observations
        map_object.update_map(labeled_pc_torch)
        total_t += time.time() - start_t

        if MEAS_RESULT and DATASET != "realworld":
            # decode pred_labels
            pred_labels = pred_labels.to(device, non_blocking=True)
            if DOWN_SAMPLE_FEATURE:
                pred_labels = back_project_fn(pred_labels)
            pred_labels = seg_module.decoding_feature(pred_labels, map_object.category_feature)
            pred_labels = pred_labels.softmax(dim=-1)
            
            if pred_labels.shape[1] > 1:
                pred_labels = torch.argmax(pred_labels, dim=1)
            else:
                pred_labels = pred_labels.view(-1)
            
            # camera frame to global frame
            unlabeld_pc_torch = map_object.camera_to_global(unlabeld_pc_torch)
            unlabeld_pc_torch_list = torch.vstack((unlabeld_pc_torch_list,unlabeld_pc_torch.detach().cpu()))
            pred_labels_list = torch.hstack((pred_labels_list, pred_labels.detach().cpu()))
            gt_labels_list = torch.hstack((gt_labels_list, gt_labels.view(-1)))
    
    last_scene = scene_id
    last_frame_id = frame_id
    frame_num += 1

# post processing
if SAVE_MAP:
    SAVE_MAP_PATH = os.path.join(RESULT_SAVE, last_scene)
    if not os.path.exists(SAVE_MAP_PATH):
        os.makedirs(SAVE_MAP_PATH)
    save_map()

if MEAS_RESULT and DATASET != "realworld":
    # if KNN_INFERENCE:
    inference(unlabeld_pc_torch_list, pred_labels_list, gt_labels_list, map_object, results, SAVE_MAP_PATH, WITH_VARIANCE)
    
    with open(os.path.join(RESULT_SAVE, 'result.txt'), 'w') as file:
        file.write("Final results:\n")
        file.write("Seg:\n")
        for i in range(NUM_CLASSES):
            file.write(f"{i}: {(sequence_int_seg[i] / sequence_un_seg[i] * 100).item()} ({sequence_int_seg[i]} / {sequence_un_seg[i]})\n")
        file.write("BKI:\n")
        for i in range(NUM_CLASSES):
            file.write(f"{i}: {(sequence_int_bki[i] / sequence_un_bki[i] * 100).item()} ({sequence_int_bki[i]} / {sequence_un_bki[i]})\n")
            
        file.write("Map_update statistics:\n")
        
        all_intersections = results.all_intersections[results.all_unions > 0]
        all_unions = results.all_unions[results.all_unions > 0]
        all_miou = np.mean(all_intersections / all_unions)
        file.write(f'Average map accuracy: {results.num_correct/results.num_total}\n')
        file.write(f'Map miou: {all_miou}\n')
        
        all_intersections_seg = results.all_intersections_seg[results.all_unions_seg > 0]
        all_unions_seg = results.all_unions_seg[results.all_unions_seg > 0]
        all_miou_seg = np.mean(all_intersections_seg / all_unions_seg)
        file.write(f'Average segmentation network accuracy: {results.num_correct_seg/results.num_total_seg}\n')
        file.write(f'Segmentation network miou: {all_miou_seg}\n')



    