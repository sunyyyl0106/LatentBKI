import os
import yaml
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import clip

# Custom Imports
from Data.utils import *
from Models.LatentBKI import *
from Models.mapping_utils import *
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
    print(global_map.shape)
    np.save(os.path.join(SAVE_MAP_PATH, "global_map.npy"), global_map)
    np.save(os.path.join(SAVE_MAP_PATH, "global_map_latent.npy"), map_object.global_map)

def inference(unlabeld_pc_torch_list, pred_labels_list, gt_labels_list, map_object, results, SAVE_MAP_PATH, with_variance):
    # first save the points
    torch.save(unlabeld_pc_torch_list, os.path.join(SAVE_MAP_PATH, "unlabeld_pc_torch_list.pt"))
    torch.save(pred_labels_list, os.path.join(SAVE_MAP_PATH, "pred_labels_list.pt"))
    torch.save(gt_labels_list, os.path.join(SAVE_MAP_PATH, "gt_labels_list.pt"))

    print(f"Inference {current_scene} ... ")
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
    print(f"{current_scene} stats:")
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
    
    with open(os.path.join(SAVE_MAP_PATH, 'result_inference.txt'), 'w') as file:
        file.write(f"{current_scene} stats:\n")
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

MODEL_NAME = "LatentBKI_default"
# MODEL_NAME = "LatentBKI_realworld"
# MODEL_NAME = "LatentBKI_vlmap"
# MODEL_NAME = "LatentBKI_kitti"
RESULT_SAVE = 'Results/LatentBKI_default_mp3d_3_0.5_64_0.1_1'
# scenes = ['5LpN3gDmAk7_1' , 'gTV8FGcVJC9_1', ]
scenes = ['5LpN3gDmAk7_1' ]
# scenes = ['08']
WIHT_VARIANCE = False
DISCRETE = True
BATCH_SIZE = 100000

print("Model is:", MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", device)

print("---------------")
print("with variance: ", WIHT_VARIANCE)
print("discrete_knn: ", DISCRETE)
print("result save:", RESULT_SAVE)
print("batch size:", BATCH_SIZE)

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

# PCA feature reduction functions
down_sampling_fn = None
back_project_fn = None
CATEGORY_CLIP = None
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
    seg_module = SPVCNN_Module(device)

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

# Evaluation Loop
for current_scene in scenes:
    # load data
    print(f"Processing {current_scene}")
    folder = os.path.join(RESULT_SAVE, current_scene)
    unlabeld_pc_torch_list = torch.load(f"{folder}/unlabeld_pc_torch_list.pt")
    pred_labels_list = torch.load(f"{folder}/pred_labels_list.pt")
    gt_labels_list = torch.load(f"{folder}/gt_labels_list.pt")
    map_object.global_map = torch.tensor(np.load(f"{folder}/global_map_latent.npy"), dtype=torch.float)
    
    print("map shape:", map_object.global_map.shape)

    SAVE_MAP_PATH = os.path.join(RESULT_SAVE, current_scene)
    if not os.path.exists(SAVE_MAP_PATH):
        print(SAVE_MAP_PATH)
        os.makedirs(SAVE_MAP_PATH)
        
    inference(unlabeld_pc_torch_list, pred_labels_list, gt_labels_list, map_object, results, SAVE_MAP_PATH, WITH_VARIANCE)

# Write result to file
with open(os.path.join(RESULT_SAVE, 'result_inference.txt'), 'w') as file:
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