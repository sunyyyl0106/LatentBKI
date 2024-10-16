# LatentBKI
This repository is the implementation of [Latent BKI](), aiming to reproduce the experiment results shown in the paper. 

![Teaser](/figs/teaser.png)

## Installation (test with python 3.7 + torch1.11 + cuda11.3)

```
conda env create -f environment.yml
conda activate latentbki_env
```

There are some package needs manual installation:

1. Clip: follow the [repo](https://github.com/openai/CLIP).
2. [torchsparse](https://github.com/mit-han-lab/torchsparse) (sudo apt-get install libsparsehash-dev, pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0)
3. [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source)

(Optional) Install [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) to visualize map in Rviz

> Note: Also checkout all [dependent codebase](#acknowledgements) if facing any issues.

## Data preparation

### Matterport 3D (MP3D)
Follow VLMap's [Generate dataset](https://github.com/vlmaps/vlmaps?tab=readme-ov-file#generate-dataset) section to get MP3D sequences. The ground truth semantic generated is a bit incorrect, so we provided a modifcation to obtain correct ground truth data [here](https://drive.google.com/drive/folders/1dWJXcvHyBimh8KMvA7e3zwApXju6tXTZ?usp=drive_link). 

### Semantic KITTI
Follow 2DPASS' [Data Preparation](https://github.com/yanx27/2DPASS?tab=readme-ov-file#data-preparation) section to obtain Semantic KIITI dataset under `Dataset` folder

Download SPVCNN model checkpoint from [here](https://drive.google.com/drive/folders/1VpY2MCp5i654pXjizFMw0mrmuxC_XboW) and put it under `./TwoDPASS/pretrained/SPVCNN`.

### Real World data with iPhone/iPad
1. Download [Record3D](https://record3d.app/) app on iPhone/iPad, record a video and export it as `.r3d` file. 
2. Extract the files in `.r3d` same as extracting from zip files, you will get a folder named `rgbd` and a `metadata` file.
3. Run `Data/select_r3d_frames.py` with customized parameters to create following real world dataset folder structure: 
```
/[your dataset path]
├── 
├── ...
└── real_world/
    ├──[sequence name]          
        ├── conf/	
        |   ├── 000000.npy
        |   ├── 000001.npy
        |   └── ...
        └── depth/ 
        |   ├── 000000.npy
        |   ├── 000001.npy
        |   └── ...
        └── rgb/ 
        |   ├── 000000.jpg
        |   ├── 000001.jpg
        |   └── ...
        └── intrinsics.txt
        └── poses.txt
```
You can download an already processed Record3D data example, `my_house_long.zip`, [here](https://drive.google.com/drive/folders/1dWJXcvHyBimh8KMvA7e3zwApXju6tXTZ?usp=drive_link).

### Download PCA downsampler

Download `mp3d_pca_64.pkl` [here](https://drive.google.com/drive/folders/1dWJXcvHyBimh8KMvA7e3zwApXju6tXTZ?usp=drive_link) to `./PCAonGPU/PCA_instance`

## Usage

### MP3D data
Required to provide path in `./config/mp3d.yaml`to following parameters:

```
- data_dir: "/path/to/realworld/dataset/folder"
- pca_path: "/path/to/trained/pca/.pkl/file"
```

Other parameters are optional if only want to reproduce the result.

### Real-world data 

Modify `realworld.yaml` under `./config`


```
Required parameters:
- num_classes: [number of class desired]
- data_dir: "/path/to/realworld/dataset/folder"
- pca_path: "/path/to/trained/pca/.pkl/file"
- intrinsic: [matrix from the intrinsic.txt]
- sequences: [
  [your_sequences_name]
]
- category: [
    [List of words you want decode]
]

Optional parameters:
- feature_size: [PCA downsampled size, default 64]
- grid_mask: [ignore points outside local grid, default True]
- down_sample_feature: [default True]
- raw_data: [Set to True only if features are saved to disk]
- subsample_points: [How much pixel feature to use, default 1, use all feature]
- feature_dir: [set it only if you save latent feature to disk]

```

### KITTI

NOTE: `semantic_kitti.yaml` is used to provide additional parameters, such as feature size. We are using the dataloader in 2DPASS. Change the following parameters in `TwoDPASS/config/SPVCNN-semantickitti.yaml`:

```
train_data_loader:
    data_path: "/path/to/kitti/dataset/sequences"

val_data_loader:
    data_path: "/path/to/kitti/dataset/sequences"
```

### run mapping algorithm

In `./generate_results.py`, set `MODEL_NAME` to one of the following:

- "LatentBKI_default": latent mapping using MP3D
- "LatentBKI_kitti": latent mapping using semantic KITTI
- "LatentBKI_vlmap": including vlmap heuristic for comparison experiment
- "LatentBKI_realworld": map real-world environment captured by Record3D

Generated latent map and evaluation result for each sequence will be under `Results` folder.

### Evaluate map
In `./inference.py`, provide the following parameters:
```
- RESULT_SAVE: the folder that contain the map you want to evaluate
- MODEL_NAME: The model you used to create the above map
- scenes: the sequences you want to evaluate
```

The evalution result will be under the folder you provided to `RESULT_SAVE` as a `results.txt` file.

### Visualize map (ROS required)
1. Run `./publish_map.py` with `latent_map` and `category_map` set to the map you want to visualize.
2. Open Rviz and subscribe to topic `visualization_marker_array`

### Open-Dictionary Query Demo (ROS required)
1. Run `./publish_map.py` with customized `MODEL_NAME` and `latent_map_path` parameter.
2. Open Rviz and subscribe to topic `Open_Query/Heatmap` and `Open_Query/Uncertainty`
3. In terminal follow the prompt to query arbitrary word.

## Acknowledgements

code is built based on [ConvBKI](https://github.com/UMich-CURLY/NeuralBKI), [VLMaps](https://github.com/vlmaps/vlmaps), [2DPASS](https://github.com/yanx27/2DPASS/blob/main/README.md?plain=1), [Record3D](https://github.com/marek-simonik/record3d)