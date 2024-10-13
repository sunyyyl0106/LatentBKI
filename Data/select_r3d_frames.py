import os
import sys
import yaml
import numpy as np
import cv2
import liblzfse 
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm

def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        depth_img = depth_img.reshape((192, 256)) 
    return depth_img

def load_conf(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf = np.frombuffer(decompressed_bytes, dtype=np.int8)
        conf = conf.reshape((192, 256))
    return np.float32(conf)

def main(args):
    # set data directory
    data_dir = "/Users/multyxu/Desktop/Programming/LatentBKI data/record3d/my_house_long"
    output_dir = os.path.join('/Users/multyxu/Desktop/Programming/LatentBKI data/', os.path.basename(data_dir))
    
    # create directoris
    rgb_dir = os.path.join(output_dir, "rgb")
    dpeth_dir = os.path.join(output_dir, "depth")
    conf_dir = os.path.join(output_dir, "conf")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(rgb_dir).mkdir(parents=True, exist_ok=True)
    Path(dpeth_dir).mkdir(parents=True, exist_ok=True)
    Path(conf_dir).mkdir(parents=True, exist_ok=True)
    
    # read meata data: camera intrinsics and pose
    metadata_path = os.path.join(data_dir, 'metadata')
    metadata = yaml.safe_load(open(metadata_path, 'r'))
    K = np.asarray(metadata['K']).reshape(3,3).T
    poses = np.array(metadata['poses'])
    q_list = np.hstack((poses[:, 3].reshape(-1,1), poses[:, :3])) # (w, x, y, z)
    xyz_list = poses[:, 4:]
    
    # write intrinsics to file
    np.savetxt(os.path.join(output_dir, 'intrinsics.txt'), K)
    
    # count total number of frames
    num_frames = len(glob(os.path.join(data_dir,'rgbd/*.depth' )))
    print("Num Frames:", num_frames)

    pose_list = []
    frame_count = 0
    # fps is very high, take images every 5 frames, making it 15fps
    for i in tqdm(range(0, num_frames, 5)):
        depth_filepath = os.path.join(data_dir, f'rgbd/{i}.depth')
        rgb_filepath = os.path.join(data_dir, f'rgbd/{i}.jpg')
        conf_filepath = os.path.join(data_dir, f'rgbd/{i}.conf')
        
        depth_img = load_depth(str(depth_filepath))
        conf = load_conf(str(conf_filepath))
        
        depth_resized = cv2.resize(depth_img, (960, 720))
        conf_resized = cv2.resize(conf, (960, 720), cv2.INTER_NEAREST_EXACT)
        
        # save to file
        rgb_output = os.path.join(rgb_dir, '%06i.jpg' % frame_count)
        depth_output = os.path.join(dpeth_dir, '%06i.npy' % frame_count)
        conf_output = os.path.join(conf_dir, '%06i.npy' % frame_count)
        
        shutil.copyfile(rgb_filepath, rgb_output)
        np.save(depth_output, depth_resized)
        np.save(conf_output, conf_resized)
        
        # append OpenGL camera pose
        pose_list.append(np.hstack((xyz_list[i], q_list[i])).tolist())
        
        frame_count += 1
    
    # save poses
    np.savetxt(os.path.join(output_dir, "poses.txt"), np.array(pose_list))

if __name__ == '__main__':
    main(sys.argv)