# %%
import sys
import os
import clip
from pathlib import Path
sys.path.append('/workspace/LatentBKI')
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from Models.Lseg.Lseg_module import Lseg_module
from torchvision.io import read_image

# %%
PCA_PATH = '/mp3d/PCA/64/ipca_7.pkl'
seg_module = Lseg_module(pca_path=PCA_PATH)

data_path = '/mp3d/vlmaps_data_dir/vlmaps_dataset/gTV8FGcVJC9_1'
pred_64_save = os.path.join(data_path, 'lseg_pred_64_new')
Path(pred_64_save).mkdir(parents=True, exist_ok=True)

# %%
# category_clip = torch.tensor(np.load("/workspace/LatentBKI/Data/category_vlmap_features.npy"),device='cuda').to(torch.float32)

CATEGORY = np.loadtxt("/workspace/LatentBKI/Data/category_vlmap.txt", dtype='str')
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(CATEGORY).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text)
    text_features = text_features.to(torch.float32)
    CATEGORY_CLIP = text_features / text_features.norm(dim=-1, keepdim=True)
    
# %%
img_path_list = glob(os.path.join(data_path,'rgb/*.png'))
img_path_list.sort()
print(img_path_list)
print(len(img_path_list))

# %%
for i, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
    img = read_image(img_path) #.to(torch.float) / 255 
    img = img[:3]
    img = img.permute(1,2,0)
    # encode image
    clip_features = seg_module.encoding_feature(img)
    clip_features = clip_features.squeeze(0).permute(1,2,0)
    clip_features = clip_features.reshape(-1,512)
    # project to 64 and back to 512
    clip_features_64 = seg_module.down_sampling(clip_features)
    clip_featreu_bp = seg_module.backproject_to_clip(clip_features_64)
    # to category
    lseg_logits_64 = seg_module.decoding_feature(clip_featreu_bp, CATEGORY_CLIP)
    lseg_logits_64 = lseg_logits_64.softmax(dim=-1) # convert to probability
    lseg_pred_64 = lseg_logits_64.argmax(dim=-1)
    lseg_pred_64 = lseg_pred_64.reshape(720,1080).cpu().numpy()
    # save image
    save_path = os.path.join(pred_64_save, '%06i.npy' % i)
    np.save(save_path, lseg_pred_64)


