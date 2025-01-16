"""
This code is adapted from the open-source project VLMaps
Original source: https://github.com/vlmaps/vlmaps
License: MIT License
"""
import math
import os
from pathlib import Path
import gdown

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image

from .models.lseg_net import LSegEncNet

def init_lseg(device):
    crop_size = 480  # 480
    base_size = 1080  # 520
    lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
    model_state_dict = lseg_model.state_dict()
    checkpoint_dir = Path(__file__).resolve().parents[0] / "lseg" / "checkpoints"
    checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not checkpoint_path.exists():
        print("Downloading LSeg checkpoint...")
        # the checkpoint is from official LSeg github repo
        # https://github.com/isl-org/lang-seg
        checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
        gdown.download(checkpoint_url, output=str(checkpoint_path))

    pretrained_state_dict = torch.load(checkpoint_path, map_location=device)
    pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
    model_state_dict.update(pretrained_state_dict)
    lseg_model.load_state_dict(pretrained_state_dict)

    lseg_model.eval()
    lseg_model = lseg_model.to(device)

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    lseg_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    clip_feat_dim = lseg_model.out_c
    return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std, clip_feat_dim

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.shape #.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    # hsv_step = int(179 / n)
    # for j in range(0, n):
    #     hsv = np.array([hsv_step * j, 255, 255], dtype=np.uint8).reshape((1,1,3))
    #     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #     rgb = rgb.reshape(-1)
    #     pallete[j * 3 + 0] = rgb[0]
    #     pallete[j * 3 + 1] = rgb[1]
    #     pallete[j * 3 + 2] = rgb[2]

    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None, ignore_ids_list=[]):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches



def get_lseg_feat(
    model: LSegEncNet,
    image: np.array,
    labels,
    transform,
    device,
    crop_size=480,
    base_size=1080,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5],
    vis=False,
):
    if vis:
        vis_image = image.clone().numpy() #.copy()
    image = transform(image).unsqueeze(0).to(device)
    # print("image shape: ", image.shape)
    img = image[0].permute(1, 2, 0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    # print("h: ", h)
    # print("w: ", w)
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)
    # print("stride: ", stride)

    # long_size = int(math.ceil(base_size * scale))
    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{"mode": "bilinear", "align_corners": True})
    # print("cur_img size", np.shape(cur_img))

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        with torch.no_grad():
            # outputs = model(pad_img)
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        # print("pad_img shape", pad_img.shape)
        _, _, ph, pw = pad_img.shape  # .size()
        assert ph >= height and pw >= width
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().to(device)
                logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().to(device)
            count_norm = image.new().resize_(batch, 1, ph, pw).zero_().to(device)
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    # output = model(pad_crop_img)
                    output, logits = model(pad_crop_img, labels)
                    # print("pad_crop_img.shape", pad_crop_img.shape)
                cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                outputs[:, :, h0:h1, w0:w1] += cropped
                logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                count_norm[:, :, h0:h1, w0:w1] += 1
        assert (count_norm == 0).sum() == 0
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:, :, :height, :width]
        logits_outputs = logits_outputs[:, :, :height, :width]
    # outputs = resize_image(outputs, h, w, **{'mode': 'bilinear', 'align_corners': True})
    # outputs = resize_image(outputs, image.shape[0], image.shape[1], **{'mode': 'bilinear', 'align_corners': True})
    outputs = outputs.cpu()
    outputs = outputs.numpy()  # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]
    #print("pred.shape: ", pred.shape)
    if vis:
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")
        cv2.imshow("image", vis_image[:, :, [2, 1, 0]])
        cv2.waitKey()
        fig = plt.figure()
        plt.imshow(seg)
        plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 20})
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    #print("outputs.shape", outputs.shape)
    return outputs
