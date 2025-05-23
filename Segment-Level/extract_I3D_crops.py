#!/usr/bin/env python3

import os
import sys
import ast
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import argparse
from collections import Counter

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Extract I3D features + clip-level action labels for PDAN."
)
parser.add_argument("--csv", type=str,
                    default="/data/annotations/bounding_boxes.csv",
                    help="Path to unified bounding boxes CSV")
parser.add_argument("--frames_dir", type=str,
                    default="/data/frames",
                    help="Base directory where frame folders are located")
parser.add_argument("--output_dir", type=str,
                    default="./i3d_features_for_pdan_crop",
                    help="Directory to save per-track .npz with feats+labels")
parser.add_argument("--clip_len", type=int, default=16,
                    help="Number of frames per clip for I3D input")
parser.add_argument("--frame_stride", type=int, default=16,
                    help="Stride between consecutive clips within a track")
args = parser.parse_args()

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

# --- Load I3D R50 K400 and remove projection ---
print("[Info] Loading I3D R50 K400 model...")
try:
    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
    model = model.to(device).eval()
    if hasattr(model.blocks[6], 'proj'):
        feature_dim = model.blocks[6].proj.in_features
        model.blocks[6].proj = nn.Identity()
    elif hasattr(model, 'head') and hasattr(model.head, 'proj'):
        feature_dim = model.head.proj.in_features
        model.head.proj = nn.Identity()
    else:
        raise RuntimeError("Cannot find final proj layer to replace.")
    print(f"[Info] Extractor ready, feature_dim={feature_dim}")
except Exception as e:
    print(f"[Error] Loading model: {e}", file=sys.stderr)
    sys.exit(1)

# --- Transforms ---
side_size = 256
crop_size = 224
mean = [0.45, 0.45, 0.45]
std  = [0.225, 0.225, 0.225]

pil_to_tensor   = T.ToTensor()
frame_transform = T.Compose([
    T.Resize(side_size),
    T.CenterCrop(crop_size),
    T.Normalize(mean, std)
])

# --- Load CSV into dict: video->track_idx->{frame: [(bbox, action_id)]} ---
print(f"[Info] Loading CSV: {args.csv}")
try:
    df = pd.read_csv(args.csv, sep=";", dtype={'video_name': str, 'bounding_boxes': str, 'frame': int})
except Exception as e:
    print(f"[Error] loading CSV: {e}", file=sys.stderr)
    sys.exit(1)

frame_boxes = {}
for _, row in df.dropna(subset=['bounding_boxes']).iterrows():
    video = row['video_name']
    frame = int(row['frame'])
    try:
        boxes = ast.literal_eval(row['bounding_boxes'])
    except:
        continue
    for b in boxes:
        if len(b) < 6: 
            continue
        x0, y0, x1, y1, action_id, bird_id = b
        bird_id = int(bird_id)
        action_id = int(action_id)
        frame_boxes.setdefault(video, {}) \
                   .setdefault(bird_id, {}) \
                   .setdefault(frame, []) \
                   .append(((x0, y0, x1, y1), action_id))

# --- Ensure output directory ---
os.makedirs(args.output_dir, exist_ok=True)

# --- Process per-track ---
total_crops = 0
for video in tqdm(sorted(frame_boxes.keys()), desc="Videos"):
    video_dir = os.path.join(args.frames_dir, video)
    if not os.path.isdir(video_dir):
        continue

    # sort frames per track
    for track_id, frames_dict in frame_boxes[video].items():
        frames_sorted = sorted(frames_dict.items())  # [(frame_nr, [((bbox),action_id),...]), ...]
        feat_clips = []
        label_clips = []

        N = len(frames_sorted)
        for start in range(0, N, args.frame_stride):
            end = start + args.clip_len
            if end > N:
                break
            clip = frames_sorted[start:end]
            tensors = []
            clip_actions = []

            valid = True
            for frame_nr, detections in clip:
                if not detections:
                    valid = False
                    break
                # take the first detection for this bird
                (x0, y0, x1, y1), action_id = detections[0]
                img_path = os.path.join(video_dir, f"frame_{frame_nr:05d}.jpg")
                if not os.path.exists(img_path):
                    valid = False
                    break
                img = Image.open(img_path).convert("RGB")
                crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
                t = pil_to_tensor(crop)
                t = frame_transform(t)
                tensors.append(t)
                clip_actions.append(action_id)
                total_crops += 1

            if not valid or len(tensors) < args.clip_len:
                continue

            # stack and extract I3D
            clip_tensor = torch.stack(tensors, dim=1).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(clip_tensor)
            if out.ndim == 2 and out.size(0) == 1:
                feat = out.squeeze(0).cpu().numpy()
                feat_clips.append(feat)

                # compute majority action_id
                most_common = Counter(clip_actions).most_common(1)[0]
                clip_label = most_common[0]
                label_clips.append(clip_label)

                # print debug
                print(f"[Debug] {video} track{track_id} clip {start}-{end}: "
                      f"actions={clip_actions} -> label={clip_label}")

        if feat_clips:
            feats_arr  = np.stack(feat_clips, axis=0)           # (n_clips, feat_dim)
            labels_arr = np.array(label_clips, dtype=np.int64)  # (n_clips,)

            out_file = os.path.join(args.output_dir, f"{video}_track{track_id}.npz")
            np.savez_compressed(out_file,
                                feats=feats_arr,
                                labels=labels_arr)
            print(f"[Saved] {out_file} → feats{feats_arr.shape} labels{labels_arr.shape}")

# --- Summary ---
print(f"[Info] Extraction complete.")
print(f"[Info] Total tracks processed: {sum(len(tb) for tb in frame_boxes.values())}")
print(f"[Info] Total crops processed: {total_crops}")
print(f"[Info] Files in {args.output_dir}")  
