#!/usr/bin/env python3
# extract_features.py
"""
Extrae features clip-level (16 frames, stride = 16) con un único backbone
preentrenado en Kinetics y guarda cada pista en:
    <video>_track<i>.npz  con:
        feats       (n_clips , feat_dim)
        labels      (n_clips,)
        labels_seq  (n_clips , 16)
Backbones:
    • r2plus1d  : R(2+1)D-R50 (K400)  – 112×112
    • mvit16x4  : MViT-B 16×4 (K400) – 224×224
"""
import os, ast, argparse
import numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image

import torch, torch.nn as nn
import torchvision.transforms as T
from pytorchvideo.models.hub import r2plus1d_r50, mvit_base_16x4

# ---------- CLI ---------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--csv", type=str,
               default="/data/annotations/bounding_boxes.csv")
p.add_argument("--frames_dir", type=str,
               default="/data/frames")
p.add_argument("--output_dir", default="./r2plus1d_features_for_pdan_2")
p.add_argument("--clip_len",     type=int, default=16)
p.add_argument("--frame_stride", type=int, default=16)
p.add_argument("--backbone",     type=str, required=True,
               choices=["r2plus1d", "mvit16x4"])
args = p.parse_args()

CLIP_LEN = args.clip_len
STRIDE   = args.frame_stride
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- cargar backbone --------------------------------------
if args.backbone == "r2plus1d":
    print("[Info] Cargando R(2+1)D-R50…")
    net = r2plus1d_r50(pretrained=True).to(device).eval()
    backbone = nn.Sequential(*net.blocks[:-1]).to(device).eval()
    feat_dim = 2048

    def extract_feat(x):
        with torch.no_grad():
            fmap = backbone(x)           # (B,2048,T',H',W')
            return fmap.mean([2, 3, 4])  # (B,2048)

    tfm = T.Compose([
        T.Resize(128), T.CenterCrop(112),
        T.Normalize([0.45]*3, [0.225]*3)
    ])

elif args.backbone == "mvit16x4":
    print("[Info] Cargando MViT-B 16×4…")
    backbone = mvit_base_16x4(pretrained=True).to(device).eval()

    # --- quitar genéricamente la última capa lineal (cabeza) ----
    # 1) recopilar todos los nn.Linear del modelo
    linears = [(name, module)
               for name, module in backbone.named_modules()
               if isinstance(module, nn.Linear)]
    # 2) seleccionar el último y extraer su in_features
    last_name, last_linear = linears[-1]
    feat_dim = last_linear.in_features
    # 3) navegar hasta su módulo padre y reemplazarlo por Identity
    parent_name, _, child_name = last_name.rpartition('.')
    parent = backbone if parent_name == "" else eval(f"backbone.{parent_name}")
    setattr(parent, child_name, nn.Identity())
    print(f"[Info] MViT feature dim ajustada = {feat_dim}")
    # --------------------------------------------------------------

    def extract_feat(x):
        with torch.no_grad():
            return backbone(x)   # ahora devuelve el embedding (B, feat_dim)

    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),   # resolución oficial MViT
        T.Normalize([0.45]*3, [0.225]*3)
    ])


print(f"[Info] Feature dim = {feat_dim or 'pendiente de primer clip'}")

# ---------- leer CSV y agrupar por pista -------------------------
print("[Info] Leyendo CSV…")
df = pd.read_csv(args.csv, sep=";")
frame_boxes = {}
for _, row in df.dropna(subset=['bounding_boxes']).iterrows():
    vid = row['video_name']; f = int(row['frame'])
    for bb in ast.literal_eval(row['bounding_boxes']):
        if len(bb) < 6: continue
        act, track = int(bb[4]), int(bb[5])
        frame_boxes.setdefault((vid, track), {})[f] = act

os.makedirs(args.output_dir, exist_ok=True)
to_tensor = T.ToTensor()

# ---------- procesar cada pista ----------------------------------
for (vid, track), fdict in tqdm(frame_boxes.items(), desc="Pistas"):
    frames_dir = os.path.join(args.frames_dir, vid)
    if not os.path.isdir(frames_dir):
        continue

    frames_sorted = sorted(fdict.items())
    frame_idx = [f for f, _ in frames_sorted]
    acts      = [a for _, a in frames_sorted]

    feats_list, labels_clip, labels_seq = [], [], []

    for start in range(0, len(frame_idx) - CLIP_LEN + 1, STRIDE):
        idxs   = frame_idx[start:start+CLIP_LEN]
        acts16 = acts[start:start+CLIP_LEN]

        # cargar frames
        imgs = []
        for fi in idxs:
            path = os.path.join(frames_dir, f"frame_{fi:05d}.jpg")
            if not os.path.exists(path):
                break
            img = tfm(to_tensor(Image.open(path).convert("RGB")))
            imgs.append(img)
        if len(imgs) < CLIP_LEN:
            continue

        clip = torch.stack(imgs, 0).permute(1,0,2,3).unsqueeze(0).to(device)
        feat = extract_feat(clip).squeeze(0).cpu().numpy()

        if feat_dim is None:
            feat_dim = feat.size
            print(f"[Info] Feature dim detectada = {feat_dim}")

        feats_list.append(feat)
        labels_seq.append(acts16)
        labels_clip.append(max(set(acts16), key=acts16.count))

    if feats_list:
        out_path = os.path.join(args.output_dir, f"{vid}_track{track}.npz")
        np.savez_compressed(
            out_path,
            feats=np.stack(feats_list, 0),
            labels=np.array(labels_clip, dtype=np.int64),
            labels_seq=np.array(labels_seq, dtype=np.int64)
        )
        print("   ↳", os.path.basename(out_path),
              f"{args.backbone}={len(feats_list)}×{feat_dim}")

print("[Done] Features en", args.output_dir)
