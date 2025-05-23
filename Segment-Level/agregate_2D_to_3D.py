#!/usr/bin/env python3
"""
Convierte embeddings frame‑level de DINO (.npz) a clip‑level (.npz)
para PDAN.
"""

import os, argparse, numpy as np
from tqdm import tqdm
from collections import Counter
from extract_a_train_aug import BBoxesDataset

# --------------------------------------------------------------------------- #
def aggregate_split(npz_path, split, csv, frames_dir, out_dir,
                    clip_len, frame_stride, pooling):
    # 1) cargar NPZ DINO y filtrar “base”
    data    = np.load(npz_path, allow_pickle=True)
    mask    = (data["augs"] == "base")
    feats   = data["feats"][mask]                     # (N, D)
    # -------- reconstruir lista de muestras ----------
    ds       = BBoxesDataset(csv, frames_dir, subset=split)
    samples  = ds.samples
    if len(samples) != len(feats):
        raise RuntimeError(f"{split}: mismatch feats({len(feats)}) != samples({len(samples)})")

    # 2) agrupar por video → bird_id → frame
    tracks = {}
    for samp, feat in zip(samples, feats):
        vid   = samp["video"]
        bid   = int(samp["idx"])            # bird_id original
        frm   = int(samp["frame"])
        act   = int(samp["label"])          # acción del frame
        tracks.setdefault(vid, {}) \
              .setdefault(bid, {}) \
              .setdefault(frm,  []) \
              .append((feat, act))

    # 3) slide‑window 16 frames y guardar
    os.makedirs(out_dir, exist_ok=True)
    for vid, birds in tqdm(tracks.items(), desc=f"{split}"):
        for bid, frames_map in birds.items():
            frames_sorted = sorted(frames_map.keys())
            if len(frames_sorted) < clip_len:
                continue

            #  --- un embedding por frame (promedio de crops) ---
            frame_feats  = []
            frame_labels = []
            for fr in frames_sorted:
                items = frames_map[fr]
                frame_feats .append(np.stack([it[0] for it in items]).mean(axis=0))
                frame_labels.append(Counter([it[1] for it in items]).most_common(1)[0][0])
            frame_feats = np.stack(frame_feats, axis=0)           # (N, D)

            #  --- ventana deslizante ---
            clip_feats, clip_labels = [], []
            N = len(frames_sorted)
            for s in range(0, N, frame_stride):
                e = s + clip_len
                if e > N: break
                window_f = frame_feats [s:e]
                window_l = frame_labels[s:e]
                pooled   = window_f.mean(axis=0) if pooling == "mean" else window_f.max(axis=0)
                clip_feats .append(pooled)
                clip_labels.append(Counter(window_l).most_common(1)[0][0])

            if clip_feats:
                feats_arr  = np.stack(clip_feats, axis=0)
                labels_arr = np.array(clip_labels, dtype=np.int64)
                out_name   = f"{vid}_track{bid}.npz"
                np.savez_compressed(os.path.join(out_dir, out_name),
                                    feats=feats_arr,
                                    labels=labels_arr)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",  "-c", default="/workspace/cache_embeddings")
    ap.add_argument("--csv",        "-a", default="/data/annotations/bounding_boxes.csv")
    ap.add_argument("--frames_dir", "-f", default="/data/frames")
    ap.add_argument("--output_dir", "-o", default="./agreg_dino_clip_npz")
    ap.add_argument("--clip_len",    type=int, default=16)
    ap.add_argument("--frame_stride",type=int, default=16)
    ap.add_argument("--pooling",     choices=["mean","max"], default="mean")
    args = ap.parse_args()

    mapping = {"train":"train_dino_water_silhouette_erasing.npz",
               "val"  :"val_dino.npz",
               "test" :"test_dino.npz"}

    for split, fname in mapping.items():
        path = os.path.join(args.cache_dir, fname)
        if not os.path.isfile(path):
            print(f"[Warn] {path} no existe → salto {split}")
            continue
        aggregate_split(path, split,
                        args.csv, args.frames_dir,
                        args.output_dir,
                        args.clip_len, args.frame_stride,
                        args.pooling)

    print("\n[Done] archivos .npz en", args.output_dir)
