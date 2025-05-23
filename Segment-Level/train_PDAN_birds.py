#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena PDAN sobre .npz (no‑overlap, clip_len=16) con mAP scheduler.
"""
from __future__ import division
import os, sys, time, copy, random, argparse
from glob import glob
import numpy as np;  from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.autograd import Variable
from PDAN import PDAN
from apmeter import APMeter
# ─────────────────────────────────────────────────────────
# DATASET
# ---------------------------------------------------------
CLIP_LEN = 16   # fijo (=stride)
class BirdsClipDataset(Dataset):
    def __init__(self, split, feature_root, num_classes, split_dict):
        vids_ok = set(split_dict[f"{split}_set"])
        files   = [p for p in glob(os.path.join(feature_root,"*.npz"))
                   if os.path.basename(p).split("_track")[0] in vids_ok]
        self.tracks=[]
        for p in files:
            d=np.load(p,allow_pickle=True)
            feats  = d["feats"]                   # (T,D)
            labels = d["labels"]                  # (T,)  label mayoritaria
            T      = feats.shape[0]
            mask   = np.ones((T,),np.float32)
            lbl_b  = np.zeros((T,num_classes),np.float32)
            lbl_b[np.arange(T),labels]=1.0
            self.tracks.append(dict(
                feats=torch.tensor(feats,dtype=torch.float32).t(), # (D,T)
                mask =torch.tensor(mask[None],dtype=torch.float32),# (1,T)
                lbl_i=torch.tensor(labels,dtype=torch.long),
                lbl_b=torch.tensor(lbl_b,dtype=torch.float32)
            ))
    def __len__(self): return len(self.tracks)
    def __getitem__(s,idx):
        t=s.tracks[idx]; return t["feats"],t["mask"],t["lbl_i"],t["lbl_b"],None

def birds_collate(batch):
    feats,masks,lbl_i,lbl_b,_=zip(*batch)
    maxT=max(f.size(1) for f in feats); D=feats[0].size(0); C=lbl_b[0].size(1); N=len(batch)
    F=torch.zeros(N,D,maxT); M=torch.zeros(N,1,maxT)
    I=torch.full((N,maxT),-100,dtype=torch.long); B=torch.zeros(N,maxT,C)
    for i,(f,m,li,lb) in enumerate(zip(feats,masks,lbl_i,lbl_b)):
        T=f.size(1); F[i,:,:T]=f; M[i,:,:T]=m; I[i,:T]=li; B[i,:T]=lb
    return F,M,I,B,None
# ─────────────────────────────────────────────────────────
# ARG PARSER
# ---------------------------------------------------------
def str2bool(v): return v.lower() in ('yes','true','t','1')
parser=argparse.ArgumentParser("Train PDAN frame‑level (mAP scheduler)")
parser.add_argument("--feature_root",type=str,required=True)
parser.add_argument("--num_classes",type=int,required=True)
parser.add_argument("--feature_dim",type=int,required=True)
parser.add_argument("--pdan_stages",type=int,default=1)
parser.add_argument("--pdan_layers",type=int,default=5)
parser.add_argument("--pdan_f_maps",type=int,default=512)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--patience",type=int,default=8)
parser.add_argument("--weight_decay",type=float,default=5e-4)
parser.add_argument("--gpu",type=str,default='0')
parser.add_argument("--save_dir",type=str,default="./checkpoints_birds")
parser.add_argument("--run_name",type=str,default="pdan_framelevel")
parser.add_argument("--ap_type",type=str,choices=["map","wap"],default="map",
                    help="'map'=media de clases (paper), 'wap'=sin promediar (weighted‑AP)")
args=parser.parse_args()

# splits hardcoded ----------------------------------------------------
split_dict = {
    "train_set": ["159-yellow_legged_gull", "095-white_wagtail", "053-eurasian_moorhen", "071-black_winged_stilt", "162-glossy_ibis", "164-white_wagtail", "004-squacco_heron", "149-squacco_heron", "061-northern_shoveler", "038-gadwall", "125-eurasian_magpie", "015-black_headed_gull", "072-black_winged_stilt", "122-black_headed_gull", "033-gadwall", "010-yellow_legged_gull", "013-black_headed_gull", "087-little_ringed_plover", "078-eurasian_magpie", "146-squacco_heron", "144-squacco_heron", "132-eurasian_magpie", "152-squacco_heron", "028-gadwall", "016-black_headed_gull", "145-squacco_heron", "098-white_wagtail", "155-black_winged_stilt", "031-gadwall", "026-black_headed_gull", "091-little_ringed_plover", "043-mallard", "023-glossy_ibis", "046-mallard", "020-eurasian_coot", "133-eurasian_magpie", "035-gadwall", "143-squacco_heron", "123-black_headed_gull", "148-squacco_heron", "056-eurasian_moorhen", "076-black_winged_stilt", "175-eurasian_coot", "111-eurasian_coot", "034-gadwall", "119-eurasian_moorhen", "017-black_headed_gull", "172-black_headed_gull", "151-squacco_heron", "067-northern_shoveler", "118-eurasian_moorhen", "045-mallard", "054-eurasian_moorhen", "168-white_wagtail", "084-yellow_legged_gull", "048-glossy_ibis", "106-eurasian_coot", "110-eurasian_coot", "139-eurasian_coot", "176-black_headed_gull", "040-mallard", "006-yellow_legged_gull", "104-eurasian_coot", "018-eurasian_coot", "030-gadwall", "127-eurasian_magpie", "153-squacco_heron", "044-mallard", "062-northern_shoveler", "008-yellow_legged_gull", "055-eurasian_moorhen", "065-northern_shoveler", "068-northern_shoveler", "093-little_ringed_plover", "086-little_ringed_plover", "097-white_wagtail", "069-northern_shoveler", "050-eurasian_moorhen", "063-northern_shoveler", "094-white_wagtail", "116-eurasian_moorhen", "022-little_ringed_plover", "137-eurasian_magpie", "077-black_winged_stilt", "117-eurasian_moorhen", "081-yellow_legged_gull", "126-eurasian_magpie", "079-eurasian_magpie", "096-white_wagtail", "105-eurasian_coot", "100-eurasian_coot", "011-yellow_legged_gull", "060-northern_shoveler", "112-eurasian_coot", "170-black_winged_stilt", "115-eurasian_moorhen", "101-eurasian_coot", "088-little_ringed_plover", "157-black_winged_stilt", "047-glossy_ibis", "156-black_winged_stilt", "158-black_winged_stilt", "080-yellow_legged_gull", "066-northern_shoveler", "140-mallard", "147-squacco_heron", "032-gadwall", "135-eurasian_magpie", "166-white_wagtail", "141-mallard", "173-black_headed_gull", "059-northern_shoveler", "131-eurasian_magpie", "103-eurasian_coot", "025-black_headed_gull", "113-eurasian_moorhen", "007-yellow_legged_gull", "129-eurasian_magpie", "161-glossy_ibis", "114-eurasian_moorhen", "099-white_wagtail", "169-black_winged_stilt", "037-gadwall", "160-glossy_ibis"],
    "val_set": ["107-eurasian_coot", "090-little_ringed_plover", "051-eurasian_moorhen", "109-eurasian_coot", "005-squacco_heron", "089-little_ringed_plover", "074-black_winged_stilt", "136-eurasian_magpie", "134-eurasian_magpie", "001-white_wagtail", "138-eurasian_magpie", "002-squacco_heron", "070-northern_shoveler", "120-eurasian_moorhen", "102-eurasian_coot", "039-gadwall", "049-glossy_ibis", "014-black_headed_gull", "036-gadwall", "012-black_headed_gull", "058-northern_shoveler", "041-mallard", "128-eurasian_moorhen", "075-black_winged_stilt", "083-yellow_legged_gull", "163-white_wagtail", "085-yellow_legged_gull"],
    "test_set": ["124-black_headed_gull", "142-mallard", "171-black_winged_stilt", "130-eurasian_magpie", "064-northern_shoveler", "024-black_headed_gull", "009-yellow_legged_gull", "177-eurasian_magpie", "052-eurasian_moorhen", "154-northern_shoveler", "150-squacco_heron", "165-glossy_ibis", "019-eurasian_coot", "003-squacco_heron", "057-eurasian_moorhen", "092-little_ringed_plover", "021-little_ringed_plover", "167-white_wagtail", "178-white_wagtail", "082-yellow_legged_gull", "121-eurasian_moorhen", "108-eurasian_coot", "042-mallard", "073-black_winged_stilt", "174-eurasian_coot", "027-gadwall", "029-gadwall"]
}
# device / seed -------------------------------------------------------
gpu_ids=[int(g) for g in args.gpu.split(',') if g.isdigit()]
device=torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() and gpu_ids else 'cpu')
torch.manual_seed(0); np.random.seed(0)
print("[Device]",device)

# data loaders --------------------------------------------------------
train_ds=BirdsClipDataset('train',args.feature_root,args.num_classes,split_dict)
val_ds  =BirdsClipDataset('val',  args.feature_root,args.num_classes,split_dict)
train_loader=torch.utils.data.DataLoader(train_ds,batch_size=args.batch_size,
        shuffle=True,num_workers=4,collate_fn=birds_collate,pin_memory=device.type=='cuda')
val_loader  =torch.utils.data.DataLoader(val_ds,batch_size=1,shuffle=False,
        num_workers=2,collate_fn=birds_collate,pin_memory=device.type=='cuda')
print(f"[Data] train {len(train_ds)} tracks | val {len(val_ds)} tracks")

# model / optim -------------------------------------------------------
model=PDAN(args.pdan_stages,args.pdan_layers,args.pdan_f_maps,
           args.feature_dim,args.num_classes).to(device)
criterion=nn.CrossEntropyLoss(ignore_index=-100,label_smoothing=0.1)
optim=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
sched=lr_scheduler.ReduceLROnPlateau(optim,mode='max',patience=args.patience,
                                     factor=0.5,verbose=True)
apm=APMeter(); 
best_map, best_wts, no_improve = 0.0, None, 0

# helpers -------------------------------------------------------------
def step(loader,train=True):
    model.train(train); apm.reset(); tot_loss=0; n=0
    for F,M,I,B,_ in loader:
        F,M,I,B=F.to(device),M.to(device),I.to(device),B.to(device)
        if train: optim.zero_grad()
        logits=model(F,M)[-1].permute(0,2,1)       # (B,T,C)
        valid=M.squeeze(1).bool()
        lv,li,lb=logits[valid],I[valid],B[valid]
        loss=criterion(lv,li)
        if train: loss.backward(); optim.step()
        apm.add(torch.sigmoid(lv.detach()).cpu(),  lb.detach().cpu())
        tot_loss+=loss.item(); n+=1
    m=apm.value(); map_val=100*m.mean().item() if args.ap_type=="map" else 100*m.item()
    return map_val, tot_loss/max(n,1)

# ------------- logging  lists-------
log_train_map, log_val_map, log_train_loss, log_val_loss = [], [], [], []

# training loop -------------------------------------------------------
os.makedirs(args.save_dir,exist_ok=True)
for epoch in range(args.epochs):
    tr_map,tr_loss=step(train_loader,True)
    val_map,val_loss=step(val_loader,False)
    log_train_map.append(tr_map); log_val_map.append(val_map)
    log_train_loss.append(tr_loss); log_val_loss.append(val_loss)

    sched.step(val_map)
    print(f"Epoch {epoch:03d} | Train mAP {tr_map:.2f} | Val mAP {val_map:.2f}")
    if val_map>best_map:
        best_map, best_wts = val_map, copy.deepcopy(
              (model.module if isinstance(model, nn.DataParallel) else model).state_dict())
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f"{args.run_name}_best_map.pt"))
        no_improve = 0
        print("  ↳ saved best checkpoint")
    else:
        no_improve += 1
        # ---- early‑stopping ----
    if no_improve >= 25:
        print(f"[EarlyStop] {no_improve} epochs sin mejorar mAP. Fin de entrenamiento.")
        break
print("Best Val mAP:",best_map)

# --------- (3) guardar CSV -------------------
import pandas as pd, matplotlib
matplotlib.use("Agg")               # backend “headless”
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "epoch": np.arange(1, len(log_train_map)+1),
    "train_map":  log_train_map,
    "val_map":    log_val_map,
    "train_loss": log_train_loss,
    "val_loss":   log_val_loss,
})
csv_path = os.path.join(args.save_dir, f"{args.run_name}_metrics.csv")
df.to_csv(csv_path, index=False)
print("[Log] CSV ->", csv_path)

# --------- (4) generar gráficos --------------
def _save_curve(y1, y2, ylabel, fname):
    fig, ax = plt.subplots()
    ax.plot(df.epoch, y1, label="Train")
    ax.plot(df.epoch, y2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    out_path = os.path.join(args.save_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Log] Figura ->", out_path)

_save_curve(df.train_map,  df.val_map,  "mAP (%)",
            f"{args.run_name}_map_curve.png")
_save_curve(df.train_loss, df.val_loss, "Loss",
            f"{args.run_name}_loss_curve.png")
