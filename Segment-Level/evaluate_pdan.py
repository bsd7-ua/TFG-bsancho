#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evalúa PDAN (frame‑level mAP) en val+test.
"""
import os, sys, argparse, numpy as np
from glob import glob; from tqdm import tqdm
import torch; from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PDAN import PDAN; from apmeter import APMeter
import matplotlib.pyplot as plt

CLIP_LEN=16
# ------- splits hardcoded (igual que antes) ------------
split_dict = {
    "train_set": ["159-yellow_legged_gull", "095-white_wagtail", "053-eurasian_moorhen", "071-black_winged_stilt", "162-glossy_ibis", "164-white_wagtail", "004-squacco_heron", "149-squacco_heron", "061-northern_shoveler", "038-gadwall", "125-eurasian_magpie", "015-black_headed_gull", "072-black_winged_stilt", "122-black_headed_gull", "033-gadwall", "010-yellow_legged_gull", "013-black_headed_gull", "087-little_ringed_plover", "078-eurasian_magpie", "146-squacco_heron", "144-squacco_heron", "132-eurasian_magpie", "152-squacco_heron", "028-gadwall", "016-black_headed_gull", "145-squacco_heron", "098-white_wagtail", "155-black_winged_stilt", "031-gadwall", "026-black_headed_gull", "091-little_ringed_plover", "043-mallard", "023-glossy_ibis", "046-mallard", "020-eurasian_coot", "133-eurasian_magpie", "035-gadwall", "143-squacco_heron", "123-black_headed_gull", "148-squacco_heron", "056-eurasian_moorhen", "076-black_winged_stilt", "175-eurasian_coot", "111-eurasian_coot", "034-gadwall", "119-eurasian_moorhen", "017-black_headed_gull", "172-black_headed_gull", "151-squacco_heron", "067-northern_shoveler", "118-eurasian_moorhen", "045-mallard", "054-eurasian_moorhen", "168-white_wagtail", "084-yellow_legged_gull", "048-glossy_ibis", "106-eurasian_coot", "110-eurasian_coot", "139-eurasian_coot", "176-black_headed_gull", "040-mallard", "006-yellow_legged_gull", "104-eurasian_coot", "018-eurasian_coot", "030-gadwall", "127-eurasian_magpie", "153-squacco_heron", "044-mallard", "062-northern_shoveler", "008-yellow_legged_gull", "055-eurasian_moorhen", "065-northern_shoveler", "068-northern_shoveler", "093-little_ringed_plover", "086-little_ringed_plover", "097-white_wagtail", "069-northern_shoveler", "050-eurasian_moorhen", "063-northern_shoveler", "094-white_wagtail", "116-eurasian_moorhen", "022-little_ringed_plover", "137-eurasian_magpie", "077-black_winged_stilt", "117-eurasian_moorhen", "081-yellow_legged_gull", "126-eurasian_magpie", "079-eurasian_magpie", "096-white_wagtail", "105-eurasian_coot", "100-eurasian_coot", "011-yellow_legged_gull", "060-northern_shoveler", "112-eurasian_coot", "170-black_winged_stilt", "115-eurasian_moorhen", "101-eurasian_coot", "088-little_ringed_plover", "157-black_winged_stilt", "047-glossy_ibis", "156-black_winged_stilt", "158-black_winged_stilt", "080-yellow_legged_gull", "066-northern_shoveler", "140-mallard", "147-squacco_heron", "032-gadwall", "135-eurasian_magpie", "166-white_wagtail", "141-mallard", "173-black_headed_gull", "059-northern_shoveler", "131-eurasian_magpie", "103-eurasian_coot", "025-black_headed_gull", "113-eurasian_moorhen", "007-yellow_legged_gull", "129-eurasian_magpie", "161-glossy_ibis", "114-eurasian_moorhen", "099-white_wagtail", "169-black_winged_stilt", "037-gadwall", "160-glossy_ibis"],
    "val_set": ["107-eurasian_coot", "090-little_ringed_plover", "051-eurasian_moorhen", "109-eurasian_coot", "005-squacco_heron", "089-little_ringed_plover", "074-black_winged_stilt", "136-eurasian_magpie", "134-eurasian_magpie", "001-white_wagtail", "138-eurasian_magpie", "002-squacco_heron", "070-northern_shoveler", "120-eurasian_moorhen", "102-eurasian_coot", "039-gadwall", "049-glossy_ibis", "014-black_headed_gull", "036-gadwall", "012-black_headed_gull", "058-northern_shoveler", "041-mallard", "128-eurasian_moorhen", "075-black_winged_stilt", "083-yellow_legged_gull", "163-white_wagtail", "085-yellow_legged_gull"],
    "test_set": ["124-black_headed_gull", "142-mallard", "171-black_winged_stilt", "130-eurasian_magpie", "064-northern_shoveler", "024-black_headed_gull", "009-yellow_legged_gull", "177-eurasian_magpie", "052-eurasian_moorhen", "154-northern_shoveler", "150-squacco_heron", "165-glossy_ibis", "019-eurasian_coot", "003-squacco_heron", "057-eurasian_moorhen", "092-little_ringed_plover", "021-little_ringed_plover", "167-white_wagtail", "178-white_wagtail", "082-yellow_legged_gull", "121-eurasian_moorhen", "108-eurasian_coot", "042-mallard", "073-black_winged_stilt", "174-eurasian_coot", "027-gadwall", "029-gadwall"]
}
# ------- dataset ---------------------------------------
class BirdsClipDS(Dataset):
    def __init__(self, split, root, num_classes):
        vids = set(split_dict[f"{split}_set"])
        self.items = [
            p for p in glob(os.path.join(root, "*.npz"))
            if os.path.basename(p).split("_track")[0] in vids
        ]
        self.C = num_classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = np.load(self.items[idx], allow_pickle=True)
        x = torch.tensor(d["feats"], dtype=torch.float32).t()   # (D,T_clip)
        y = torch.tensor(d["labels_seq"], dtype=torch.long)     # (T_clip,16)
        mask = torch.ones(1, y.shape[0])
        return x, mask, y, os.path.basename(self.items[idx])

def collate(batch):
    xs, ms, ys, names = zip(*batch)
    maxT = max(x.size(1) for x in xs)
    F, M, Y = [], [], []
    for x, m, y in zip(xs, ms, ys):
        pad = maxT - x.size(1)
        F.append(torch.nn.functional.pad(x, (0, pad)))
        M.append(torch.nn.functional.pad(m, (0, pad)))
        Y.append(torch.nn.functional.pad(y, (0, pad), value=-100))
    return torch.stack(F), torch.stack(M), torch.stack(Y), names

def evaluate_split(name, ds, model, device, save_dir, fh):
    if isinstance(ds, ConcatDataset):
        all_items = []
        for sub in ds.datasets:        # each sub is a BirdsClipDS
            all_items.extend(sub.items)
    else:
        all_items = ds.items
    print(f"\n---- Split: {name} ----")
    tracks = [os.path.basename(p) for p in all_items]
    print(f"Tracks ({len(tracks)}): {tracks}\n")

    # also write to file
    fh.write(f"---- Split: {name} ----\n")
    fh.write(f"Tracks ({len(tracks)})\n")

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    apm = APMeter()
    preds, labs = [], []

    with torch.no_grad():
        for F, M, Y, _ in tqdm(loader, desc=name):
            F, M, Y = F.to(device), M.to(device), Y.to(device)
            out = model(F, M)[-1].permute(0, 2, 1)
            logits = out.repeat_interleave(CLIP_LEN, dim=1)
            labels = Y.view(-1)
            mask = labels != -100
            lv = logits.view(-1, 7)[mask]
            lb = labels[mask]
            # one-hot
            lb_bin = torch.zeros(lb.size(0), 7, device=device)
            lb_bin[torch.arange(lb.size(0)), lb] = 1
            apm.add(torch.sigmoid(lv).cpu(), lb_bin.cpu())
            preds.extend(lv.argmax(1).cpu().numpy())
            labs.extend(lb.cpu().numpy())

    # métricas
    ap_per_class = apm.value().numpy()            # (C,)
    map_ = 100 * ap_per_class.mean()
    counts = np.bincount(labs, minlength=7)
    wap = 100 * np.sum(ap_per_class * counts) / counts.sum()
    acc  = accuracy_score(labs, preds) * 100
    f1w  = f1_score(labs, preds, average='weighted') * 100

    summary = f"mAP: {map_:.2f}% | wAP: {wap:.2f}% | Acc: {acc:.2f}% | F1w: {f1w:.2f}%\n"
    print(summary)
    fh.write(summary + "\n")

    # matriz de confusión
    cm = confusion_matrix(labs, preds, labels=list(range(7)))
    print("Matriz de confusión:")
    print(cm)
    fh.write("Matriz de confusión:\n" + np.array2string(cm) + "\n\n")

    # ———- Guardar gráficas ————
    # 1) Confusion matrix heatmap
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Conf Mat: {name}")
    plt.xlabel("Predicciones")
    plt.ylabel("Verdaderos")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"confusion_{name}.png")
    plt.savefig(cm_path)
    plt.close()

    # 2) AP por clase
    plt.figure()
    plt.bar(range(7), ap_per_class)
    plt.title(f"AP por clase: {name}")
    plt.xlabel("Clase")
    plt.ylabel("AP")
    plt.tight_layout()
    ap_path = os.path.join(save_dir, f"ap_per_class_{name}.png")
    plt.savefig(ap_path)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--feature_root", required=True)
    p.add_argument("--feature_dim", type=int, default=1024)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--pdan_stages", type=int, required=True)
    p.add_argument("--pdan_layers", type=int, required=True)
    p.add_argument("--pdan_f_maps", type=int, required=True)
    p.add_argument("--gpu", default='0')
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Infer save_dir del checkpoint
    save_dir = os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(save_dir, exist_ok=True)

    # Abrimos t.txt en modo append
    log_path = os.path.join(save_dir, "eval.txt")
    with open(log_path, "a") as fh:
        fh.write(f"===== Evaluación {args.ckpt} =====\n\n")

        # datasets
        val_ds  = BirdsClipDS('val',  args.feature_root, args.num_classes)
        test_ds = BirdsClipDS('test', args.feature_root, args.num_classes)
        combined_ds = ConcatDataset([val_ds, test_ds])
        # modelo
        model = PDAN(
            args.pdan_stages,
            args.pdan_layers,
            args.pdan_f_maps,
            args.feature_dim,
            args.num_classes
        ).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

        # evaluaciones
        evaluate_split("Validation", val_ds,    model, device, save_dir, fh)
        evaluate_split("Test",       test_ds,   model, device, save_dir, fh)
        evaluate_split("Combined",   combined_ds, model, device, save_dir, fh)

    print(f"\nResultados guardados en {log_path} y gráficas en {save_dir}")

if __name__ == "__main__":
    main()