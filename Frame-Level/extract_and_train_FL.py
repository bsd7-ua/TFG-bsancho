#!/usr/bin/env python3

import os
import cv2
import ast
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

"""
Use:
    python3 extract_a_train_aug.py [--extractor dino] [--clf mlp2]
                                   [--apply_water] [--apply_silhouette]
                                   [--apply_erasing]
"""

parser = argparse.ArgumentParser()
parser.add_argument("--extractor",        type=str, default="dino",
                    help="Feature extractor: dino, resnet, vivit, efficientnet, mobilenet")
parser.add_argument("--clf",              type=str, default="mlp3",
                    help="Classifier type: linear, mlp, mlp2, mlp3")
parser.add_argument("--apply_water",      action="store_true",
                    help="Aplica water background usando la máscara.")
parser.add_argument("--apply_silhouette", action="store_true",
                    help="Aplica silhouette usando la máscara.")
parser.add_argument("--apply_erasing",    action="store_true",
                    help="Aplica random erasing.")
parser.add_argument("--masks_dir",        type=str,
                    default="/workspace/Birds-Classification-main/masks",
                    help="Directorio donde están las máscaras.")
parser.add_argument("--annotations_csv",  type=str,
                    default="/data/annotations/bounding_boxes.csv",
                    help="CSV unificado de bounding boxes.")
parser.add_argument("--frames_dir",       type=str,
                    default="/data/frames",
                    help="Directorio base de frames.")
parser.add_argument("--epochs",           type=int, default=20,
                    help="Épocas para entrenar la capa final.")
args = parser.parse_args()

CACHE_DIR = "./cache_embeddings"
MODEL_DIR = "./models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Init] Device: {device}")
print(f"[Init] Cache dir: {CACHE_DIR}")
print(f"[Init] Model dir: {MODEL_DIR}")

# ─── Split dict ─────────────────────────────────────────────────────────
split_dict = {
    "train_set": ["159-yellow_legged_gull", "095-white_wagtail", "053-eurasian_moorhen", "071-black_winged_stilt", "162-glossy_ibis", "164-white_wagtail", "004-squacco_heron", "149-squacco_heron", "061-northern_shoveler", "038-gadwall", "125-eurasian_magpie", "015-black_headed_gull", "072-black_winged_stilt", "122-black_headed_gull", "033-gadwall", "010-yellow_legged_gull", "013-black_headed_gull", "087-little_ringed_plover", "078-eurasian_magpie", "146-squacco_heron", "144-squacco_heron", "132-eurasian_magpie", "152-squacco_heron", "028-gadwall", "016-black_headed_gull", "145-squacco_heron", "098-white_wagtail", "155-black_winged_stilt", "031-gadwall", "026-black_headed_gull", "091-little_ringed_plover", "043-mallard", "023-glossy_ibis", "046-mallard", "020-eurasian_coot", "133-eurasian_magpie", "035-gadwall", "143-squacco_heron", "123-black_headed_gull", "148-squacco_heron", "056-eurasian_moorhen", "076-black_winged_stilt", "175-eurasian_coot", "111-eurasian_coot", "034-gadwall", "119-eurasian_moorhen", "017-black_headed_gull", "172-black_headed_gull", "151-squacco_heron", "067-northern_shoveler", "118-eurasian_moorhen", "045-mallard", "054-eurasian_moorhen", "168-white_wagtail", "084-yellow_legged_gull", "048-glossy_ibis", "106-eurasian_coot", "110-eurasian_coot", "139-eurasian_coot", "176-black_headed_gull", "040-mallard", "006-yellow_legged_gull", "104-eurasian_coot", "018-eurasian_coot", "030-gadwall", "127-eurasian_magpie", "153-squacco_heron", "044-mallard", "062-northern_shoveler", "008-yellow_legged_gull", "055-eurasian_moorhen", "065-northern_shoveler", "068-northern_shoveler", "093-little_ringed_plover", "086-little_ringed_plover", "097-white_wagtail", "069-northern_shoveler", "050-eurasian_moorhen", "063-northern_shoveler", "094-white_wagtail", "116-eurasian_moorhen", "022-little_ringed_plover", "137-eurasian_magpie", "077-black_winged_stilt", "117-eurasian_moorhen", "081-yellow_legged_gull", "126-eurasian_magpie", "079-eurasian_magpie", "096-white_wagtail", "105-eurasian_coot", "100-eurasian_coot", "011-yellow_legged_gull", "060-northern_shoveler", "112-eurasian_coot", "170-black_winged_stilt", "115-eurasian_moorhen", "101-eurasian_coot", "088-little_ringed_plover", "157-black_winged_stilt", "047-glossy_ibis", "156-black_winged_stilt", "158-black_winged_stilt", "080-yellow_legged_gull", "066-northern_shoveler", "140-mallard", "147-squacco_heron", "032-gadwall", "135-eurasian_magpie", "166-white_wagtail", "141-mallard", "173-black_headed_gull", "059-northern_shoveler", "131-eurasian_magpie", "103-eurasian_coot", "025-black_headed_gull", "113-eurasian_moorhen", "007-yellow_legged_gull", "129-eurasian_magpie", "161-glossy_ibis", "114-eurasian_moorhen", "099-white_wagtail", "169-black_winged_stilt", "037-gadwall", "160-glossy_ibis"],
    "val_set": ["107-eurasian_coot", "090-little_ringed_plover", "051-eurasian_moorhen", "109-eurasian_coot", "005-squacco_heron", "089-little_ringed_plover", "074-black_winged_stilt", "136-eurasian_magpie", "134-eurasian_magpie", "001-white_wagtail", "138-eurasian_magpie", "002-squacco_heron", "070-northern_shoveler", "120-eurasian_moorhen", "102-eurasian_coot", "039-gadwall", "049-glossy_ibis", "014-black_headed_gull", "036-gadwall", "012-black_headed_gull", "058-northern_shoveler", "041-mallard", "128-eurasian_moorhen", "075-black_winged_stilt", "083-yellow_legged_gull", "163-white_wagtail", "085-yellow_legged_gull"],
    "test_set": ["124-black_headed_gull", "142-mallard", "171-black_winged_stilt", "130-eurasian_magpie", "064-northern_shoveler", "024-black_headed_gull", "009-yellow_legged_gull", "177-eurasian_magpie", "052-eurasian_moorhen", "154-northern_shoveler", "150-squacco_heron", "165-glossy_ibis", "019-eurasian_coot", "003-squacco_heron", "057-eurasian_moorhen", "092-little_ringed_plover", "021-little_ringed_plover", "167-white_wagtail", "178-white_wagtail", "082-yellow_legged_gull", "121-eurasian_moorhen", "108-eurasian_coot", "042-mallard", "073-black_winged_stilt", "174-eurasian_coot", "027-gadwall", "029-gadwall"]
}
train_videos = set(split_dict["train_set"])
val_videos   = set(split_dict["val_set"])
test_videos  = set(split_dict["test_set"])

# ─── Carga extractor ────────────────────────────────────────────────────
def load_feature_extractor(name="dino"):
    if name == "dino":
        print("[Extractor] Cargando DINO ViT-B/16…")
        m = torch.hub.load('facebookresearch/dino:main','dino_vitb16', pretrained=True)
        m.to(device).eval()
        dim = getattr(m, "num_features", None) or m(torch.randn(1,3,224,224).to(device)).shape[1]
        return m, dim
    elif name == "resnet":
        print("[Extractor] Cargando ResNet-152…")
        import torchvision.models as models
        m0 = models.resnet152(pretrained=True)
        m = nn.Sequential(*list(m0.children())[:-1])
        m.to(device).eval()
        return m, 2048
    elif name == "vivit":
        print("[Extractor] Cargando ViT vía timm…")
        import timm
        m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        m.to(device).eval()
        dim = getattr(m, "num_features", None) or m(torch.randn(1,3,224,224).to(device)).shape[1]
        return m, dim
    elif name == "efficientnet":
        print("[Extractor] Cargando EfficientNet-B0…")
        import torchvision.models as models
        m0 = models.efficientnet_b0(pretrained=True)
        m = nn.Sequential(m0.features, m0.avgpool)
        m.to(device).eval()
        return m, 1280
    elif name == "mobilenet":
        print("[Extractor] Cargando MobileNet-V3-Large…")
        import torchvision.models as models
        m = models.mobilenet_v3_large(pretrained=True)
        # MobileNet-V3-Large produce 960 canales tras AdaptiveAvgPool2d
        ext = nn.Sequential(m.features, nn.AdaptiveAvgPool2d((1,1)))
        ext.to(device).eval()
        return ext, 960
    else:
        raise ValueError(f"Extractor desconocido: {name}")

feature_extractor, feature_dim = load_feature_extractor(args.extractor)
print(f"[Extractor] Feature dim = {feature_dim}")

# ─── Transforms ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

# ─── Augmentations ──────────────────────────────────────────────────────
def apply_water(crop, mask):
    w = Image.open("/workspace/Birds-Classification-main/crops-clf/agua.png")\
             .convert('RGB').resize(crop.size)
    m = mask.resize(crop.size)
    c_np, w_np, m_np = np.array(crop), np.array(w), (np.array(m)>127)
    comp = w_np.copy(); comp[m_np] = c_np[m_np]
    return Image.fromarray(comp)

def apply_silhouette(crop, mask):
    m = mask.resize(crop.size)
    c_np, mask_np = np.array(crop).copy(), (np.array(m)>127)
    c_np[mask_np] = 0
    return Image.fromarray(c_np)

def apply_erasing(img):
    out = img.copy(); draw = ImageDraw.Draw(out)
    w,h = out.size
    for _ in range(random.randint(1,3)):
        bw = random.randint(1, max(1,w//3))
        bh = random.randint(1, max(1,h//3))
        x0,y0 = random.randint(0,w-bw), random.randint(0,h-bh)
        draw.rectangle([x0,y0,x0+bw,y0+bh], fill=0)
    return out

# ─── Dataset de cajas ───────────────────────────────────────────────────
class BBoxesDataset(Dataset):
    def __init__(self, csv_path, frames_dir, subset="train"):
        df = pd.read_csv(csv_path, sep=";").dropna(subset=["bounding_boxes"])
        self.samples = []
        for _, r in df.iterrows():
            v = str(r["video_name"])
            if subset=="train" and v not in train_videos: continue
            if subset=="val"   and v not in val_videos:   continue
            if subset=="test"  and v not in test_videos:  continue
            f = int(r["frame"])
            if f % 5 != 0: continue
            try: boxes = ast.literal_eval(r["bounding_boxes"])
            except: continue
            img_p = os.path.join(frames_dir, v, f"frame_{f:05d}.jpg")
            if not os.path.exists(img_p): continue
            for i, b in enumerate(boxes):
                if len(b)!=6: continue
                x0,y0,x1,y1 = map(float,b[:4])
                if x1<=x0 or y1<=y0: continue
                self.samples.append({
                    "video":v, "frame":f, "img":img_p,
                    "x0":x0,"y0":y0,"x1":x1,"y1":y1,
                    "label":int(b[4]), "idx":i
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

# ─── Extracción con cache + filtrado + sub-cache ────────────────────────
    """
    Para TRAIN:
        cache_file == master_train_cache
        - Si existe, se carga completo, se filtra según flags y se devuelve.
        - Si no existe, se extrae TODO, se guarda completo y se devuelve.
    Para VAL/TEST:
        - Si existe cache_file, se carga y devuelve.
        - Si no existe, se extrae, se guarda en cache_file y se devuelve.
    """
def extract_with_cache(cache_file, dataset, use_w, use_s, use_e):
    # Determinamos si es el split de entrenamiento por el nombre del archivo
    is_train = os.path.basename(cache_file).startswith("train_")

    # 1) TRAIN: cargar y filtrar maestro
    if is_train and os.path.exists(cache_file):
        print(f"[Cache] Usando maestro → {os.path.basename(cache_file)}")
        d = np.load(cache_file, allow_pickle=True)
        feats_m, labels_m, augs_m = d["feats"], d["labels"], d["augs"]
        keep = ["base"]
        if use_w: keep.append("water")
        if use_s: keep.append("silhouette")
        if use_e: keep.append("erasing")
        mask_glob = np.isin(augs_m, keep)
        feats_f = feats_m[mask_glob]
        labels_f = labels_m[mask_glob]
        print(f"[Cache] Filtrado → {len(feats_f)} muestras")
        return feats_f, labels_f

    # 2) VAL/TEST: cargar cache si existe
    if (not is_train) and os.path.exists(cache_file):
        print(f"[Cache] Cargando {os.path.basename(cache_file)}")
        d = np.load(cache_file, allow_pickle=True)
        return d["feats"], d["labels"]

    # 3) No hay cache: extraer TODO desde cero
    print(f"[Extract] No cache: extrayendo {len(dataset)} muestras…")
    feats, labels, augs = [], [], []
    first_item = True

    for item in tqdm(dataset, desc="  Extrayendo"):
        # 1) crop original
        img = Image.open(item["img"]).convert("RGB")
        x0, y0 = item["x0"], item["y0"]
        x1, y1 = item["x1"], item["y1"]
        crop = img.crop((x0, y0, x1, y1))

        # 2) cargar máscara si procede
        mask = None
        if (use_w or use_s):
            vid, frm = item["video"], item["frame"]
            mask_path = os.path.join(args.masks_dir, vid, f"frame_{frm:05d}_box{item['idx']}.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")

        # 3) preparar lista de (nombre, imagen) para cada augment
        augmented_crops = [("base", crop)]
        if use_w and mask:
            augmented_crops.append(("water", apply_water(crop, mask)))
        if use_s and mask:
            augmented_crops.append(("silhouette", apply_silhouette(crop, mask)))
        if use_e:
            augmented_crops.append(("erasing", apply_erasing(crop)))

        # --- Print de debug tras el primer frame ---
        if first_item:
            aug_names = [name for name, _ in augmented_crops]
            print(f"[Debug] Frame {item['frame']} → augmentaciones generadas: {aug_names}")
            first_item = False

        # 4) extraer un feat independiente por cada augment
        for aug_name, aug_crop in augmented_crops:
            inp = transform(np.array(aug_crop)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(inp).squeeze().cpu().numpy()
            feats.append(feat)
            labels.append(item["label"])
            augs.append(aug_name)

    feats = np.array(feats, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    augs   = np.array(augs,   dtype='<U12')
    print(f"[Extract] Resultado: {len(feats)} embeddings.")

    # Guardar en cache
    if is_train:
        np.savez_compressed(cache_file, feats=feats, labels=labels, augs=augs)
        print(f"[Cache] Maestro guardado en {os.path.basename(cache_file)}")
    else:
        np.savez_compressed(cache_file, feats=feats, labels=labels)
        print(f"[Cache] Guardado en {os.path.basename(cache_file)}")

    return feats, labels



from sklearn.metrics import average_precision_score
import numpy as np

# ─── Entrenador ────────────────────────────────────────────────────────
def train_classifier(Xtr, ytr, Xv, yv, feature_dim, clf_type, epochs):
    print(f"[Train] {clf_type}, epochs={epochs}")
    # --- definir modelo ---
    if clf_type == "mlp3":
        model = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 7)
        )
    elif clf_type == "mlp2":
        model = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(256, 7)
        )
    elif clf_type == "mlp":
        model = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 368), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(368, 7)
        )
    else:
        model = nn.Linear(feature_dim, 7)

    model.to(device)

    # --- label smoothing + optim + scheduler ---
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    sched= optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                                factor=0.5, patience=3,
                                                verbose=True)

    # dataloaders
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xv_t  = torch.tensor(Xv,  dtype=torch.float32)
    yv_t  = torch.tensor(yv,  dtype=torch.long)
    tr = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)
    vl = DataLoader(TensorDataset(Xv_t,  yv_t), batch_size=64, shuffle=False)

    best_map, best_w, no_improve = 0.0, None, 0
    num_classes = 7

    for ep in range(1, epochs+1):
        # --- training step (standard XE) ---
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()

        # --- validation: compute mAP ---
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device)
                out = model(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(yb.numpy())
        all_probs  = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)

        # compute AP per class and mean AP
        onehot = np.zeros((len(all_labels), num_classes), dtype=int)
        onehot[np.arange(len(all_labels)), all_labels] = 1
        aps = []
        for c in range(num_classes):
            if onehot[:, c].sum() == 0:
                aps.append(0.0)
            else:
                aps.append(average_precision_score(onehot[:, c], all_probs[:, c]))
        map_val = np.mean(aps)

        print(f" Ep{ep}/{epochs} | Val mAP={map_val:.3f}")
        sched.step(map_val)

        if map_val > best_map:
            best_map, best_w, no_improve = map_val, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= 5:
                print("[EarlyStopping] sin mejora 5 epochs → STOP")
                break

    if best_w:
        model.load_state_dict(best_w)
    print(f"[Train] Best mAP: {best_map:.3f}")
    return model



# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    # datasets
    train_ds = BBoxesDataset(args.annotations_csv, args.frames_dir, subset="train")
    val_ds   = BBoxesDataset(args.annotations_csv, args.frames_dir, subset="val")
    test_ds  = BBoxesDataset(args.annotations_csv, args.frames_dir, subset="test")

    # flags y sufijo para el nombre de los caches
    want, suffix = set(), ""
    if args.apply_water:      want.add("water");      suffix += "_water"
    if args.apply_silhouette: want.add("silhouette"); suffix += "_silhouette"
    if args.apply_erasing:    want.add("erasing");    suffix += "_erasing"

    # paths de cache
    train_cache = os.path.join(CACHE_DIR, f"train_{args.extractor}_water_silhouette_erasing.npz")
    val_cache   = os.path.join(CACHE_DIR, f"val_{args.extractor}.npz")
    test_cache  = os.path.join(CACHE_DIR, f"test_{args.extractor}.npz")

    # extracción de features (cache o a cero)
    Xtr, ytr = extract_with_cache(train_cache, train_ds,
                                  args.apply_water, args.apply_silhouette, args.apply_erasing)
    Xv,  yv  = extract_with_cache(val_cache,   val_ds, False, False, False)
    Xt,  yt  = extract_with_cache(test_cache,  test_ds, False, False, False)

    # entrenamiento
    clf = train_classifier(Xtr, ytr, Xv, yv, feature_dim, args.clf, args.epochs)

    # guardado del modelo de clasificación
    name = f"{args.extractor}_{args.clf}{suffix}.pth"
    path = os.path.join(MODEL_DIR, name)
    torch.save(clf.state_dict(), path)
    print(f"[Main] Modelo guardado en {path}")
    print(f"[Main] Embeddings guardados en:\n  Train → {train_cache}\n  Val   → {val_cache}\n  Test  → {test_cache}")


if __name__=="__main__":
    main()