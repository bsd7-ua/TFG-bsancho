#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    average_precision_score
)
import argparse

# ─── Argument Parsing ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Evalúa mAP, accuracy, AP por clase y matriz de confusión sobre Val+Test, Val y Test por separado"
)
parser.add_argument("--extractor",        type=str,   default="dino",
                    help="Extractor: dino, resnet, vivit, efficientnet, mobilenet")
parser.add_argument("--clf",              type=str,   default="mlp3",
                    help="Clasificador: linear, mlp, mlp2, mlp3")
parser.add_argument("--apply_water",      action="store_true",
                    help="Indica que se entrenó con 'water' augment")
parser.add_argument("--apply_silhouette", action="store_true",
                    help="Indica que se entrenó con 'silhouette' augment")
parser.add_argument("--apply_erasing",    action="store_true",
                    help="Indica que se entrenó con 'erasing' augment")
parser.add_argument("--embeddings_dir",   type=str,   default="./cache_embeddings",
                    help="Directorio donde están los .npz de val y test")
parser.add_argument("--models_dir",       type=str,   default="./models",
                    help="Directorio donde están los .pth de los clasificadores")
args = parser.parse_args()

# ─── Construye sufijo de flags para el modelo ────────────────────────
flag = ""
if args.apply_water:      flag += "_water"
if args.apply_silhouette: flag += "_silhouette"
if args.apply_erasing:    flag += "_erasing"

# ─── Determinar feature_dim por extractor ───────────────────────────
ext = args.extractor.lower()
if ext == "dino":
    feature_dim = 768
elif ext == "resnet":
    feature_dim = 2048
elif ext == "vivit":
    feature_dim = 768
elif ext == "efficientnet":
    feature_dim = 1280
elif ext == "mobilenet":
    feature_dim = 960
else:
    raise ValueError(f"Extractor desconocido: {args.extractor}")

# ─── Rutas de los caches y modelo ───────────────────────────────────
val_cache   = os.path.join(args.embeddings_dir, f"val_{ext}.npz")
test_cache  = os.path.join(args.embeddings_dir, f"test_{ext}.npz")
model_file  = os.path.join(args.models_dir,     f"{ext}_{args.clf}{flag}.pth")

# ─── Comprueba existencia de ficheros ───────────────────────────────
for p in (val_cache, test_cache, model_file):
    if not os.path.exists(p):
        print(f"ERROR: no existe '{p}'", file=sys.stderr)
        sys.exit(1)

# ─── Prepara log en fichero ─────────────────────────────────────────
os.makedirs(args.models_dir, exist_ok=True)
log_path = os.path.join(args.models_dir, f"{ext}_{args.clf}{flag}_evaluation.txt")
if os.path.exists(log_path):
    os.remove(log_path)
def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

# ─── Carga embeddings y etiquetas de val y test ─────────────────────
d_val      = np.load(val_cache, allow_pickle=True)
feats_val  = d_val["feats"]
labels_val = d_val["labels"].astype(int)

d_test      = np.load(test_cache, allow_pickle=True)
feats_test  = d_test["feats"]
labels_test = d_test["labels"].astype(int)

log(f"Cargando Val ({len(labels_val)}) + Test ({len(labels_test)}) muestras")

# ─── Concatena ambos splits para evaluar "Total" ────────────────────
feats_all  = np.concatenate([feats_val,  feats_test ], axis=0)
labels_all = np.concatenate([labels_val, labels_test], axis=0)

# ─── Reconstruye el clasificador tal cual se entrenó ────────────────
num_classes = 7
clf_type    = args.clf.lower()

if clf_type == "mlp":
    classifier = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, 368), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(368,            num_classes)
    )
elif clf_type == "mlp2":
    classifier = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, feature_dim), nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(feature_dim, 256), nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(256,       num_classes)
    )
elif clf_type == "mlp3":
    classifier = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
else:
    classifier = nn.Linear(feature_dim, num_classes)

# ─── Carga pesos y pasa a .eval() ──────────────────────────────────
checkpoint = torch.load(model_file, map_location="cpu")
classifier.load_state_dict(checkpoint)
classifier.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# ─── Función de inferencia sobre un array de features ──────────────
def infer(feats_np):
    X = torch.tensor(feats_np, dtype=torch.float32, device=device)
    if X.shape[1] < feature_dim:
        pad = torch.zeros(X.shape[0], feature_dim - X.shape[1], device=device)
        X = torch.cat([X, pad], dim=1)
    with torch.no_grad():
        logits = classifier(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
    return probs, preds

# ─── Inferencia Total, Val y Test ───────────────────────────────────
probs_all,  preds_all  = infer(feats_all)
probs_val,  preds_val  = infer(feats_val)
probs_test, preds_test = infer(feats_test)

# ─── Cálculo y muestra de métricas ──────────────────────────────────
def compute_metrics(labels, preds, probs, split_name):
    acc = accuracy_score(labels, preds) * 100
    onehot = np.zeros((len(labels), num_classes), dtype=int)
    onehot[np.arange(len(labels)), labels] = 1
    aps = []
    for c in range(num_classes):
        if onehot[:, c].sum() == 0:
            aps.append(0.0)
        else:
            aps.append(average_precision_score(onehot[:, c], probs[:, c]))
    mAP = np.mean(aps) * 100

    log(f"\n=== RESULTADOS {split_name} ===")
    log(f"Muestras          : {len(labels)}")
    log(f"Accuracy          : {acc:.2f}%")
    log(f"mAP               : {mAP:.2f}%")
    log(f"AP por clase      : {[f'{ap*100:.2f}%' for ap in aps]}")
    if split_name == "VAL+TEST":
        cm  = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, digits=4)
        log("Matriz de Confusión:")
        log(str(cm))
        log("Reporte de clasificación:")
        log(report)

    return acc, mAP, aps

acc_all,  mAP_all,  aps_all  = compute_metrics(labels_all,  preds_all,  probs_all,  "VAL+TEST")
acc_val,  mAP_val,  aps_val  = compute_metrics(labels_val,  preds_val,  probs_val,  "VAL")
acc_test, mAP_test, aps_test = compute_metrics(labels_test, preds_test, probs_test, "TEST")

# ─── Guardar gráficos de AP por clase ───────────────────────────────
"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for aps, name in [(aps_all, "VAL+TEST")]:
    plt.figure()
    plt.bar(range(num_classes), [ap*100 for ap in aps])
    plt.title(f"AP por clase: {name}")
    plt.xlabel("Clase")
    plt.ylabel("AP (%)")
    plt.tight_layout()
    ap_path = os.path.join(args.models_dir, f"ap_per_class_{name}.png")
    plt.savefig(ap_path)
    plt.close()
    log(f"Gráfico AP por clase ({name}) guardado en: {ap_path}")"""

log(f"\nTodos los resultados también han sido guardados en: {log_path}")
