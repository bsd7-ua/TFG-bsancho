#!/usr/bin/env bash

# ---------- Datos ------------------------------------------------
FEATURE_ROOT="./r2plus1d_features_for_pdan"   # .npz con labels_seq
NUM_CLASSES=7
FEATURE_DIM=1024

# ---------- Modelo ----------------------------------------------
PDAN_STAGES=1
PDAN_LAYERS=5
PDAN_F_MAPS=512

# ---------- Entrenamiento ---------------------------------------
EPOCHS=200
BATCH_SIZE=64
LR=1e-4
PATIENCE=6
W_DECAY=5e-5
GPU_ID="0"
NUM_WORKERS=4

# Fase 1: ablation arquitectónica (stages, layers, f_maps)
RUN_NAME="pdan_r2plus1d_${PDAN_STAGES}_l${PDAN_LAYERS}_f${PDAN_F_MAPS}"

# Fase 2: ajuste de hiperparámetros (batch_size, lr, weight_decay, patience)
#RUN_NAME="pdan_r2plus1d_${BATCH_SIZE}_lr${LR}_wd${W_DECAY}_pt${PATIENCE}"

SAVE_DIR="./checkpoints_birds/fase1_r2/${RUN_NAME}"
#SAVE_DIR="./checkpoints_birds/fase2_r2/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"

echo "=============================================================="
echo " PDAN FRAME-LEVEL TRAINING"
echo " Feature root : ${FEATURE_ROOT}"
echo " Save dir     : ${SAVE_DIR}"
echo "=============================================================="

python train_PDAN_birds.py \
  --feature_root   "${FEATURE_ROOT}" \
  --num_classes    "${NUM_CLASSES}" \
  --feature_dim    "${FEATURE_DIM}" \
  --pdan_stages    "${PDAN_STAGES}" \
  --pdan_layers    "${PDAN_LAYERS}" \
  --pdan_f_maps    "${PDAN_F_MAPS}" \
  --epochs         "${EPOCHS}" \
  --batch_size     "${BATCH_SIZE}" \
  --lr             "${LR}" \
  --patience       "${PATIENCE}" \
  --weight_decay   "${W_DECAY}" \
  --gpu            "${GPU_ID}" \
  --save_dir       "${SAVE_DIR}" \
  --run_name       "${RUN_NAME}" \
  --ap_type        "map"          # ó "wap" si quieres weighted-AP

echo "Training completed."
echo "=============================================================="

# ---------- Evaluación ------------------------------------------
CHECKPOINT="${SAVE_DIR}/${RUN_NAME}_best_map.pt"

echo "=============================================================="
echo "  PDAN FRAME-LEVEL EVALUATION"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  Feature root: ${FEATURE_ROOT}"
echo "=============================================================="

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: checkpoint no encontrado en ${CHECKPOINT}"
  exit 1
fi
if [[ ! -d "${FEATURE_ROOT}" ]]; then
  echo "Error: features no encontradas en ${FEATURE_ROOT}"
  exit 1
fi

python evaluate_pdan.py \
  --ckpt          "${CHECKPOINT}" \
  --feature_root  "${FEATURE_ROOT}" \
  --feature_dim   "${FEATURE_DIM}" \
  --num_classes   "${NUM_CLASSES}" \
  --pdan_stages    "${PDAN_STAGES}" \
  --pdan_layers    "${PDAN_LAYERS}" \
  --pdan_f_maps    "${PDAN_F_MAPS}" \
  --gpu           "${GPU_ID}"

echo "Evaluation done."