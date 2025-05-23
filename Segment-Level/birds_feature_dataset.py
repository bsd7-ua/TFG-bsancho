import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import pandas as pd
import os
import os.path
from tqdm import tqdm
import math
import sys
import ast
import json

# --- Hardcoded Split Dictionary ---
SPLIT_DICT = {
    "train_set": ["159-yellow_legged_gull", "095-white_wagtail", "053-eurasian_moorhen", "071-black_winged_stilt", "162-glossy_ibis", "164-white_wagtail", "004-squacco_heron", "149-squacco_heron", "061-northern_shoveler", "038-gadwall", "125-eurasian_magpie", "015-black_headed_gull", "072-black_winged_stilt", "122-black_headed_gull", "033-gadwall", "010-yellow_legged_gull", "013-black_headed_gull", "087-little_ringed_plover", "078-eurasian_magpie", "146-squacco_heron", "144-squacco_heron", "132-eurasian_magpie", "152-squacco_heron", "028-gadwall", "016-black_headed_gull", "145-squacco_heron", "098-white_wagtail", "155-black_winged_stilt", "031-gadwall", "026-black_headed_gull", "091-little_ringed_plover", "043-mallard", "023-glossy_ibis", "046-mallard", "020-eurasian_coot", "133-eurasian_magpie", "035-gadwall", "143-squacco_heron", "123-black_headed_gull", "148-squacco_heron", "056-eurasian_moorhen", "076-black_winged_stilt", "175-eurasian_coot", "111-eurasian_coot", "034-gadwall", "119-eurasian_moorhen", "017-black_headed_gull", "172-black_headed_gull", "151-squacco_heron", "067-northern_shoveler", "118-eurasian_moorhen", "045-mallard", "054-eurasian_moorhen", "168-white_wagtail", "084-yellow_legged_gull", "048-glossy_ibis", "106-eurasian_coot", "110-eurasian_coot", "139-eurasian_coot", "176-black_headed_gull", "040-mallard", "006-yellow_legged_gull", "104-eurasian_coot", "018-eurasian_coot", "030-gadwall", "127-eurasian_magpie", "153-squacco_heron", "044-mallard", "062-northern_shoveler", "008-yellow_legged_gull", "055-eurasian_moorhen", "065-northern_shoveler", "068-northern_shoveler", "093-little_ringed_plover", "086-little_ringed_plover", "097-white_wagtail", "069-northern_shoveler", "050-eurasian_moorhen", "063-northern_shoveler", "094-white_wagtail", "116-eurasian_moorhen", "022-little_ringed_plover", "137-eurasian_magpie", "077-black_winged_stilt", "117-eurasian_moorhen", "081-yellow_legged_gull", "126-eurasian_magpie", "079-eurasian_magpie", "096-white_wagtail", "105-eurasian_coot", "100-eurasian_coot", "011-yellow_legged_gull", "060-northern_shoveler", "112-eurasian_coot", "170-black_winged_stilt", "115-eurasian_moorhen", "101-eurasian_coot", "088-little_ringed_plover", "157-black_winged_stilt", "047-glossy_ibis", "156-black_winged_stilt", "158-black_winged_stilt", "080-yellow_legged_gull", "066-northern_shoveler", "140-mallard", "147-squacco_heron", "032-gadwall", "135-eurasian_magpie", "166-white_wagtail", "141-mallard", "173-black_headed_gull", "059-northern_shoveler", "131-eurasian_magpie", "103-eurasian_coot", "025-black_headed_gull", "113-eurasian_moorhen", "007-yellow_legged_gull", "129-eurasian_magpie", "161-glossy_ibis", "114-eurasian_moorhen", "099-white_wagtail", "169-black_winged_stilt", "037-gadwall", "160-glossy_ibis"],
    "val_set": ["107-eurasian_coot", "090-little_ringed_plover", "051-eurasian_moorhen", "109-eurasian_coot", "005-squacco_heron", "089-little_ringed_plover", "074-black_winged_stilt", "136-eurasian_magpie", "134-eurasian_magpie", "001-white_wagtail", "138-eurasian_magpie", "002-squacco_heron", "070-northern_shoveler", "120-eurasian_moorhen", "102-eurasian_coot", "039-gadwall", "049-glossy_ibis", "014-black_headed_gull", "036-gadwall", "012-black_headed_gull", "058-northern_shoveler", "041-mallard", "128-eurasian_moorhen", "075-black_winged_stilt", "083-yellow_legged_gull", "163-white_wagtail", "085-yellow_legged_gull"],
    "test_set": ["124-black_headed_gull", "142-mallard", "171-black_winged_stilt", "130-eurasian_magpie", "064-northern_shoveler", "024-black_headed_gull", "009-yellow_legged_gull", "177-eurasian_magpie", "052-eurasian_moorhen", "154-northern_shoveler", "150-squacco_heron", "165-glossy_ibis", "019-eurasian_coot", "003-squacco_heron", "057-eurasian_moorhen", "092-little_ringed_plover", "021-little_ringed_plover", "167-white_wagtail", "178-white_wagtail", "082-yellow_legged_gull", "121-eurasian_moorhen", "108-eurasian_coot", "042-mallard", "073-black_winged_stilt", "174-eurasian_coot", "027-gadwall", "029-gadwall"]
}

def video_to_tensor_pooled(features_np_pooled):
    """Input: (N, C) numpy -> Output: (C, N) tensor"""
    if features_np_pooled.ndim != 2: raise ValueError(f"Expected 2D numpy array, got {features_np_pooled.shape}")
    return torch.from_numpy(features_np_pooled.T.astype(np.float32))

def parse_bbox_string(bbox_str):
    """ Safely evaluates the string representation of list of tuples. """
    try:
        boxes = ast.literal_eval(bbox_str);
        if not isinstance(boxes, list): return []
        valid_boxes = []
        for b in boxes:
            if isinstance(b, tuple) and len(b) >= 5:
                 try: int(b[4]); valid_boxes.append(b)
                 except: continue
        return valid_boxes
    except: return []

def make_dataset_segment(csv_file, split_videos, root, num_classes, feature_dim, clip_len, frame_stride):
    """
    Lee CSV, carga features 2D pre-pooled (.npy), y asigna etiquetas duales.
    NO aplica pooling aquí.
    """
    dataset = []
    try:
        # Carga y validación del CSV
        df_full = pd.read_csv(csv_file, sep=";", dtype={'video_name': str})
        required_cols = ['video_name', 'frame', 'bounding_boxes']
        if not all(col in df_full.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_full.columns];
            print(f"[Error] CSV {csv_file}: Faltan {missing_cols}", file=sys.stderr); return dataset
        df_full['video_name'] = df_full['video_name'].astype(str)
        df_full['frame'] = pd.to_numeric(df_full['frame'], errors='coerce')
        df_full.dropna(subset=['frame', 'bounding_boxes'], inplace=True)
        df_full['frame'] = df_full['frame'].astype(int)
    except Exception as e:
        print(f"[Error] Leyendo CSV {csv_file}: {e}", file=sys.stderr); return dataset

    print(f"Procesando {len(split_videos)} videos definidos para este split...")

    # Pre-parseo de etiquetas
    frame_labels = {}
    print("Parseando etiquetas desde bounding_boxes...")
    for _, row in tqdm(df_full.iterrows(), total=len(df_full), desc="Parsing Labels"):
        video_id = row['video_name']; frame_num = row['frame']
        boxes = parse_bbox_string(row['bounding_boxes'])
        if video_id not in frame_labels: frame_labels[video_id] = {}
        current_labels = set()
        for bbox_tuple in boxes:
            try: current_labels.add(int(bbox_tuple[4]))
            except: continue
        if current_labels:
             if frame_num not in frame_labels[video_id]: frame_labels[video_id][frame_num] = set()
             frame_labels[video_id][frame_num].update(current_labels)

    print(f"Creando dataset final (leyendo features 2D pre-pooled)...")
    processed_count = 0; skipped_load = 0; skipped_dim = 0; skipped_empty = 0

    # Iterar sobre videos del split
    for video_id in tqdm(split_videos):
        feature_path = os.path.join(root, f"{video_id}.npy")
        if not os.path.exists(feature_path): continue

        # --- Cargar y validar features 2D ---
        try:
            features_np_pooled = np.load(feature_path) # Carga directa
            # Validar forma (N_clips, feature_dim)
            if features_np_pooled.ndim != 2 or features_np_pooled.shape[1] != feature_dim:
                # print(f"[Warning] Forma 2D inesperada {features_np_pooled.shape} para {feature_path}. Esperado (N, {feature_dim}). Skipping.", file=sys.stderr)
                skipped_dim += 1; continue
            num_segments = features_np_pooled.shape[0]
            if num_segments == 0: skipped_empty += 1; continue
            # --- NO HAY POOLING AQUÍ ---

        except Exception as e:
            # print(f"[Error] Cargando features 2D desde {feature_path}: {e}. Skipping.", file=sys.stderr)
            skipped_load += 1; continue

        # --- Crear y asignar etiquetas (igual que antes) ---
        segment_labels_idx = np.full(num_segments, -100, dtype=np.int64)
        segment_labels_bin = np.zeros((num_segments, num_classes), dtype=np.float32)
        if video_id in frame_labels:
            video_label_map = frame_labels[video_id];
            for i in range(num_segments):
                start_frame = i * frame_stride; end_frame = start_frame + clip_len
                labels_in_segment = set();
                for frame_num, labels_set in video_label_map.items():
                    if start_frame <= frame_num < end_frame: labels_in_segment.update(labels_set)
                valid_labels = [lbl for lbl in labels_in_segment if 0 <= lbl < num_classes]
                for lbl_idx in valid_labels: segment_labels_bin[i, lbl_idx] = 1.0
                if len(valid_labels) == 1: segment_labels_idx[i] = valid_labels[0]

        # Guardar: id, etiq_indice, etiq_binaria, num_seg, features_pooled (N,C)
        dataset.append((video_id, segment_labels_idx, segment_labels_bin, num_segments, features_np_pooled))
        processed_count += 1

    # Prints de resumen
    print(f"[Debug] Fin bucle. Procesados: {processed_count}, SkipLoad: {skipped_load}, SkipDim: {skipped_dim}, SkipEmpty: {skipped_empty}")
    if not dataset: print("[Warning] Dataset final vacío.", file=sys.stderr)
    return dataset

# --- class BirdsFeatureDataset ---
class BirdsFeatureDataset(data_utl.Dataset):
    """
    Dataset que carga features 2D pre-pooled y genera/devuelve etiquetas duales.
    Usa SPLIT_DICT para filtrar videos.
    """
    def __init__(self, csv_file, split, root, num_classes, feature_dim, clip_len, frame_stride):
        self.num_classes = num_classes; self.feature_dim = feature_dim; self.root = root
        self.split = split; self.csv_file = csv_file; self.clip_len = clip_len; self.frame_stride = frame_stride

        # Seleccionar videos del split
        if split == 'train': self.split_videos = set(SPLIT_DICT.get("train_set", []))
        elif split == 'val': self.split_videos = set(SPLIT_DICT.get("val_set", []))
        elif split == 'test': self.split_videos = set(SPLIT_DICT.get("test_set", []))
        else: sys.exit(f"[Error] Split inválido: '{split}'.")

        if not self.split_videos: self.raw_data = []
        else:
            self.raw_data = make_dataset_segment(
                csv_file, self.split_videos, root, num_classes, feature_dim, clip_len, frame_stride
            )
        print(f"Dataset creado para split '{split}' con {len(self.raw_data)} videos.")
        if len(self.raw_data) == 0 and self.split_videos:
             print(f"[Warning] Dataset para split '{split}' vacío!", file=sys.stderr)

    def __getitem__(self, index):
        """
        Devuelve: features (D, T_seg), labels_idx (T_seg,), labels_bin (T_seg, C), info [vid, T_seg]
        """
        # Acceder a los datos pre-procesados (features ya son 2D pooled)
        video_id, segment_labels_idx, segment_labels_bin, num_segments, features_np_pooled = self.raw_data[index]

        if num_segments == 0:
             return torch.empty((self.feature_dim, 0)), torch.empty((0,)), torch.empty((0, self.num_classes)), [video_id, 0]

        try:
            # Convertir features (N, C) -> (D, N)
            features = video_to_tensor_pooled(features_np_pooled)
            labels_idx = torch.from_numpy(segment_labels_idx.astype(np.int64)) # Long
            labels_bin = torch.from_numpy(segment_labels_bin.astype(np.float32)) # Float
        except Exception as e:
             print(f"[Error] __getitem__ {index} ({video_id}): {e}", file=sys.stderr)
             return torch.empty((self.feature_dim, 0)), torch.empty((0,)), torch.empty((0, self.num_classes)), [video_id, 0]

        return features, labels_idx, labels_bin, [video_id, num_segments]

    def __len__(self):
        return len(self.raw_data)

# --- birds_collate_fn ---
def birds_collate_fn(batch):
    """
    Pads features, labels_idx, labels_bin, and creates masks.
    Espera una tupla de 4 elementos del dataset: (features, labels_idx, labels_bin, info)
    Devuelve una tupla de 5 elementos: (feats_pad, mask, labels_idx_pad, labels_bin_pad, info_list)
    """
    valid_batch = []
    for b in batch: # b = (features, labels_idx, labels_bin, info)
        # Validar basado en features y concordancia de longitudes
        if b[0].numel() > 0 and b[1].numel() > 0 and b[2].numel() > 0 and \
           b[0].shape[1] == b[1].shape[0] and b[0].shape[1] == b[2].shape[0]:
            valid_batch.append(b)
        elif b[0].numel() == 0 and b[1].numel() == 0 and b[2].numel() == 0 and b[3][1] == 0:
             valid_batch.append(b) # Permitir vacíos consistentes
        # else: print("[Debug] Item inválido descartado en collate")

    if not valid_batch: return None, None, None, None, None # 5 Nones
    batch = valid_batch
    max_len_seg = 0

    # Obtener dims del primer elemento válido
    first_valid_feat = batch[0][0]; first_valid_lbl_idx = batch[0][1]; first_valid_lbl_bin = batch[0][2]
    if first_valid_feat.numel() == 0 or first_valid_lbl_idx.numel() == 0 or first_valid_lbl_bin.numel() == 0:
         return None, None, None, None, None
    feature_dim = first_valid_feat.shape[0]; num_classes = first_valid_lbl_bin.shape[1]

    # Encontrar max T_seg
    for b in batch:
        if b[0].shape[1] > max_len_seg: max_len_seg = b[0].shape[1]
    if max_len_seg == 0: return None, None, None, None, None

    # Inicializar tensores con padding
    batch_size = len(batch)
    features_padded = torch.zeros(batch_size, feature_dim, max_len_seg, dtype=torch.float32)
    labels_idx_padded = torch.full((batch_size, max_len_seg), -100, dtype=torch.long)
    labels_bin_padded = torch.zeros(batch_size, max_len_seg, num_classes, dtype=torch.float32)
    masks = torch.zeros(batch_size, 1, max_len_seg, dtype=torch.float32)
    info_list = []

    # Llenar tensores
    for i, b in enumerate(batch):
        feat, lbl_idx, lbl_bin, info = b; t_seg = feat.shape[1]
        if t_seg > 0:
             features_padded[i, :, :t_seg] = feat; labels_idx_padded[i, :t_seg] = lbl_idx
             labels_bin_padded[i, :t_seg, :] = lbl_bin; masks[i, 0, :t_seg] = 1.0
        info_list.append(info)

    # Devuelve: feats_pad, mask, labels_idx_pad, labels_bin_pad, info_list
    return features_padded, masks, labels_idx_padded, labels_bin_padded, info_list