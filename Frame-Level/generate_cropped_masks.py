#!/usr/bin/env python3

import os
import time
import argparse
import pandas as pd
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
import numpy as np
from transformers import SamModel, SamProcessor
from skimage.morphology import remove_small_holes, disk, binary_closing, remove_small_objects
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
from torch.utils.data import Dataset

def refine_mask_with_skimage(mask_np, closing_radius=5, min_obj_size=100, hole_area_threshold=500):
    """
    Aplica una serie de operaciones morfológicas y rellena agujeros pequeños
    en la máscara, devolviendo una máscara binaria refinada en formato uint8.
    """
    mask_bool = mask_np > 127
    selem = disk(closing_radius)
    mask_closed = binary_closing(mask_bool, footprint=selem)
    mask_filled = binary_fill_holes(mask_closed)
    mask_no_small_holes = remove_small_holes(mask_filled, area_threshold=hole_area_threshold)
    mask_final = remove_small_objects(mask_no_small_holes, min_size=min_obj_size)
    return (mask_final.astype(np.uint8) * 255)


class MaskGenerationDataset(Dataset):
    def __init__(self, csv_file, frames_dir, masks_output_root, test_mode=False, max_samples_test=50):
        self.frames_dir = frames_dir
        self.masks_output_root = masks_output_root
        self.samples = []  # Almacena tuplas (image_path, bbox_coords_4, output_path)

        print("Leyendo archivo CSV y preparando muestras (según formato BirdsDataset)...")
        try:
            # Leer CSV asegurando que la columna problemática se trate como string
            df = pd.read_csv(csv_file, sep=";", dtype={'bounding_boxes': str})
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo CSV en {csv_file}")
            raise
        except Exception as e:
            print(f"Error al leer el archivo CSV {csv_file}: {e}")
            raise

        # Contadores para diagnóstico
        num_processed_bboxes = 0
        num_rows_without_bboxes = 0
        num_skipped_wrong_coord_count = 0  # Esperando 6 coords
        num_skipped_conversion_errors = 0
        num_skipped_invalid_coords = 0
        num_frames_not_found = 0

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Procesando CSV"):
            video_name = row['video_name']
            frame_num = row['frame']
            
            # --- Downsampling de 1 de cada 5 frames ---
            if frame_num % 5 != 0:
                continue  # Salta este frame si no es múltiplo de 5
            
            image_path = os.path.join(self.frames_dir, str(video_name), f"frame_{str(frame_num).zfill(5)}.jpg")

            # Verificar si existe el frame ANTES de procesar bboxes
            if not os.path.exists(image_path):
                num_frames_not_found += 1
                continue  # Saltar fila si el frame no existe

            bboxes_str_raw = row['bounding_boxes']

            # Verificar si la celda es NaN, None o una cadena vacía/solo espacios
            if pd.isna(bboxes_str_raw) or not str(bboxes_str_raw).strip():
                num_rows_without_bboxes += 1
                continue  # Saltar fila si no hay anotación

            # Aplicar limpieza y split exactamente como en BirdsDataset
            bboxes_str = str(bboxes_str_raw).strip("[(").strip(')]')

            if isinstance(bboxes_str, str) and bboxes_str.strip() != "":
                 # Split basado en el delimitador de los ejemplos
                 bbox_list = bboxes_str.split("), (")
            else:
                 bbox_list = []  # La cadena quedó vacía después del strip o ya estaba vacía

            # Procesar cada string de bounding box individual
            for bbox_idx, bbox_str in enumerate(bbox_list):
                # Dividir por coma
                coords = bbox_str.split(",")

                # Verificar que hay exactamente 6 coordenadas
                if len(coords) != 6:
                    num_skipped_wrong_coord_count += 1
                    continue  # Saltar este bbox

                try:
                    # Convertir las 6 coordenadas a float
                    all_coords_float = tuple(map(float, map(str.strip, coords)))

                    # Extraer las primeras 4 para la geometría del crop
                    x0, y0, x1, y1 = all_coords_float[:4]

                    # Validación básica de coordenadas
                    if x1 <= x0 or y1 <= y0:
                        num_skipped_invalid_coords += 1
                        continue  # Saltar este bbox

                    # Generar ruta de salida y añadir a la lista
                    out_dir = os.path.join(self.masks_output_root, str(video_name))
                    out_name = f"frame_{str(frame_num).zfill(5)}_box{bbox_idx}.png"
                    out_path = os.path.join(out_dir, out_name)

                    self.samples.append((image_path, (x0, y0, x1, y1), out_path))
                    num_processed_bboxes += 1

                    # Aplicar límite de muestras en test_mode
                    if test_mode and len(self.samples) >= max_samples_test:
                        break

                except ValueError:
                    # Error al convertir a float
                    num_skipped_conversion_errors += 1
                    continue

            # Salir del bucle de filas si se alcanzó el límite en modo test
            if test_mode and len(self.samples) >= max_samples_test:
                break

        # Imprimir resumen diagnóstico
        print(f"\n--- Resumen del Procesamiento del CSV ---")
        print(f"Dataset inicializado.")
        print(f"Total de máscaras a generar (muestras creadas): {len(self.samples)}")
        print(f"Bounding Boxes procesados exitosamente: {num_processed_bboxes}")
        print(f"Filas omitidas por no encontrar el frame: {num_frames_not_found}")
        print(f"Filas omitidas por no tener anotación de BBox (NaN o vacío): {num_rows_without_bboxes}")
        print(f"BBoxes individuales omitidos por tener != 6 coordenadas: {num_skipped_wrong_coord_count}")
        print(f"BBoxes individuales omitidos por error de conversión a float: {num_skipped_conversion_errors}")
        print(f"BBoxes individuales omitidos por coordenadas inválidas (x1<=x0 o y1<=y0): {num_skipped_invalid_coords}")
        print(f"----------------------------------------")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Espera (image_path, bbox_coords_4, out_path) de self.samples
        image_path, bbox_coords_4, out_path = self.samples[idx]
        try:
            full_img = Image.open(image_path).convert('RGB')
            x0, y0, x1, y1 = bbox_coords_4
            img_w, img_h = full_img.size
            # Clamp coordinates (ajuste a límites de la imagen)
            x0_c, y0_c = max(0, x0), max(0, y0)
            x1_c, y1_c = min(img_w, x1), min(img_h, y1)

            if x1_c <= x0_c or y1_c <= y0_c:
                return {"crop": None, "out_path": out_path,
                        "error": f"Invalid BBox dimensions after clamping for {image_path}: "
                                 f"Orig({x0},{y0},{x1},{y1}) -> Clamped({x0_c},{y0_c},{x1_c},{y1_c})"}

            cropped_img = full_img.crop((x0_c, y0_c, x1_c, y1_c))

            if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
                return {"crop": None, "out_path": out_path,
                        "error": f"Crop resulted in zero dimension for {image_path}"}

            return {"crop": cropped_img, "out_path": out_path, "error": None}

        except FileNotFoundError:
            return {"crop": None, "out_path": out_path,
                    "error": f"FileNotFoundError: {image_path}"}
        except UnidentifiedImageError:
            return {"crop": None, "out_path": out_path,
                    "error": f"UnidentifiedImageError: {image_path}"}
        except Exception as e:
            # Añadir más detalles al error si es posible
            return {"crop": None, "out_path": out_path,
                    "error": f"Unexpected error in __getitem__ for {image_path} (coords: {bbox_coords_4}): {e}"}


def main():
    parser = argparse.ArgumentParser(description="Genera máscaras de segmentación usando SAM de forma secuencial y lazy.")
    parser.add_argument('--csv_file', type=str, default="/workspace/crops-clf/data-behavs/train_bounding_boxes.csv",
                        help="CSV con las anotaciones.")
    parser.add_argument('--frames_dir', type=str, default="/data/frames",
                        help="Directorio de frames organizados por video.")
    parser.add_argument('--masks_dir', type=str, default="/workspace/masks",
                        help="Directorio de salida para las máscaras (dataset completo).")
    parser.add_argument('--sam_checkpoint', type=str, default="facebook/sam-vit-huge",
                        help="Checkpoint de SAM.")
    parser.add_argument('--test_mode', action="store_true",
                        help="Activar modo de prueba: segmenta solo 50 imágenes y guarda en 'segmentation_test_50'.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Cargando modelo SAM desde {args.sam_checkpoint}...")
    try:
        sam_model = SamModel.from_pretrained(args.sam_checkpoint).to(device)
        sam_processor = SamProcessor.from_pretrained(args.sam_checkpoint)
    except Exception as e:
        print(f"Error al cargar el modelo SAM o el procesador: {e}")
        return

    masks_output_root = "segmentation_test_50" if args.test_mode else args.masks_dir
    print(f"Las máscaras se guardarán en: {masks_output_root}")

    try:
        dataset = MaskGenerationDataset(
            csv_file=args.csv_file,
            frames_dir=args.frames_dir,
            masks_output_root=masks_output_root,
            test_mode=args.test_mode
        )
    except Exception as e:
        print(f"Fallo al inicializar el Dataset: {e}")
        return

    if len(dataset) == 0:
        print("El dataset está vacío o no se pudieron parsear BBoxes válidos. No hay máscaras para generar. Saliendo.")
        return

    print(f"Total de máscaras listas en el dataset para procesar: {len(dataset)}")
    masks_generadas = 0
    masks_fallidas = 0

    pbar = tqdm(dataset, desc="Generando máscaras", unit="mask", ncols=100, total=len(dataset))
    for item in pbar:
        try:
            crop = item.get("crop")
            out_path = item.get("out_path")
            error_getitem = item.get("error")

            if error_getitem or crop is None or out_path is None:
                pbar.set_postfix(status=f"Error Carga/Item: {error_getitem or 'Datos inválidos'}",
                                 file=os.path.basename(out_path) if out_path else "N/A")
                masks_fallidas += 1
                continue

            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            if not isinstance(crop, Image.Image):
                pbar.set_postfix(status="Error: Crop no es objeto PIL.Image",
                                 file=os.path.basename(out_path))
                masks_fallidas += 1
                continue

            w, h = crop.size
            cx, cy = w // 2, h // 2
            input_points = [[[float(cx), float(cy)]]]

            t_start_proc = time.time()
            try:
                inputs = sam_processor(crop, input_points=input_points, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = sam_model(**inputs)

                masks_out = sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]
                
                mask_tensor = masks_out[0, 0, :, :]
                mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)
                refined_mask_np = refine_mask_with_skimage(mask_np)

                refined_mask_img = Image.fromarray(refined_mask_np, mode='L')
                refined_mask_img.save(out_path)

                masks_generadas += 1
                dt = time.time() - t_start_proc
                pbar.set_postfix(time=f"{dt:.2f}s", file=os.path.basename(out_path), status="OK")
            except Exception as e_sam:
                pbar.set_postfix(status=f"Error SAM/Proc: {e_sam}", file=os.path.basename(out_path))
                masks_fallidas += 1

        except Exception as e_loop:
            print(f"\nError inesperado en bucle principal procesando un item: {e_loop}")
            masks_fallidas += 1
            current_out_path = item.get('out_path', 'Desconocido')
            pbar.set_postfix(status=f"Error Loop: {e_loop}",
                             file=os.path.basename(current_out_path) if current_out_path != 'Desconocido' else 'N/A')
            continue

    print(f"\nProceso completado.")
    print(f"Total de máscaras generadas exitosamente: {masks_generadas}")
    print(f"Total de máscaras fallidas o saltadas (carga + proceso): {masks_fallidas}")


if __name__ == "__main__":
    main()
