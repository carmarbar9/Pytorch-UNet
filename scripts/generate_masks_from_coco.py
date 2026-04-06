import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Script para generar máscaras a partir de anotaciones COCO.

def process_split(split_dir, output_img_dir, output_mask_dir):
    split_dir = Path(split_dir)
    json_path = split_dir / "_annotations.coco.json"

    if not json_path.exists():
        print(f"[AVISO] No existe {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Info del JSON
    images_info = {img["id"]: img for img in coco["images"]}

    # Agrupar anotaciones
    annotations_per_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_per_image.setdefault(img_id, []).append(ann)

    # Crear mapa REAL de imágenes en disco
    real_images = {}
    for img_path in split_dir.glob("*.jpg"):
        real_images[img_path.name] = img_path

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"\nProcesando split: {split_dir.name}")
    print(f"Imágenes en JSON: {len(images_info)}")
    print(f"Imágenes reales en carpeta: {len(real_images)}")

    saved = 0

    for img_id, anns in tqdm(annotations_per_image.items()):
        img_info = images_info[img_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Intento 1: nombre exacto
        src_img_path = real_images.get(file_name)

        # Intento 2: búsqueda parcial 
        if src_img_path is None:
            for real_name, real_path in real_images.items():
                if file_name.split(".")[0] in real_name:
                    src_img_path = real_path
                    break

        if src_img_path is None:
            print(f"[AVISO] No encontrada: {file_name}")
            continue

        # Cargar imagen 
        img = cv2.imdecode(np.fromfile(str(src_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[AVISO] No se pudo leer: {src_img_path}")
            continue

        # Crear máscara
        mask = np.zeros((height, width), dtype=np.uint8)

        # Dibujar polígonos
        for i, ann in enumerate(anns, start=1):
            segmentation = ann.get("segmentation", [])

            for poly in segmentation:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], color=i)

        # Guardar
        stem = Path(file_name).stem
        out_img_path = Path(output_img_dir) / f"{stem}.png"
        out_mask_path = Path(output_mask_dir) / f"{stem}.png"

        # Guardar imagen
        cv2.imencode(".png", img)[1].tofile(str(out_img_path))

        # Guardar máscara
        cv2.imencode(".png", mask)[1].tofile(str(out_mask_path))

        saved += 1

    print(f"[OK] Guardadas {saved} imágenes en {split_dir.name}")


def main():
    base_raw = Path(r"C:\Users\Carlos\Desktop\Año 5\PID\brain-tumor-unet\dataset_raw")
    base_processed = Path(r"C:\Users\Carlos\Desktop\Año 5\PID\brain-tumor-unet\dataset_processed")

    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = base_raw / split
        output_img_dir = base_processed / split / "imgs"
        output_mask_dir = base_processed / split / "masks"

        process_split(split_dir, output_img_dir, output_mask_dir)

    print("\nTodo listo.")


if __name__ == "__main__":
    main()