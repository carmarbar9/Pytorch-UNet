import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_png(path: Path, image: np.ndarray):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"No se pudo guardar la imagen: {path}")
    encoded.tofile(str(path))


def process_split(split_dir: Path, output_img_dir: Path, output_mask_dir: Path):
    json_path = split_dir / "_annotations.coco.json"

    if not json_path.exists():
        print(f"[AVISO] No existe {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco["images"]}

    annotations_per_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_per_image.setdefault(img_id, []).append(ann)

    real_images = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in split_dir.glob(ext):
            real_images[img_path.name] = img_path

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"\nProcesando split: {split_dir.name}")
    print(f"Ruta origen: {split_dir}")
    print(f"Imágenes en JSON: {len(images_info)}")
    print(f"Imágenes reales en carpeta: {len(real_images)}")

    saved = 0
    skipped = 0

    for img_id, anns in tqdm(annotations_per_image.items(), desc=f"Generando {split_dir.name}"):
        img_info = images_info.get(img_id)

        if img_info is None:
            skipped += 1
            continue

        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        src_img_path = real_images.get(file_name)

        if src_img_path is None:
            stem = Path(file_name).stem
            for real_name, real_path in real_images.items():
                if stem in Path(real_name).stem:
                    src_img_path = real_path
                    break

        if src_img_path is None:
            print(f"[AVISO] No encontrada: {file_name}")
            skipped += 1
            continue

        img = read_image(src_img_path)

        if img is None:
            print(f"[AVISO] No se pudo leer: {src_img_path}")
            skipped += 1
            continue

        mask = np.zeros((height, width), dtype=np.uint8)

        for i, ann in enumerate(anns, start=1):
            segmentation = ann.get("segmentation", [])

            for poly in segmentation:
                if len(poly) < 6:
                    continue

                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], color=i)

        stem = Path(file_name).stem
        out_img_path = output_img_dir / f"{stem}.png"
        out_mask_path = output_mask_dir / f"{stem}.png"

        write_png(out_img_path, img)
        write_png(out_mask_path, mask)

        saved += 1

    print(f"[OK] Guardadas {saved} imágenes y máscaras en {split_dir.name}")
    print(f"[INFO] Omitidas: {skipped}")


def main():
    # El script se ejecuta desde la carpeta del repo Pytorch-UNet
    repo_root = Path.cwd()
    project_root = repo_root.parent

    base_raw = project_root / "dataset_raw"
    base_processed = project_root / "dataset_processed"

    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = base_raw / split
        output_img_dir = base_processed / split / "imgs"
        output_mask_dir = base_processed / split / "masks"

        process_split(split_dir, output_img_dir, output_mask_dir)

    print("\nTodo listo.")


if __name__ == "__main__":
    main()