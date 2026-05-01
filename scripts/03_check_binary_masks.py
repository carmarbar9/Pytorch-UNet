from pathlib import Path

import cv2
import numpy as np


# Cambiar este valor si se quiere comprobar otro split: "train", "valid" o "test"
SPLIT = "train"


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def main():
    # El script se ejecuta desde la carpeta del repo Pytorch-UNet
    repo_root = Path.cwd()
    project_root = repo_root.parent

    mask_dir = project_root / "dataset_processed" / SPLIT / "masks"

    if not mask_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de máscaras: {mask_dir}")

    print(f"Analizando máscaras en: {mask_dir}\n")

    all_values = set()
    masks_not_binary = []

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = read_image(mask_path)

        if mask is None:
            print(f"[ERROR] No se pudo leer: {mask_path}")
            continue

        unique_vals = np.unique(mask)
        all_values.update(unique_vals.tolist())

        print(f"{mask_path.name} → {unique_vals}")

        if not set(unique_vals.tolist()).issubset({0, 1}):
            masks_not_binary.append((mask_path.name, unique_vals))

    print("\n--- RESUMEN GLOBAL ---")
    print(f"Valores únicos en TODAS las máscaras analizadas: {sorted(all_values)}")

    print("\n--- MÁSCARAS NO BINARIAS ---")
    if len(masks_not_binary) == 0:
        print("Todas son binarias")
    else:
        for name, vals in masks_not_binary:
            print(f"{name} → {vals}")

        print(f"\nTotal máscaras no binarias: {len(masks_not_binary)}")


if __name__ == "__main__":
    main()