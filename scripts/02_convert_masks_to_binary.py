from pathlib import Path

import cv2
import numpy as np


# Cambiar este valor si se quiere procesar otro split: "train", "valid" o "test"
SPLIT = "train"


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def write_png(path: Path, image: np.ndarray):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"No se pudo guardar la máscara: {path}")
    encoded.tofile(str(path))


def is_binary(mask: np.ndarray) -> bool:
    unique_vals = np.unique(mask)
    return set(unique_vals.tolist()).issubset({0, 1})


def main():
    # El script se ejecuta desde la carpeta del repo Pytorch-UNet
    repo_root = Path.cwd()
    project_root = repo_root.parent

    mask_dir = project_root / "dataset_processed" / SPLIT / "masks"

    if not mask_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de máscaras: {mask_dir}")

    print(f"\nProcesando split: {SPLIT}")
    print(f"Ruta: {mask_dir}\n")

    converted = 0
    skipped = 0

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = read_image(mask_path)

        if mask is None:
            print(f"[ERROR] No se pudo leer: {mask_path}")
            continue

        unique_vals = np.unique(mask)

        if is_binary(mask):
            skipped += 1
            continue

        binary_mask = (mask > 0).astype(np.uint8)

        write_png(mask_path, binary_mask)

        print(f"[OK] Convertida: {mask_path.name} {unique_vals} → [0 1]")
        converted += 1

    print("\n--- RESUMEN ---")
    print(f"Convertidas: {converted}")
    print(f"Ya binarias (no tocadas): {skipped}")


if __name__ == "__main__":
    main()