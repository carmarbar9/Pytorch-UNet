import cv2
import numpy as np
from pathlib import Path

# Script para convertir máscaras a formato binario (0 y 1) si no lo están ya.

# CAMBIA SOLO ESTO
SPLIT = "train"   # "train", "valid" o "test"

base_path = Path(r"C:\Users\Carlos\Desktop\Año 5\PID\brain-tumor-unet\dataset_processed")
mask_dir = base_path / SPLIT / "masks"

print(f"\nProcesando split: {SPLIT}")
print(f"Ruta: {mask_dir}\n")

converted = 0
skipped = 0

for mask_path in sorted(mask_dir.glob("*.png")):
    # leer bien (compatible con ñ)
    mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if mask is None:
        print(f"[ERROR] No se pudo leer: {mask_path}")
        continue

    unique_vals = np.unique(mask)

    # Si ya es binaria - no tocar
    if set(unique_vals).issubset({0, 1}):
        skipped += 1
        continue

    # Binarizar
    binary_mask = (mask > 0).astype(np.uint8)

    # guardar sobrescribiendo
    cv2.imencode(".png", binary_mask)[1].tofile(str(mask_path))

    print(f"[OK] Convertida: {mask_path.name} {unique_vals} → [0 1]")
    converted += 1


print("\n--- RESUMEN ---")
print(f"Convertidas: {converted}")
print(f"Ya binarias (no tocadas): {skipped}")