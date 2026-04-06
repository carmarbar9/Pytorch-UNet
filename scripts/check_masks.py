import cv2
import numpy as np
from pathlib import Path

# Este script analiza las máscaras en el conjunto de entrenamiento para verificar que sean binarias (0 y 1)
#  y detectar cualquier valor atípico que pueda afectar el entrenamiento del modelo.
BASE = Path(__file__).resolve().parent.parent

# Cambiar ruta si es necesario para apuntar al directorio correcto de máscaras
mask_dir = BASE / "dataset_processed" / "train" / "masks"

print(f"Analizando máscaras en: {mask_dir}\n")

all_values = set()
masks_not_binary = []

for mask_path in sorted(mask_dir.glob("*.png")):
    mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if mask is None:
        print(f"[ERROR] No se pudo leer: {mask_path}")
        continue

    unique_vals = np.unique(mask)
    all_values.update(unique_vals)

    print(f"{mask_path.name} → {unique_vals}")

    # Detectar si NO es binaria pura {0,1}
    if not set(unique_vals).issubset({0, 1}):
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