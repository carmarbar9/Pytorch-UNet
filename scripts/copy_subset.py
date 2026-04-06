import shutil
from pathlib import Path

# Copia un subconjunto de 120 imágenes y máscaras del conjunto de entrenamiento
# a la carpeta de datos de Pytorch-UNet

BASE = Path(__file__).resolve().parent.parent

# Rutas de origen y destino
src_imgs = BASE / "dataset_processed" / "train" / "imgs"
src_masks = BASE / "dataset_processed" / "train" / "masks"

dst_imgs = BASE / "Pytorch-UNet" / "data" / "imgs"
dst_masks = BASE / "Pytorch-UNet" / "data" / "masks"

dst_imgs.mkdir(parents=True, exist_ok=True)
dst_masks.mkdir(parents=True, exist_ok=True)

for i, img_path in enumerate(sorted(src_imgs.glob("*.png"))):
    if i == 120:
        break

    mask_path = src_masks / img_path.name

    shutil.copy(img_path, dst_imgs / img_path.name)
    shutil.copy(mask_path, dst_masks / mask_path.name)

print("Copiadas 120 imágenes y máscaras")