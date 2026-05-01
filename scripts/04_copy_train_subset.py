import shutil
from pathlib import Path


# Número de imágenes y máscaras que se copiarán al repositorio
NUM_SAMPLES = 120


def clear_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

    for file_path in folder.glob("*"):
        if file_path.is_file():
            file_path.unlink()


def main():
    # El script se ejecuta desde la carpeta del repo Pytorch-UNet
    repo_root = Path.cwd()
    project_root = repo_root.parent

    src_imgs = project_root / "dataset_processed" / "train" / "imgs"
    src_masks = project_root / "dataset_processed" / "train" / "masks"

    dst_imgs = repo_root / "data" / "imgs"
    dst_masks = repo_root / "data" / "masks"

    if not src_imgs.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes procesadas: {src_imgs}")

    if not src_masks.exists():
        raise FileNotFoundError(f"No existe la carpeta de máscaras procesadas: {src_masks}")

    clear_folder(dst_imgs)
    clear_folder(dst_masks)

    copied = 0

    for i, img_path in enumerate(sorted(src_imgs.glob("*.png"))):
        if i == NUM_SAMPLES:
            break

        mask_path = src_masks / img_path.name

        if not mask_path.exists():
            print(f"[AVISO] No existe máscara para: {img_path.name}")
            continue

        shutil.copy(img_path, dst_imgs / img_path.name)
        shutil.copy(mask_path, dst_masks / mask_path.name)

        copied += 1

    print("\n--- RESUMEN ---")
    print(f"Copiadas {copied} imágenes y máscaras")
    print(f"Origen imágenes: {src_imgs}")
    print(f"Origen máscaras: {src_masks}")
    print(f"Destino imágenes: {dst_imgs}")
    print(f"Destino máscaras: {dst_masks}")


if __name__ == "__main__":
    main()