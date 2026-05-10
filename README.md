# Segmentación semántica de tumores cerebrales con U-Net

Este repositorio es una adaptación del proyecto original [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) para segmentar tumores cerebrales en imágenes de resonancia magnética.

El modelo utilizado es U-Net y el problema se trata como segmentación binaria:

- `0`: fondo
- `1`: tumor

Por ello, todos los entrenamientos y predicciones deben ejecutarse con:

```bash
--classes 1
```

o, en predicción:

```bash
-c 1
```

---

## 1. Diferencias respecto al repositorio original

El repositorio original estaba preparado principalmente para segmentación sobre el dataset Carvana. En este proyecto se ha adaptado para segmentación médica de tumores cerebrales.

| Elemento | Repositorio original | Este proyecto |
|---|---|---|
| Dataset | Carvana Image Masking Challenge | Brain Tumor Image Dataset - Semantic Segmentation |
| Tipo de imágenes | Coches | Resonancias magnéticas cerebrales |
| Tipo de problema | Segmentación general | Segmentación binaria de tumores |
| Clases | Variable | 1 clase: tumor |
| Máscaras | Ya preparadas | Generadas desde anotaciones COCO |
| Valores de máscara | Según el dataset original | `0` fondo y `1` tumor |
| Predicción binaria | Flujo original | Adaptada con `sigmoid` y umbral |
| Visualización | `--viz` | Se añade `--overlay` |
| Comparación con máscara real | No incluida directamente | Añadida usando `data/pred/masks` |
| Dice Score por imagen | No incluido en predicción | Añadido en la predicción con `--overlay` |
| Guardado del mejor modelo | Checkpoints por época | Se guarda `checkpoints/best_model.pth` |
| Entrenamiento en CPU | Configuración general | Optimizado para CPU |
| Weights & Biases | Activo por defecto | Desactivado por defecto |
| Optimizador | RMSprop | AdamW en la versión final |

---

## 2. Contribución propia

Las principales aportaciones realizadas sobre el repositorio original son:

- Adaptación del proyecto al dataset de tumores cerebrales.
- Preparación de máscaras binarias a partir de anotaciones COCO.
- Scripts para generar, convertir, comprobar y copiar máscaras.
- Adaptación del modelo a segmentación binaria usando `--classes 1`.
- Modificación de la predicción para usar `sigmoid` y umbral en vez de `argmax`.
- Nueva opción `--overlay` para visualizar la predicción sobre la imagen original.
- Comparación automática con la máscara real cuando existe en `data/pred/masks`.
- Cálculo del Dice Score individual en predicción.
- Guardado automático del mejor modelo como `checkpoints/best_model.pth`.
- Optimización del entrenamiento para poder ejecutarlo en CPU.
- Cambio experimental del optimizador a AdamW.

---

## 3. Dataset utilizado

El dataset utilizado es:

**Brain Tumor Image Dataset - Semantic Segmentation**

Disponible en Kaggle:

```text
https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation
```

El dataset original contiene imágenes y anotaciones en formato COCO. Para entrenar el modelo, esas anotaciones deben convertirse en máscaras binarias.

Este repositorio incluye un subconjunto ya preparado de 120 imágenes y 120 máscaras en:

```text
data/imgs/
data/masks/
```

Por tanto, se puede entrenar directamente sin preparar el dataset desde cero.

---

## 4. Estructura esperada

La estructura recomendada es:

```text
brain-tumor-unet/
|
|-- dataset_raw/
|   |-- train/
|   |   |-- _annotations.coco.json
|   |   |-- imagenes...
|   |-- valid/
|   |   |-- _annotations.coco.json
|   |   |-- imagenes...
|   |-- test/
|       |-- _annotations.coco.json
|       |-- imagenes...
|
|-- Pytorch-UNet/
    |-- data/
    |   |-- imgs/
    |   |-- masks/
    |   |-- pred/
    |       |-- imgs/
    |       |-- masks/
    |-- scripts/
    |-- train.py
    |-- predict.py
    |-- requirements.txt
```

La carpeta `dataset_raw` solo es necesaria si se quiere regenerar el dataset desde cero.

---

## 5. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/carmarbar9/Pytorch-UNet.git
cd Pytorch-UNet
```

Crear entorno virtual.

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

En Linux o macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Actualizar pip e instalar dependencias:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Se recomienda usar Python 3.10, 3.11 o 3.12.

---

## 6. Preparar el dataset desde cero

Este paso no es obligatorio si se usan las imágenes ya incluidas en `data/imgs` y `data/masks`.

Si se quiere preparar el dataset desde las anotaciones COCO originales, primero hay que descargar el dataset de Kaggle y colocarlo en:

```text
brain-tumor-unet/dataset_raw/
```

Después, desde la carpeta `Pytorch-UNet`, ejecutar:

```bash
python scripts/01_generate_masks_from_coco.py
python scripts/02_convert_masks_to_binary.py
python scripts/03_check_binary_masks.py
python scripts/04_copy_train_subset.py
```

Estos scripts hacen lo siguiente:

1. Generan máscaras desde las anotaciones COCO.
2. Convierten las máscaras a formato binario.
3. Comprueban que las máscaras solo contienen `0` y `1`.
4. Copian un subconjunto a `data/imgs` y `data/masks`.

Para cambiar el número de imágenes copiadas, modificar en `scripts/04_copy_train_subset.py`:

```python
NUM_SAMPLES = 120
```

Por ejemplo:

```python
NUM_SAMPLES = 200
```

---

## 7. Entrenamiento

Ejemplo básico de entrenamiento:

```bash
python train.py --epochs 5 --batch-size 2 --classes 1
```

Ejemplo usado en las pruebas:

```bash
python train.py --epochs 7 --batch-size 4 --classes 1
```

Otro ejemplo:

```bash
python train.py --epochs 9 --batch-size 2 --classes 1
```

Durante el entrenamiento se guardan checkpoints en:

```text
checkpoints/
```

Además, el mejor modelo se guarda automáticamente en:

```text
checkpoints/best_model.pth
```

Ese es el modelo recomendado para hacer predicciones.

---

## 8. Predicción

Para predecir una imagen:

```bash
python predict.py -m checkpoints/best_model.pth -i data/pred/imgs/imagen_test.png -o data/pred/imagen_test_OUT.png -c 1
```

Donde:

- `-m`: modelo entrenado.
- `-i`: imagen de entrada.
- `-o`: imagen de salida.
- `-c 1`: indica que el modelo tiene una única clase de salida.

---

## 9. Predicción con overlay

Para visualizar la máscara predicha sobre la imagen original:

```bash
python predict.py -m checkpoints/best_model.pth -i data/pred/imgs/imagen_test.png -o data/pred/imagen_test_OUT.png -c 1 --overlay
```

La predicción se muestra en rojo.

Si existe una máscara real con el mismo nombre en `data/pred/masks`, también se genera una comparación visual y se calcula el Dice Score de esa imagen.

Estructura esperada:

```text
data/pred/
|-- imgs/
|   |-- imagen_test.png
|-- masks/
    |-- imagen_test.png
```

Archivos de salida habituales:

```text
imagen_test_OUT.png
imagen_test_OUT_overlay.png
imagen_test_OUT_overlay_gt.png
```

---

## 10. Ejecución rápida

Si se usan los datos ya preparados del repositorio, basta con ejecutar:

```bash
git clone https://github.com/carmarbar9/Pytorch-UNet.git
cd Pytorch-UNet

python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install -r requirements.txt

python train.py --epochs 5 --batch-size 2 --classes 1

python predict.py -m checkpoints/best_model.pth -i data/pred/imgs/imagen_test.png -o data/pred/imagen_test_OUT.png -c 1 --overlay
```

En Linux o macOS, sustituir:

```bash
venv\Scripts\activate
```

por:

```bash
source venv/bin/activate
```

---

## 11. Problemas frecuentes

### Error al cargar el modelo en predicción

Comprobar que se está usando `-c 1`:

```bash
python predict.py -m checkpoints/best_model.pth -i data/pred/imgs/imagen_test.png -o data/pred/out.png -c 1
```

### La predicción sale negra

Puede ocurrir porque la máscara se guarda con valores `0` y `1`, difíciles de ver directamente. Se recomienda usar:

```bash
--overlay
```

### El entrenamiento tarda demasiado

El proyecto se ha probado principalmente en CPU. Para reducir el tiempo:

- Usar pocas épocas al principio.
- Usar 120 imágenes.
- Mantener `--scale 0.5`.
- Usar batch size `2` o `4`.

---

## 12. Autores

- Carlos Martín de Prado Barragán
- Manuel Jesús Cádiz Santillana
- Antonio Membrive Martínez

---

## 13. Referencias

- Repositorio original PyTorch-UNet: https://github.com/milesial/Pytorch-UNet
- Dataset utilizado: https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation
- Paper original de U-Net: https://arxiv.org/abs/1505.04597
