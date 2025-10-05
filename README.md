# Proyecto TTS — Tacotron 2 (En desarrollo)

Repositorio con código para experimentar con Tacotron 2 en español usando el dataset CSS10 (Spanish).

## Contenido

- Código fuente (preprocesamiento, selección de datos y entrenamiento).
- Scripts para construir vocabularios fonémicos usando `phonemizer` + `espeak-ng`.

## Datos

Este repo NO incluye los datos del dataset CSS10. Para reproducir los experimentos descarga el CSS10 Spanish:

- CSS10 (Spanish): https://github.com/Kyubyong/css10

Coloca los archivos en la ruta esperada por los scripts: `data/raw/CSS10_spanish`.

## Requisitos (mínimos)

- Python 3.10+
- PyTorch (compatible con tu CUDA si usas GPU)
- librosa, torchaudio
- phonemizer

Instalar dependencias básicas (ejemplo):

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install librosa phonemizer soundfile
```

## Instalar espeak-ng (Windows)

Para usar `phonemizer` con el backend `espeak-ng` en Windows, instala `espeak-ng` y añade su binario al PATH.

1. Descarga e instala espeak-ng para Windows (builds comunitarios) o usa scoop/chocolatey si los tienes:

```powershell
# Con Chocolatey (ejemplo)
choco install espeak

# Con Scoop (ejemplo)
scoop install espeak
```

2. Verifica que el ejecutable esté en el PATH:

```powershell
espeak-ng --version
```

3. En Python, `phonemizer` detectará el backend si `espeak-ng` está disponible.

## Uso rápido

- Preprocesado y construcción de vocab:

```powershell
python scripts/build_vocab.py
```

- Entrenamiento (ejemplo, desde la raíz del repo):

```powershell
python main.py
```

## Enlaces útiles

- CSS10 dataset: https://github.com/Kyubyong/css10
- Phonemizer: https://github.com/bootphon/phonemizer
- eSpeak NG: https://github.com/espeak-ng/espeak-ng

---

