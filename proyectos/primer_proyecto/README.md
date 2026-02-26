# CharCNN OCR Service

Servicio web para clasificar caracteres individuales usando el modelo CNN exportado desde el notebook.

## ✅ Requisitos previos

1. Ejecuta el notebook `ocr-engines-vs-cnn-character-recognition-showdown.ipynb` y guarda el modelo en:

```
./exports/charcnn_state_dict.pt
./exports/charcnn_metadata.json
```

## Instalación

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Opcional (tests):

```
pip install -r requirements-dev.txt
```

## Ejecutar el servicio

```
uvicorn app:app --reload
```

Luego abre `http://127.0.0.1:8000` para usar la interfaz web.

## API

- `POST /predict` — Subir imagen (PNG/JPG) y obtener clasificación.
- `POST /predict_document` — Subir credencial y obtener nombre/domicilio con EasyOCR.
- `GET /health` — Estado del servicio.

Ejemplo de respuesta:

```json
{
  "prediction": "A",
  "confidence": 0.97,
  "probabilities": [0.0, 0.0, 0.97, "..."],
  "classes": ["0", "1", "2", "...", "Z"]
}

Ejemplo de respuesta para credenciales:

```json
{
  "nombre": "JUAN PEREZ",
  "domicilio": "AV. UNIVERSIDAD 123 COL. CENTRO",
  "raw_text": "NOMBRE: JUAN PEREZ\nDOMICILIO: AV. UNIVERSIDAD 123",
  "lines": ["NOMBRE: JUAN PEREZ", "DOMICILIO: AV. UNIVERSIDAD 123"]
}
```
```

## Tests

```
pytest
```

## Notas

- Si `exports/` no existe o está vacío, el endpoint `/predict` devolverá error 503.
- El modelo se carga al primer request para ahorrar memoria al iniciar.
- El endpoint `/predict_document` usa EasyOCR y puede tardar en el primer request porque descarga modelos.
