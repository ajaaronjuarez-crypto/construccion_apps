from __future__ import annotations

import io
import json
import random
import re
from pathlib import Path
from typing import List, Optional

import cv2
import easyocr
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "exports"
MODEL_PATH = EXPORT_DIR / "charcnn_state_dict.pt"
META_PATH = EXPORT_DIR / "charcnn_metadata.json"
DATASET_DIR = BASE_DIR / "OCR"

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_sample_path(label: str, split: str = "training", dataset: str = "data") -> Path:
    if split not in {"training", "testing"}:
        raise ValueError("split must be training or testing")
    if dataset not in {"data", "data2"}:
        raise ValueError("dataset must be data or data2")
    if not label.isalnum():
        raise ValueError("label must be alphanumeric")
    folder = DATASET_DIR / dataset / f"{split}_data" / label.upper()
    if not folder.exists():
        raise FileNotFoundError("Label folder not found")
    candidates = list(folder.glob("*.png"))
    if not candidates:
        raise FileNotFoundError("No images found for label")
    return random.choice(candidates)


class CharCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: List[float]
    classes: List[str]


class DocumentResponse(BaseModel):
    nombre: Optional[str]
    domicilio: Optional[str]
    raw_text: str
    lines: List[str]


class ModelService:
    def __init__(self) -> None:
        self.device = get_device()
        self.model: CharCNN | None = None
        self.metadata: dict | None = None

    def load(self) -> None:
        if self.model is not None and self.metadata is not None:
            return
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Model files not found. Run the notebook to export the model into ./exports."
            )
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        num_classes = int(metadata.get("num_classes", 0))
        if num_classes <= 0:
            raise ValueError("Metadata is missing num_classes.")
        model = CharCNN(num_classes=num_classes)
        state = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        self.metadata = metadata

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        if self.metadata is None:
            raise RuntimeError("Model metadata not loaded.")
        input_size = int(self.metadata.get("input_size", 32))
        gray = image.convert("L")
        resized = gray.resize((input_size, input_size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tensor

    def predict(self, image: Image.Image) -> PredictionResponse:
        self.load()
        assert self.model is not None
        assert self.metadata is not None
        x = self.prepare_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        classes = self.metadata.get("classes", [])
        if not classes:
            classes = [str(i) for i in range(probs.numel())]
        probs_list = probs.cpu().tolist()
        top_idx = int(torch.argmax(probs).item())
        prediction = classes[top_idx] if top_idx < len(classes) else str(top_idx)
        confidence = float(probs_list[top_idx])
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probs_list,
            classes=classes,
        )


def normalize_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned


def extract_document_fields(lines: List[str]) -> dict:
    normalized = [normalize_line(line) for line in lines if line.strip()]
    normalized_upper = [line.upper() for line in normalized]

    def clean_nombre(value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        cleaned = re.sub(r"^SEXO\s*[HM]\s+", "", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bSEXO\b.*", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned or None

    def find_label_value(label_keywords: List[str], stop_labels: List[str]) -> Optional[str]:
        for idx, line in enumerate(normalized_upper):
            if any(keyword in line for keyword in label_keywords):
                original = normalized[idx]
                parts = re.split(r":|-", original, maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    return parts[1].strip()
                for keyword in label_keywords:
                    pattern = re.compile(rf"{re.escape(keyword)}\s+(.*)", re.IGNORECASE)
                    match = pattern.search(original)
                    if match and match.group(1).strip():
                        return match.group(1).strip()
                collected: List[str] = []
                for offset in range(1, 5):
                    if idx + offset >= len(normalized):
                        break
                    candidate = normalized[idx + offset].strip()
                    if not candidate:
                        break
                    if any(stop_label in candidate.upper() for stop_label in stop_labels):
                        break
                    collected.append(candidate)
                if collected:
                    return " ".join(collected)
        return None

    stop_labels = [
        "NOMBRE",
        "NOMBRES",
        "DOMICILIO",
        "DIRECCION",
        "DIRECCIÓN",
        "ADDRESS",
        "SEXO",
        "CLAVE",
        "CURP",
        "AÑO",
        "ANIO",
        "REGISTRO",
        "FECHA",
        "SECCION",
        "SECCIÓN",
        "VIGENCIA",
        "INSTITUTO",
        "CREDENCIAL",
        "ELECTOR",
    ]
    stop_labels_name = [label for label in stop_labels if label != "SEXO"]
    nombre = clean_nombre(
        find_label_value(["NOMBRE", "NAME", "NOMBRES"], stop_labels_name)
    )

    domicilio = None
    domicilio_labels = ["DOMICILIO", "DIRECCION", "DIRECCIÓN", "ADDRESS"]
    for idx, line in enumerate(normalized_upper):
        if any(label in line for label in domicilio_labels):
            original = normalized[idx]
            parts = re.split(r":|-", original, maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                domicilio = parts[1].strip()
            else:
                for label in domicilio_labels:
                    pattern = re.compile(rf"{re.escape(label)}\s+(.*)", re.IGNORECASE)
                    match = pattern.search(original)
                    if match and match.group(1).strip():
                        domicilio = match.group(1).strip()
                        break
            if domicilio is None:
                candidates = []
                for offset in range(1, 5):
                    if idx + offset < len(normalized):
                        candidate = normalized[idx + offset].strip()
                        if candidate and not any(lbl in candidate.upper() for lbl in stop_labels):
                            candidates.append(candidate)
                if candidates:
                    domicilio = " ".join(candidates)
            break

    if domicilio is None:
        address_tokens = ("CALLE", "AV", "AV.", "COL", "COL.", "CP", "C.P", "NUM", "NO.")
        for line in normalized_upper:
            if any(token in line for token in address_tokens) and any(char.isdigit() for char in line):
                idx = normalized_upper.index(line)
                domicilio = normalized[idx]
                break

    return {"nombre": nombre, "domicilio": domicilio}


def preprocess_document_image(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    scale = 2.0
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurred, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

    (h, w) = blurred.shape[:2]
    center = (w // 2, h // 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        blurred,
        rotation,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    thresh = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(thresh, -1, kernel)
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)


def crop_region(image: np.ndarray, bbox: List[List[float]], padding: int = 20) -> np.ndarray:
    h, w = image.shape[:2]
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    x1 = max(int(min(xs)) - padding, 0)
    y1 = max(int(min(ys)) - padding, 0)
    x2 = min(int(max(xs)) + padding, w)
    y2 = min(int(max(ys)) + padding, h)
    return image[y1:y2, x1:x2]


class DocumentOcrService:
    def __init__(self) -> None:
        self.reader: easyocr.Reader | None = None

    def load(self) -> None:
        if self.reader is None:
            self.reader = easyocr.Reader(["es", "en"], gpu=torch.cuda.is_available())

    def _score_lines(self, lines: List[str]) -> int:
        if not lines:
            return 0
        joined = " ".join(lines)
        alnum = sum(char.isalnum() for char in joined)
        return alnum

    def _run_ocr(self, image_array: np.ndarray, **kwargs) -> List[str]:
        assert self.reader is not None
        results = self.reader.readtext(image_array, paragraph=False, **kwargs)
        return [normalize_line(text) for _, text, _ in results if text.strip()]

    def predict(self, image: Image.Image) -> DocumentResponse:
        self.load() 
        assert self.reader is not None
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        if min(h, w) < 600:
            scale = 2.5
            image_array = cv2.resize(image_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        lines = self._run_ocr(image_array)
        best_lines = lines
        best_score = self._score_lines(lines)

        label_keywords = ("NOMBRE", "DOMICILIO", "DIRECCION", "DIRECCIÓN")
        label_boxes = []
        for bbox, text, _ in self.reader.readtext(image_array, paragraph=False):
            if any(keyword in text.upper() for keyword in label_keywords):
                label_boxes.append(bbox)

        if label_boxes:
            xs = [point[0] for bbox in label_boxes for point in bbox]
            ys = [point[1] for bbox in label_boxes for point in bbox]
            height, width = image_array.shape[:2]
            x1 = max(int(min(xs)) - 40, 0)
            x2 = min(int(max(xs)) + 40, width)
            y1 = max(int(min(ys)) - 40, 0)
            y2 = min(int(max(ys)) + 40 + int(height * 0.35), height)
            roi = image_array[y1:y2, x1:x2]
            roi_lines = self._run_ocr(roi)
            roi_score = self._score_lines(roi_lines)
            if roi_score >= best_score:
                best_lines = roi_lines
                best_score = roi_score

        contrast_lines = self._run_ocr(
            image_array, contrast_ths=0.05, adjust_contrast=0.6, text_threshold=0.6, low_text=0.3
        )
        contrast_score = self._score_lines(contrast_lines)
        if contrast_score > best_score:
            best_lines = contrast_lines
            best_score = contrast_score

        raw_text = "\n".join(best_lines)
        fields = extract_document_fields(best_lines)

        if not fields.get("nombre") and not fields.get("domicilio"):
            prepared = preprocess_document_image(image)
            fallback_lines = self._run_ocr(prepared)
            fallback_score = self._score_lines(fallback_lines)
            if fallback_score > best_score:
                best_lines = fallback_lines
                raw_text = "\n".join(best_lines)
                fields = extract_document_fields(best_lines)
        return DocumentResponse(
            nombre=fields.get("nombre"),
            domicilio=fields.get("domicilio"),
            raw_text=raw_text,
            lines=best_lines,
        )


service = ModelService()
document_service = DocumentOcrService()

app = FastAPI(title="CharCNN OCR Service", version="1.0.0") 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "interfaz.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": service.model is not None,
        "model_path": str(MODEL_PATH),
        "meta_path": str(META_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Optional[UploadFile] = File(None),
    dataset_label: Optional[str] = None,
    split: str = "training",
    dataset: str = "data",
) -> PredictionResponse:
    try:
        if dataset_label:
            sample_path = get_dataset_sample_path(dataset_label, split=split, dataset=dataset)
            image = Image.open(sample_path)
            return service.predict(image)
        if file is None:
            raise HTTPException(status_code=400, detail="Provide file or dataset_label")
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        raw = await file.read()
        image = Image.open(io.BytesIO(raw))
        return service.predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process image: {exc}") from exc


@app.post("/predict_document", response_model=DocumentResponse)
async def predict_document(file: UploadFile = File(...)) -> DocumentResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return document_service.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process document: {exc}") from exc


@app.get("/dataset/sample") 
def dataset_sample(label: str, split: str = "training", dataset: str = "data") -> FileResponse:
    if split not in {"training", "testing"}:
        raise HTTPException(status_code=400, detail="split must be training or testing")
    if dataset not in {"data", "data2"}:
        raise HTTPException(status_code=400, detail="dataset must be data or data2")
    if not label.isalnum():
        raise HTTPException(status_code=400, detail="label must be alphanumeric")
    folder = DATASET_DIR / dataset / f"{split}_data" / label.upper()
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Label folder not found")
    candidates = list(folder.glob("*.png"))
    if not candidates:
        raise HTTPException(status_code=404, detail="No images found for label")
    choice = random.choice(candidates)
    return FileResponse(choice)
