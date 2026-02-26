from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import app, extract_document_fields


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_returns_503_without_model() -> None:
    image = Image.new("L", (32, 32), color=255)
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("sample.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code in {400, 503}


def test_extract_document_fields() -> None:
    lines = [
        "NOMBRE: Juan Perez",
        "DOMICILIO: Av. Universidad 123",
        "Col. Centro",
    ]
    fields = extract_document_fields(lines)
    assert fields["nombre"] == "Juan Perez"
    assert "Universidad" in (fields["domicilio"] or "")
