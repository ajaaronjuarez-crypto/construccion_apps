import base64
import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def load_app(tmp_path: Path):
    os.environ["ACCESS_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["ACCESS_DEFAULT_USER"] = "user@example.com"
    os.environ["ACCESS_DEFAULT_PASSWORD"] = "Secret123!"
    os.environ["ACCESS_SALT"] = "testsalt"

    tareas_dir = Path(__file__).resolve().parents[1]
    if str(tareas_dir) not in sys.path:
        sys.path.insert(0, str(tareas_dir))

    module = importlib.import_module("20260302_access_service_b64")
    importlib.reload(module)
    return module.app


def build_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def test_login_and_platform(tmp_path):
    app = load_app(tmp_path)
    client = TestClient(app)

    headers = build_auth_header("user@example.com", "Secret123!")
    response = client.post("/login", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"] == "ok"

    platform_response = client.get("/platform", headers=headers)
    assert platform_response.status_code == 200
    assert "SecureAccess" in platform_response.text


def test_login_rejects_invalid_credentials(tmp_path):
    app = load_app(tmp_path)
    client = TestClient(app)

    headers = build_auth_header("user@example.com", "wrong")
    response = client.post("/login", headers=headers)
    assert response.status_code == 401


def test_user_crud_flow(tmp_path):
    app = load_app(tmp_path)
    client = TestClient(app)

    headers = build_auth_header("user@example.com", "Secret123!")

    create_response = client.post(
        "/users",
        headers=headers,
        json={"username": "new-user", "password": "Password1!"},
    )
    assert create_response.status_code == 201

    list_response = client.get("/users", headers=headers)
    assert list_response.status_code == 200
    assert "new-user" in list_response.json()["users"]

    update_response = client.put(
        "/users/new-user",
        headers=headers,
        json={"password": "UpdatedPass1!"},
    )
    assert update_response.status_code == 200

    delete_response = client.delete("/users/new-user", headers=headers)
    assert delete_response.status_code == 200

    list_response = client.get("/users", headers=headers)
    assert "new-user" not in list_response.json()["users"]
