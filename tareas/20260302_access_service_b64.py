from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent


## Usuario: admin@secureaccess.local
# Contraseña: Admin123!

ACCESS_HTML_PATH = APP_DIR / "20260302_access.html"
PLATFORM_HTML_PATH = APP_DIR / "20260302_platform.html"

DB_PATH = Path(os.getenv("ACCESS_DB_PATH", str(APP_DIR / "access_users.db")))
DEFAULT_USER = os.getenv("ACCESS_DEFAULT_USER", "admin@secureaccess.local")
DEFAULT_PASSWORD = os.getenv("ACCESS_DEFAULT_PASSWORD", "Admin123!")
PASSWORD_SALT = os.getenv("ACCESS_SALT", "change-me")

security = HTTPBasic()


@asynccontextmanager
async def lifespan(_: FastAPI):
	init_db()
	yield


app = FastAPI(title="SecureAccess - Basic Auth", lifespan=lifespan)


class UserCreate(BaseModel):
	username: str = Field(..., min_length=3)
	password: str = Field(..., min_length=4)


class UserUpdate(BaseModel):
	password: str = Field(..., min_length=4)


def hash_password(password: str) -> str:
	digest = hashlib.sha256(f"{PASSWORD_SALT}{password}".encode("utf-8")).hexdigest()
	return digest


def init_db() -> None:
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	with sqlite3.connect(DB_PATH) as connection:
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS users (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				username TEXT UNIQUE NOT NULL,
				password_hash TEXT NOT NULL
			);
			"""
		)
		cursor = connection.execute(
			"SELECT id FROM users WHERE username = ?",
			(DEFAULT_USER,),
		)
		if cursor.fetchone() is None:
			connection.execute(
				"INSERT INTO users (username, password_hash) VALUES (?, ?)",
				(DEFAULT_USER, hash_password(DEFAULT_PASSWORD)),
			)
		connection.commit()


def verify_credentials(credentials: HTTPBasicCredentials) -> bool:
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.execute(
			"SELECT password_hash FROM users WHERE username = ?",
			(credentials.username,),
		)
		row = cursor.fetchone()
		if row is None:
			return False
		expected_hash = row[0]
	provided_hash = hash_password(credentials.password)
	return secrets.compare_digest(expected_hash, provided_hash)


def require_basic_auth(
	credentials: HTTPBasicCredentials = Depends(security),
) -> HTTPBasicCredentials:
	if not verify_credentials(credentials):
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid authentication credentials",
			headers={"WWW-Authenticate": "Basic"},
		)
	return credentials


def load_html(path: Path) -> str:
	if not path.exists():
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Missing HTML template: {path.name}",
		)
	return path.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def access_page() -> HTMLResponse:
	return HTMLResponse(load_html(ACCESS_HTML_PATH))


@app.post("/login")
def login(_: HTTPBasicCredentials = Depends(require_basic_auth)) -> JSONResponse:
	return JSONResponse({"message": "ok"})


@app.get("/platform", response_class=HTMLResponse)
def platform_page(_: HTTPBasicCredentials = Depends(require_basic_auth)) -> HTMLResponse:
	return HTMLResponse(load_html(PLATFORM_HTML_PATH))


@app.get("/users")
def list_users(_: HTTPBasicCredentials = Depends(require_basic_auth)) -> JSONResponse:
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		rows = connection.execute("SELECT username FROM users ORDER BY username").fetchall()
	users = [row[0] for row in rows]
	return JSONResponse({"users": users})


@app.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(
	payload: UserCreate,
	_: HTTPBasicCredentials = Depends(require_basic_auth),
) -> dict:
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		try:
			connection.execute(
				"INSERT INTO users (username, password_hash) VALUES (?, ?)",
				(payload.username, hash_password(payload.password)),
			)
			connection.commit()
		except sqlite3.IntegrityError as exc:
			raise HTTPException(
				status_code=status.HTTP_409_CONFLICT,
				detail="User already exists",
			) from exc
	return {"message": "created", "username": payload.username}


@app.put("/users/{username}")
def update_user(
	username: str,
	payload: UserUpdate,
	_: HTTPBasicCredentials = Depends(require_basic_auth),
) -> JSONResponse:
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.execute(
			"UPDATE users SET password_hash = ? WHERE username = ?",
			(hash_password(payload.password), username),
		)
		connection.commit()
		if cursor.rowcount == 0:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="User not found",
			)
	return JSONResponse({"message": "updated", "username": username})


@app.delete("/users/{username}")
def delete_user(
	username: str,
	_: HTTPBasicCredentials = Depends(require_basic_auth),
) -> JSONResponse:
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.execute("DELETE FROM users WHERE username = ?", (username,))
		connection.commit()
		if cursor.rowcount == 0:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="User not found",
			)
	return JSONResponse({"message": "deleted", "username": username})


@app.get("/health")
def healthcheck() -> JSONResponse:
	return JSONResponse({"status": "ok"})
