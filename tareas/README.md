# Servicio SecureAccess (Basic Auth)

_Nota: ajuste mínimo para PR de prueba._

Este servicio expone un flujo de login con Basic Auth (Base64) usando FastAPI, SQLite y las vistas `20260302_access.html` y `20260302_platform.html`.

## Credenciales por defecto

- **Usuario:** `admin@secureaccess.local`
- **Contraseña:** `Admin123!`

Puedes cambiar estos valores con variables de entorno (ver abajo).

## Variables de entorno

- `ACCESS_DEFAULT_USER`: usuario inicial a registrar en la BD.
- `ACCESS_DEFAULT_PASSWORD`: contraseña inicial a registrar en la BD.
- `ACCESS_DB_PATH`: ruta de la base SQLite (por defecto `tareas/access_users.db`).
- `ACCESS_SALT`: sal para el hash SHA-256 (por defecto `change-me`).

## Ejecutar el servicio

```bash
uvicorn 20260302_access_service_b64:app --reload --port 8000
```

Luego abre:

- `http://localhost:8000/` (login)

## Endpoints CRUD (Basic Auth)

- `GET /users` → lista usuarios
- `POST /users` → crea usuario `{ "username": "...", "password": "..." }`
- `PUT /users/{username}` → actualiza contraseña `{ "password": "..." }`
- `DELETE /users/{username}` → elimina usuario

En la vista de `20260302_platform.html` hay un panel mínimo para crear, actualizar, eliminar y listar usuarios.

## Probar tests

```bash
pytest tareas/tests/test_access_service_b64.py
```
