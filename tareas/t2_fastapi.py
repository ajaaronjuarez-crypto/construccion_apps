
from fastapi import FastAPI, Query
from typing import List, Dict

app = FastAPI(title="API Paradigmas de Programación")

### Endpoint 1: Paradigma Funcional
def mayusculas(texto: str) -> str:
    return texto.upper()

def agregar_exclamacion(texto: str) -> str:
    return f"{texto}!"

@app.get("/funcional")
def endpoint_funcional(texto: str = Query(..., description="Texto a transformar")):
    """
    Endpoint que aplica composición de funciones (paradigma funcional).
    Transforma el texto a mayúsculas y agrega exclamación.
    """
    resultado = agregar_exclamacion(mayusculas(texto))
    return {
        "original": texto,
        "transformado": resultado,
        "paradigma": "funcional"
    }


### Endpoint 2: Paradigma Orientado a Datos
# Base de datos en memoria
mensajes_db = [
    {'id': 1, 'texto': 'Hola mundo', 'tipo': 'saludo'},
    {'id': 2, 'texto': 'Hello world', 'tipo': 'saludo'},
    {'id': 3, 'texto': 'Bonjour monde', 'tipo': 'saludo'}
]

def filtrar_por_idioma(mensajes: List[Dict], texto_buscar: str) -> List[Dict]:
    return [m for m in mensajes if texto_buscar.lower() in m['texto'].lower()]

@app.get("/datos")
def endpoint_datos(buscar: str = Query(None, description="Texto para filtrar mensajes")):
    """
    Endpoint que opera sobre datos (paradigma orientado a datos).
    Filtra mensajes según el texto de búsqueda.
    """
    if buscar:
        mensajes_filtrados = filtrar_por_idioma(mensajes_db, buscar)
    else:
        mensajes_filtrados = mensajes_db
    
    return {
        "total": len(mensajes_filtrados),
        "mensajes": mensajes_filtrados,
        "paradigma": "orientado a datos"
    }


### Endpoint 3: Paradigma Orientado a Objetos
class SaludoEncapsulado:
    """Clase con atributos privados y métodos getter/setter"""
    
    def __init__(self, mensaje: str = "Hola mundo"):
        self.__mensaje = mensaje  # Atributo privado
    
    def get_mensaje(self) -> str:
        return self.__mensaje
    
    def set_mensaje(self, nuevo_mensaje: str):
        self.__mensaje = nuevo_mensaje
    
    def obtener_info(self) -> Dict:
        return {
            "mensaje": self.__mensaje,
            "longitud": len(self.__mensaje),
            "mayusculas": self.__mensaje.upper()
        }

@app.post("/objetos")
def endpoint_objetos(mensaje: str = Query("Hola mundo", description="Mensaje del saludo")):
    """
    Endpoint que usa objetos con encapsulación (paradigma orientado a objetos).
    Crea un objeto SaludoEncapsulado y retorna su información.
    """
    saludo = SaludoEncapsulado(mensaje)
    return {
        "info": saludo.obtener_info(),
        "paradigma": "orientado a objetos"
    }


@app.get("/")
def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "API de Paradigmas de Programación",
        "endpoints": {
            "funcional": "/funcional/transformar",
            "datos": "/datos/mensajes",
            "objetos": "/objetos/saludo"
        }
    }