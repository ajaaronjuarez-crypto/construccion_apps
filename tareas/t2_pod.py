# Opción 2: Separación de datos y comportamiento
# Los datos están separados de las funciones que los procesan

# Datos
mensajes = [
    {'id': 1, 'texto': 'Hola mundo', 'tipo': 'saludo'},
    {'id': 2, 'texto': 'Hello world', 'tipo': 'saludo'},
    {'id': 3, 'texto': 'Bonjour monde', 'tipo': 'saludo'}
]

# Funciones que operan sobre los datos
def filtrar_por_idioma(mensajes, texto_buscar):
    return [m for m in mensajes if texto_buscar in m['texto']]

def imprimir_mensajes(mensajes):
    for mensaje in mensajes:
        print(f"{mensaje['texto']}")

# Procesamiento
mensaje_español = filtrar_por_idioma(mensajes, 'Hola')
imprimir_mensajes(mensaje_español)