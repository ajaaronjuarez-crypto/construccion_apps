# Opción 3: Usando composición de funciones
def mayusculas(texto):
    return texto.upper()

def agregar_exclamacion(texto):
    return f"{texto}!"

# Composición
mensaje = "hola mundo"
resultado = agregar_exclamacion(mayusculas(mensaje))
print(resultado)