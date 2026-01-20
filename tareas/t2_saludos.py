# Opción 2: Clase con encapsulamientoß
class SaludoEncapsulado:
    """Clase con atributos privados y métodos getter/setter"""
    
    def __init__(self, mensaje="Hola Aaron"):
        self.__mensaje = mensaje  # Atributo privado
    
    def get_mensaje(self):
        return self.__mensaje
    
    def set_mensaje(self, nuevo_mensaje):
        self.__mensaje = nuevo_mensaje
    
    def imprimir(self):
        print(self.__mensaje)

def saludoAaron():
    saludo2 = SaludoEncapsulado()
    saludo2.imprimir()