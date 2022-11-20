""" Modulo para hacer el test de la inferencia """
from texttools.inference import generate

def test_generate():
    """ Funcion para comprobar que funciona bien la inferencia 
    comprobando que devuelve más tokens de los que entran"""
    out = generate("Hola qué", 5)
    assert len(out.split(" ")) > 2
