""" Modulo para realizar la inferencia del modelo """

from texttools.models import load_pipeline

def generate(text: str, max_tokens=50, pipeline=load_pipeline()) -> str:
    """ Funcion que genera texto a partir de un input inicial y un numero máximo de palabras """
    return pipeline(text, max_length=max_tokens)[0]['generated_text']
