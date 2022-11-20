""" Modulo de test de la carga de modelos"""
import transformers
from texttools import models

def test_download_base():
    """ Este test comprueba que se ejecutan bien los valores por
     defecto de esta funcion asertando los tipos de modelo"""
    model,tokenizer = models.download_base()
    assert isinstance(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel)
    assert isinstance(tokenizer, transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast)

def test_load_model_and_tokenizer():
    """ Este test comprueba que se cargan los modelos
     guardados asertando los tipos de modelo """
    model,tokenizer = models.load_model_and_tokenizer()
    assert isinstance(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel)
    assert isinstance(tokenizer, transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast)

def test_load_pipeline():
    """ Este test comprueba que se carga bien la pipeline comprobando que
     la task que se carga es text-generation """
    pipeline = models.load_pipeline()
    assert pipeline.task == "text-generation"
