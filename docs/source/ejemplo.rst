Ejemplo
=======

Para realizar la inferencia importar generate de texttools.inference 
e introducir un texto junto a la cantidad máxima de tokens que se desee

.. code-block:: python

    from texttools.inference import generate
    out = generate(promt, max_tokens)