""" Modulo para cargar los modelos """
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.cuda import is_available as cuda_available


def download_base(model="DeepESP/gpt2-spanish", save=False):
    """ Función para descargar y guardar cualquier modelo de lenguaje """

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    if save:
        model.save_pretrained("./texttools/model/")
        tokenizer.save_pretrained("./texttools/model/")
    return model, tokenizer

def load_model_and_tokenizer():
    """ Funcion para cargar los modelos guardados """

    model = AutoModelForCausalLM.from_pretrained("./texttools/model/")
    tokenizer = AutoTokenizer.from_pretrained("./texttools/model/")
    return model, tokenizer

def load_pipeline():
    """ Funcion para cargar la tuberia de linea con los modelos guardados """
    device = 0 if cuda_available() else -1
    model, tokenizer = load_model_and_tokenizer()
    return pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    framework="pt",
                    device=device)
