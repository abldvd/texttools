""" Modulo para cargar los modelos """
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.cuda import is_available as cuda_available


def download_base(model="DeepESP/gpt2-spanish", save=False):
    """ Función para descargar y guardar cualquier modelo de lenguaje """

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    if save:
        with(open("./texttools/model/model.pkl", "wb")) as file:
            pickle.dump(model, file)
        with(open("./texttools/model/tokenizer.pkl", "wb")) as file:
            pickle.dump(tokenizer, file)

    return model, tokenizer

def load_model_and_tokenizer():
    """ Funcion para cargar los modelos guardados """

    with(open("./texttools/model/model.pkl", "rb")) as file:
        model = pickle.load(file)
    with(open("./texttools/model/tokenizer.pkl", "rb")) as file:
        tokenizer = pickle.load(file)

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
