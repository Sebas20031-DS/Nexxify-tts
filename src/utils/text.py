import re

def normalize_text(text: str) -> str:
    """
    Normaliza el texto: minúsculas, elimina símbolos no soportados, quita espacios extra.
    Es importante porque Tacotron2 espera una secuencia consistente de tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ0-9,.!?;:()'\" ]", "", text)
    return text.strip()

def tokenize(text: str):
    """
    Convierte texto en tokens de caracteres.
    Tacotron2 trabaja a nivel de caracteres por defecto.
    """
    return list(text)

def text_to_ids(tokens, vocab):
    """
    Convierte tokens a IDs usando vocabulario.
    Se filtran los tokens que no estén en el vocabulario para evitar errores en batches.
    """
    ids = [vocab[token] for token in tokens if token in vocab]
    if len(ids) == 0:
        raise ValueError(f"Tokens vacíos después de mapear a IDs: {tokens}")
    return ids
