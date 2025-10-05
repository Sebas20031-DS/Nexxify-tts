import os
import re
from num2words import num2words
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

def normalize_numbers(text: str) -> str:
    """
    Convierte números en el texto a su representación en palabras (en español).
    """
    def replacer(match):
        num = int(match.group(0))
        return num2words(num, lang="es")

    # Detecta números enteros
    return re.sub(r"\d+", replacer, text)

def normalize_text(text: str) -> str:
    """
    Normaliza el texto: minúsculas, elimina símbolos no soportados, 
    convierte números a palabras y quita espacios extra.
    """
    text = text.lower()
    text = normalize_numbers(text)
    
    # Conserva letras españolas, números y signos de puntuación útiles
    text = re.sub(r"[^a-záéíóúüñ0-9,.!?;:()'\" ]", "", text)
    
    text = re.sub(r"\s+", " ", text)  # quita espacios extra
    return text.strip()

def text_to_phonemes(text: str, backend: EspeakBackend) -> str:
    """
    Convierte texto normalizado en una secuencia de fonemas en español.
    Usa espeak-ng como backend, pasado como parámetro.
    """
    # Phonemiza el texto normalizado usando el backend ya instanciado
    phonemes = backend.phonemize(
        [text],  # El texto ya está en una lista de strings
        separator=Separator(phone=' ', word=None),  # Los fonemas estarán separados por espacio
        strip=True  # Quita espacios extras
    )
    return phonemes[0] # Devuelve la secuencia fonética completa

def tokenize_phonemes(phoneme_text: str):
    """
    Convierte secuencia de fonemas en tokens individuales.
    Cada fonema debe ser un token independiente.
    """
    return phoneme_text.split()  # Separa los fonemas por espacio


def tokenize_phonemes_with_punctuation(phoneme_text: str):
    """
    Convierte secuencia de fonemas en tokens individuales, manteniendo la puntuación
    como tokens separados.
    """
    # Aquí separaremos los fonemas y la puntuación usando una expresión regular que captura todos los signos de puntuación
    phonemes_with_punctuation = re.split(r'([,\.!?;:¿¡()"\'\-])', phoneme_text)

    # Filtramos cualquier espacio vacío
    phonemes = [token for token in phonemes_with_punctuation if token.strip()]

    return phonemes


def text_to_ids(tokens, vocab):
    """
    Convierte tokens a IDs usando vocabulario.
    Se filtran los tokens que no estén en el vocabulario para evitar errores en batches.
    """
    ids = [vocab[token] for token in tokens if token in vocab]
    if len(ids) == 0:
        raise ValueError(f"Tokens vacíos después de mapear a IDs: {tokens}")
    return ids
