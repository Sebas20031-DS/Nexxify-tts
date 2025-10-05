from src.preprocessing.text import normalize_text, text_to_phonemes, tokenize_phonemes, tokenize_phonemes_with_punctuation
from phonemizer.backend import EspeakBackend

# Inicializa el backend de espeak una vez, para ser reutilizado en todo el proceso.
backend = EspeakBackend("es", preserve_punctuation=True, with_stress=True)


texts = [
    "Ablándole con cariño.", 
    "Quiero hablar de la armada."
]

for t in texts:
    nt = normalize_text(t)
    ph = text_to_phonemes(nt, backend)
    toks = tokenize_phonemes_with_punctuation(ph)
    print("ORIGINAL:", t)
    print("NORMALIZADO:", nt)
    print("PHONEMES:", ph)
    print("TOKENS:", toks)
    print("----")
