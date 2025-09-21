import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ0-9,.!?;:()'\" ]", "", text)  # limpia símbolos raros
    return text.strip()

def tokenize(text: str):
    return list(text)  # tokenización a nivel de caracteres (simple)

def text_to_ids(tokens, vocab):
    return [vocab[token] for token in tokens if token in vocab]
