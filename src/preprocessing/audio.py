import librosa
import numpy as np

def load_audio(path, sr=22050):
    """
    Carga audio y resamplea a sample rate deseado.
    Tacotron2 espera todos los audios al mismo sr.
    """
    audio, _ = librosa.load(path, sr=sr)
    return audio

def audio_to_mel(audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    Convierte audio a espectrograma Mel.
    Luego normaliza a rango [-1, 1] para estabilidad en el entrenamiento de Tacotron2.
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalización a [-1, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())  # [0,1]
    mel_norm = mel_norm * 2 - 1  # [-1,1]
    
    return mel_norm.astype(np.float32)
