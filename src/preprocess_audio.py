import librosa
import numpy as np

def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def audio_to_mel(audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(spectrogram, ref=np.max)
    return mel_db
