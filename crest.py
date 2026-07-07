import numpy as np
import soundfile as sf


def crest_factor(filename):

    audio, sr = sf.read(filename)

    # Convertir a mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)

    peak = np.max(np.abs(audio))

    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return 0.0

    crest = 20 * np.log10(peak / rms)

    return crest