import numpy as np
import soundfile as sf

# ==========================
# CONFIGURACIÓN
# ==========================

WINDOW_SIZE = 4096
HOP_SIZE = WINDOW_SIZE // 2

BANDS = [
    ("Sub", 20, 60),
    ("Graves", 60, 250),
    ("Medios", 250, 4000),
    ("Agudos", 4000, 20000),
]

# ==========================
# CARGAR AUDIO
# ==========================

def load_audio(filename):

    audio, sr = sf.read(filename)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float64), sr


# ==========================
# FFT POR VENTANAS
# ==========================

def stft_average(audio, sr):

    window = np.hanning(WINDOW_SIZE)

    spectrum_sum = None
    windows_used = 0

    for start in range(0, len(audio)-WINDOW_SIZE, HOP_SIZE):

        frame = audio[start:start+WINDOW_SIZE]

        rms = np.sqrt(np.mean(frame**2))

        # Ignorar silencios
        if rms < 1e-4:
            continue

        frame = frame * window

        fft = np.fft.rfft(frame)

        power = np.abs(fft)**2

        if spectrum_sum is None:
            spectrum_sum = power
        else:
            spectrum_sum += power

        windows_used += 1

    if windows_used == 0:
        raise Exception("No hay suficiente audio.")

    spectrum_avg = spectrum_sum / windows_used

    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1/sr)

    return freqs, spectrum_avg


# ==========================
# DISTRIBUCIÓN TONAL
# ==========================

def tonal_distribution(freqs, spectrum):

    energies = []

    for _, low, high in BANDS:

        idx = np.where((freqs >= low) &
                       (freqs < high))[0]

        energies.append(np.sum(spectrum[idx]))

    total = np.sum(energies)

    result = {}

    for i, (name, _, _) in enumerate(BANDS):

        result[name] = energies[i] / total * 100

    return result


# ==========================
# ANALIZADOR
# ==========================

def analyze(filename):

    audio, sr = load_audio(filename)

    freqs, spectrum = stft_average(audio, sr)

    tonal = tonal_distribution(freqs, spectrum)

    return tonal