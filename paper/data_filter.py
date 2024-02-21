import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize_audio(audio):
    audio = np.array(audio, dtype=np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

data_dir = '../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41/'
layer_number = 'layer_0'
# Read .wav
fs, data = wavfile.read(data_dir + layer_number + '/mic_recording.wav')

# Input filter range
lowcut = 10000 
highcut = 20000 

# Apply filter
filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)

# Normalization
normalized_data = normalize_audio(filtered_data)

# Save filted data
wavfile.write('filtered_recording.wav', fs, (normalized_data * 32767).astype(np.int16))