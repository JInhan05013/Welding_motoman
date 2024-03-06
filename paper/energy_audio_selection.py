import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

# Function to calculate the energy of a signal
def energy(signal):
    return np.sum(np.power(signal, 2)) / float(len(signal))

for n in range(2,24):
 
    data_dir=f'../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41/layer_{n}/'
    # Microphone data
    wavfile = wave.open(data_dir + 'mic_recording.wav', 'rb')
    samplerate = 44000  # Fixed sample rate
    audio_data = np.frombuffer(wavfile.readframes(wavfile.getnframes()), dtype=np.int16)
    wavfile.close()  # It's a good practice to close the file
    # Define the frame size for energy calculation
    frame_size = samplerate // 10  # Frame size of 100 ms

    # Split the signal into frames and calculate the energy of each frame
    energies = []
    for i in range(0, len(audio_data) - frame_size + 1, frame_size):
        frame = audio_data[i:i+frame_size]
        energies.append(energy(frame))

    # Convert the list of energies to a numpy array
    energies = np.array(energies)

    # Normalize the energies
    energies /= np.max(energies)

    # Select frames with energy above the threshold (e.g., 0.1)
    threshold = 0.1
    high_energy = energies > threshold

    # Create an array of frame start times
    frame_times = np.arange(0, len(audio_data) - frame_size + 1, frame_size) / samplerate

    # Ensure that frame_times is the same length as high_energy
    frame_times = frame_times[:len(high_energy)]

    # Plotting the original audio data and highlighting high-energy frames
    times = np.arange(audio_data.size) / samplerate
    plt.figure(figsize=(15, 5))
    plt.plot(times, audio_data, label='Original Audio Data')
    plt.plot(frame_times[high_energy], np.zeros(sum(high_energy)), 'r|', ms=10, label='High Energy Frames')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Data with High-Energy Regions Highlighted')
    plt.legend()
    plt.show()