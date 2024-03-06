import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

for n in range(2,24):
    try:
        data_dir=f'../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_{n}/'
        # Microphone data
        wavfile = wave.open(data_dir + 'mic_recording.wav', 'rb')
        samplerate = 44000  # Fixed sample rate
        audio_data = np.frombuffer(wavfile.readframes(wavfile.getnframes()), dtype=np.int16)
        wavfile.close()  # It's a good practice to close the file

        # Generate spectrogram
        freq, mic_ts, Sxx = signal.spectrogram(audio_data, samplerate)

        # Calculate mean frequency
        mean_freq = np.sum(Sxx * np.arange(Sxx.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx, axis=0)

        # Determine indices for the time range
        start_index = np.searchsorted(mic_ts, mic_ts[0] + 4)  # Index after 4 seconds
        end_index = np.searchsorted(mic_ts, mic_ts[-1] - 1)   # Index 1 second before the end

        # Filter timestamps and mean frequencies
        selected_mic_ts = mic_ts[start_index:end_index]
        selected_mean_freq = mean_freq[start_index:end_index]

        # Filter for mean_freq greater than 10
        selected_freq_indices = np.flatnonzero(selected_mean_freq > 10)

        # Initialize variables to avoid reference before assignment error
        start_welding_mic = end_welding_mic = welding_duration_mic = 0

        if len(selected_freq_indices) > 0:
            start_welding_mic = selected_mic_ts[selected_freq_indices[0]]
            end_welding_mic = selected_mic_ts[selected_freq_indices[-1]]
            start_sample_index = int(190752)
            end_sample_index = len(audio_data)-int(48000)
            start_time_sec = start_sample_index/samplerate
            end_time_sec = end_sample_index/samplerate
            welding_duration_mic = end_welding_mic - start_welding_mic
            welding_audio_data = audio_data[start_sample_index:end_sample_index]
            # print(f'---------------layer{n}-------------------')
            # print('start_time_sec',start_time_sec)
            # print('end_time_sec',end_time_sec)
            # print('start_sample_index',start_sample_index)
            # print('end_sample_index',end_sample_index)
            # print('total_sample_index',len(audio_data))
            # print('end_sample_distance',len(audio_data)-end_sample_index)
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(np.linspace(0, len(audio_data) / samplerate, num=len(audio_data)), audio_data, color='blue', label='Original Audio Data')
            plt.axvline(x=start_time_sec, color='red', linestyle='--', label='Start of Selected Region')
            plt.axvline(x=end_time_sec, color='red', linestyle='--', label='End of Selected Region')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Audio Data with Selected Region Highlighted for Layer {n}')
            plt.legend()
            plt.show()

            # Save the welding audio data
            np.save(data_dir + 'welding_audio_data.npy', welding_audio_data)
        else:
            print("No signal captured")
    except:
        continue