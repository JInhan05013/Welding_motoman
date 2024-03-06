import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

# Load data
with open('Test_data/ER4043_model_150ipm_2023_10_08_08_23_41.pkl', 'rb') as file:
    layers_data = pickle.load(file)

sampling_rate = 44000

# Layers iteration
for layer_name in layers_data.keys():
    data = layers_data[layer_name]
    # Segments iteration
    for segment_index, segment_data in enumerate(data['segments']):
        audio_feature = segment_data['audio_feature']

        # Ensure audio data is in float format
        if not isinstance(audio_feature, np.ndarray):
            audio_feature = np.array(audio_feature)
        if audio_feature.dtype != np.float32:
            # Convert to float and normalize to [-1, 1] if needed
            audio_feature = audio_feature.astype(np.float32)
            if audio_feature.max() > 1 or audio_feature.min() < -1:
                audio_feature /= np.max(np.abs(audio_feature), axis=0)

        # Calculate STFT
        D = librosa.stft(audio_feature, n_fft=2048, hop_length=512)
        
        # Transfer to DB
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        segment_data['spectrogram'] = S_db
        # Calculate Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_feature, sr=sampling_rate, n_fft=2048, hop_length=512)[0]
        # print(f'Spectral Centroid of {layer_name} Segment {segment_index}: ',spectral_centroids )
        # Calculate Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_feature, frame_length=2048, hop_length=512)[0]
        # print(f'ZCR of {layer_name} Segment {segment_index}: ',zcr)
        # Calculate the mean of Spectral Centroids
        spectral_centroid_mean = np.mean(spectral_centroids)
        # Save the mean value to the segment data
        segment_data['spectral_centroid_mean'] = spectral_centroid_mean
        # Calculate RMS
        rms = librosa.feature.rms(y=audio_feature, frame_length=2048, hop_length=512)[0]
        # print(f'rms of {layer_name} Segment {segment_index}: ',rms)
        # Convert frame counts to time (sec) for plotting
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sampling_rate, hop_length=512)

        # # Plot Spectrogram with Spectral Centroid and Zero Crossing Rate
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(S_db, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='log', cmap='viridis')
        # plt.colorbar(format='%+2.0f dB')
        # plt.plot(t, spectral_centroids, color='w', label='Spectral Centroid')
        # plt.plot(t, zcr * np.max(spectral_centroids), color='r', label='Zero Crossing Rate (scaled)')
        # plt.title(f'Spectrogram, Spectral Centroid, and ZCR of {layer_name} Segment {segment_index}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Frequency (Hz)')
        # plt.legend(loc='upper right')
        # plt.show()

        # # Plot RMS separately
        # plt.figure(figsize=(10, 2))
        # plt.plot(t, rms, color='r', label='RMS')
        # plt.title(f'RMS of {layer_name} Segment {segment_index}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.legend(loc='upper right')
        # plt.show()

        with open('Test_data/ER4043_model_150ipm_2023_10_08_08_23_41_with_centroids.pkl', 'wb') as file:
            pickle.dump(layers_data, file)    