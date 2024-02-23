import pickle
import matplotlib.pyplot as plt
import numpy as np
# Load data
with open('Test_data/ER4043_model_150ipm_2023_10_08_08_23_41.pkl', 'rb') as file:
    layers_data = pickle.load(file)

sampling_rate = 44000

# Switch color
colors = plt.cm.viridis(np.linspace(0, 1, 60))  

plt.figure(figsize=(10, 6))

# Layer iteration
for layer_name in layers_data.keys():
    data = layers_data[layer_name]
    # Segments iteration
    for segment_index, segment_data in enumerate(data['segments']):
        audio_feature = segment_data['audio_feature']

        # FFT Transfer
        fft_result = np.fft.fft(audio_feature)
        n = len(audio_feature)
        frequency = np.fft.fftfreq(n, d=1/sampling_rate)

        # Magnitude calculation
        magnitude = np.abs(fft_result)

        # Plot 
        plt.plot(frequency[:n // 2], magnitude[:n // 2], color=colors[segment_index], label=f'Layer {layer_name}, Segment {segment_index}')

        # Plot parameters
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'FFT of Audio Signals Across {layer_name}')
        plt.grid(True)
        # plt.legend() 
        plt.show()
        plt.close()