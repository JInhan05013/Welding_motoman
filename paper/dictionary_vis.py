import pickle
import matplotlib.pyplot as plt

# Load data
with open('Test_data/ER4043_model_150ipm_2023_10_08_08_23_41.pkl', 'rb') as file:
    layers_data = pickle.load(file)

# Input the number of layer
layer_name = list(layers_data.keys())[5]
data = layers_data[layer_name]


# Visualize welding_audio_data
plt.plot(data['welding_audio_data'])
plt.title(f'Welding Audio Data - {layer_name}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
# plt.show()

# Visualize height_difference
plt.plot(data['height_difference'])
plt.title(f'Height Difference - {layer_name}')
plt.xlabel('Sample')
plt.ylabel('Height (mm)')
# plt.show()
plt.close()

segment_index = 4  # 5th segment
segment_data = layers_data[layer_name]['segments'][segment_index]

audio_segment = segment_data['audio_segment']  
height_segment = segment_data['height_segment'] 
audio_feature = segment_data['audio_feature']

# Print data to make sure
print("Audio Segment Data:", audio_segment)
print("Height Segment Data:", height_segment)
print("audio_feature:", audio_feature)
# Plot audio data
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.plot(audio_segment)
plt.title('Audio Data of 5th Segment in Layer 5')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(1, 3, 2)
plt.plot(audio_feature)
plt.title('Audio feature of 5th Segment in Layer 5')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Plot height data
plt.subplot(1, 3, 3)
plt.plot(height_segment)
plt.title('Height Data of 5th Segment in Layer 5')
plt.xlabel('Sample Index')
plt.ylabel('Height Difference (mm)')

plt.tight_layout()
plt.show()