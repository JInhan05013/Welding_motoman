import os, math, librosa
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import scipy.fftpack
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import mean_squared_error

data_dir = '../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41'

# 遍历文件夹提取数据
def extract_data(data_dir):
    layers_data = []  # 存储每层的数据
    audio_data_total = []
    height_data_total = []
    layer_index = 0  # 用于跟踪当前层的索引
    for folder_name in sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1])):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            audio_file_path = os.path.join(folder_path, 'welding_audio_data.npy')
            height_file_path = os.path.join(folder_path, 'height_difference.npy')
            if os.path.exists(audio_file_path) and os.path.exists(height_file_path):
                audio_data = np.load(audio_file_path)
                height_data = np.load(height_file_path)
                # 每隔一层对audio_data进行翻转
                if layer_index % 2 == 1:  # 如果是奇数层，则翻转audio_data
                    audio_data = np.flip(audio_data, axis=0)  # 假设我们沿着时间轴翻转数据
                
                layers_data.append((audio_data, height_data))
                audio_data_total.append(audio_data)
                height_data_total.append(height_data)
                
                layer_index += 1  # 更新层索引
                
    return layers_data, audio_data_total, height_data_total

# 绘制数据并使用高度差信号的长度作为x轴
def plot_data(layers_data, sample_distance=0.1):
    for i, (audio_data, height_data) in enumerate(layers_data):

        # 根据高度差信号的长度创建x轴
        x_axis = np.arange(len(height_data)) * sample_distance
        # 将声音信号长度调整至与高度差信号长度一致
        audio_x_axis = np.linspace(0, x_axis[-1], len(audio_data))

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Height Difference', color=color)
        ax1.plot(x_axis, height_data, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # 实例化一个第二个y轴
        color = 'tab:blue'
        ax2.set_ylabel('Audio Signal', color=color)
        ax2.plot(audio_x_axis, audio_data, color=color, alpha=0.5)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Layer {i+2}')
        plt.show()

def extract_features(audio_signal, height_signal, sr=44100, segment_length_mm=0.9, sample_distance_mm=0.1):
    # 假设高度差信号每个样本间距0.1mm，每个segment对应的样本数量
    segment_samples = int(segment_length_mm / sample_distance_mm)
    audio_features = []
    height_features = []
    audio_segment_length = int(len(audio_signal) / (len(height_signal) / segment_samples))
    for start_idx in range(0, len(height_signal), segment_samples):
        end_idx = start_idx + segment_samples
        if end_idx > len(height_signal):
            break
        
        # 高度差信号特征
        height_segment = height_signal[start_idx:end_idx]
        height_mean = np.mean(height_segment)
        height_std = np.std(height_segment)
        height_features.append([height_mean, height_std])
        
        # 声音信号特征
        audio_start_idx = int(start_idx / segment_samples * audio_segment_length)
        audio_end_idx = int(end_idx / segment_samples * audio_segment_length)
        audio_segment = audio_signal[audio_start_idx:audio_end_idx]
        
        # 计算Mel频谱/MFCC
        # mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr)
        # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # print('mel_spec_db',mel_spec_db)

        reduced_noise_audio = nr.reduce_noise(y=audio_segment, sr=samplerate)
        # 1. 计算FFT
        n_mfcc = 13
        frame = reduced_noise_audio
        fft_result = np.abs(scipy.fftpack.fft(frame))
        fft_result_half = fft_result[:len(frame) // 2]
        # 2. 应用梅尔滤波器
        n_fft = (len(fft_result_half) - 1) * 2
        mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=40)
        mel_spec = np.dot(mel_filter, np.square(fft_result_half))
        
        # 计算对数梅尔频谱
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # 应用离散余弦变换得到MFCC
        mfcc = scipy.fftpack.dct(log_mel_spec, type=2, norm='ortho')[:n_mfcc]
        # 计算平均频率和标准差
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
        freq_mean = np.mean(spectral_centroids)
        freq_std = np.std(spectral_centroids)
        audio_features.append([mfcc[0] ,freq_mean, freq_std])
        
    return audio_features, height_features

def split_dataset(audio_features, height_features, test_size=0.2):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(audio_features, height_features, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='linear')  # 假设输出是高度差的平均值和标准差
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def visualize_predictions(y_true, y_pred, title='Predicted vs Actual Values'):
    plt.figure(figsize=(14, 6))

    # Assuming first column is the mean and second column is the std deviation
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'k--', lw=2)
    plt.xlabel('Actual Mean (mm)')
    plt.ylabel('Predicted Mean (mm)')
    plt.title('Mean Comparison')

    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].min(), y_true[:, 1].max()], 'k--', lw=2)
    plt.xlabel('Actual Std Deviation')
    plt.ylabel('Predicted Std Deviation')
    plt.title('Std Deviation Comparison')

    plt.suptitle(title)
    plt.show()

samplerate = 44100
layers_data,audio_data_total,height_data_total = extract_data(data_dir)

# plot_data(layers_data)
audio_data_list_raw = np.concatenate(audio_data_total)
audio_data_list = audio_data_list_raw.astype('float32') / np.max(np.abs(audio_data_list_raw))
height_data_list = np.concatenate(height_data_total)
audio_data_list = np.array(audio_data_list, dtype=np.float32)
height_data_list = np.array(height_data_list, dtype=np.float32)

# 提取和处理数据
audio_features, height_features = extract_features(audio_data_list,height_data_list, sr=44100, segment_length_mm=1.5, sample_distance_mm=0.1)
np.save('audio_features.npy',audio_features)
np.save('height_features.npy',height_features)
# print("audio_features:", audio_features)
print("audio_features.size:", np.size(audio_features))
print("audio_features.len:", len(audio_features))
# print("height_features:", height_features)
print("height_features.size:", np.size(height_features))
print("height_features.len:", len(height_features))
# quit()
# 示例: 数据分割 (这里需要先实际处理数据)
X_train, X_test, y_train, y_test = split_dataset(np.array(audio_features), np.array(height_features), test_size=0.2)
np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)
quit()
# 构建模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),  # Adjusted to match your feature vector size
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='linear')  # Outputs two features: mean and std deviation of height differences
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test dataset
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print evaluation results
print("MSE: ", mse)
print("RMSE: ", rmse)



visualize_predictions(y_test, predictions)