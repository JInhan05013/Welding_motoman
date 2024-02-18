import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = '../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41'

# 遍历文件夹提取数据
def extract_data(data_dir):
    layers_data = []  # 存储每层的数据
    audio_data_total = []
    height_data_total = []
    for folder_name in sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1])):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            audio_file_path = os.path.join(folder_path, 'welding_audio_data.npy')
            height_file_path = os.path.join(folder_path, 'height_difference.npy')
            if os.path.exists(audio_file_path) and os.path.exists(height_file_path):
                audio_data = np.load(audio_file_path)
                height_data = np.load(height_file_path)
                layers_data.append((audio_data, height_data))
                audio_data_total.append(audio_data)
                height_data_total.append(height_data)
    return layers_data, audio_data_total, height_data_total

# 绘制数据并使用高度差信号的长度作为x轴
def plot_data(layers_data, sample_distance=0.1):
    for i, (audio_data, height_data) in enumerate(layers_data):
        x_axis_length = len(height_data) * sample_distance
        if i % 2 == 1:
            # 反向时，调整x轴刻度使其显示为70到0
            x_axis = np.linspace(x_axis_length, 0, len(height_data))
        else:
            # 正向时，x轴刻度从0到70
            x_axis = np.linspace(0, x_axis_length, len(height_data))
        
        audio_x_axis = np.linspace(x_axis[0], x_axis[-1], len(audio_data))

        fig, ax1 = plt.subplots(figsize=(10, 6))  # 调整图表尺寸以便于观察

        color = 'tab:red'
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Height Difference (mm)', color=color)
        ax1.plot(x_axis, height_data, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Audio Signal', color=color)
        ax2.plot(audio_x_axis, audio_data, color=color, alpha=0.5)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Layer {i+2}')
        plt.show()

samplerate = 44100
layers_data, audio_data_total, height_data_total = extract_data(data_dir)

plot_data(layers_data)
