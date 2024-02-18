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
    return layers_data,audio_data_total,height_data_total

# 绘制数据并使用高度差信号的长度作为x轴
def plot_data(layers_data, sample_distance=0.1):
    for i, (audio_data, height_data) in enumerate(layers_data):
        # 每隔一层声音信号与高度差信号方向相反
        if i % 2 == 1:
            audio_data = np.flip(audio_data)

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
samplerate = 44100
layers_data,audio_data_total,height_data_total = extract_data(data_dir)

plot_data(layers_data)