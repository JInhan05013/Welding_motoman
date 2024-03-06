import numpy as np
import pickle
import matplotlib.pyplot as plt

# 载入字典文件
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def compute_fft(audio_segment, samplerate=44000):
    # 计算频域数据
    N = len(audio_segment)
    T = 1.0 / samplerate
    yf = np.fft.fft(audio_segment)
    xf = np.fft.fftfreq(N, T)[:N//2]
    return xf, 2.0/N * np.abs(yf[0:N//2])

def spectral_centroid(xf, yf):
    # 计算频谱质心
    normalized_spectrum = yf / np.sum(yf)
    centroid = np.sum(xf * normalized_spectrum)
    return centroid

def plot_segments(data, layer_nums, segment_nums):
    plt.figure(figsize=(15, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(segment_nums)))
    
    for i, (layer_num, segment_num) in enumerate(zip(layer_nums, segment_nums)):
        layer_key = list(data.keys())[layer_num]
        segment_info = data[layer_key]['segments'][segment_num]

        # FFT and Spectral Centroid Calculation
        xf, yf = compute_fft(segment_info['audio_feature'])
        centroid = spectral_centroid(xf, yf)

        # Plotting
        plt.subplot(1, 2, 1)
        plt.plot(xf, yf, label=f'Layer {layer_num} Segment {segment_num}', color=colors[i])
        plt.title('FFT Audio Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(segment_info['height_segment'], label=f'Layer {layer_num} Segment {segment_num}', color=colors[i])
        plt.title('Height Difference Comparison')
        plt.xlabel('Sample Number')
        plt.ylabel('Height (mm)')
        plt.legend()

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    file_path = 'Test_data/ER4043_model_150ipm_2023_10_08_08_23_41.pkl'  # 更新为您的文件路径
    data = load_data(file_path)

    # 用户输入
    layer_nums = [4, 9]  # 示例层数
    segment_nums = [25, 25]  # 示例段序号

    plot_segments(data, layer_nums, segment_nums)

if __name__ == "__main__":
    main()
