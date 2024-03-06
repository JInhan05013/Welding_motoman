import numpy as np
import pickle
import matplotlib.pyplot as plt

# 载入字典文件
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def extract_and_average(lst, height_length):
    midpoint = len(lst) // 2
    half_height = height_length // 2
    start = int(midpoint - half_height)  # 明确转换为整数
    end = int(midpoint + half_height + 1)  # 明确转换为整数，并加1因为结束索引是不包含的
    selected_values = lst[start:end]
    average = sum(selected_values) / len(selected_values)
    return average

def find_top_segment_pairs(data):
    top_segment_pairs = {}
    for layer_number, (layer, layer_info) in enumerate(data.items()):
        # 跳过前两层
        if layer_number < 2:
            continue
        
        segments = layer_info['segments']
        # 考虑排除前8个和后8个segments
        valid_segments = segments[8:-8]
        num_valid_segments = len(valid_segments)
        
        height_diffs = np.zeros((num_valid_segments, num_valid_segments))
        v = layer_info['layer_velocity']
        height_length =int(v * 0.05 * 10)
        print('height_length',height_length)
        # 计算有效段对之间的高度差异
        for i in range(num_valid_segments):
            for j in range(i+1, num_valid_segments):
                diff = np.abs(extract_and_average(valid_segments[i]['height_segment'], height_length) - extract_and_average(valid_segments[j]['height_segment'], height_length))
                height_diffs[i][j] = diff
        
        # 找到高度差异最大的5对段
        indices = np.dstack(np.unravel_index(np.argsort(-height_diffs.ravel()), (num_valid_segments, num_valid_segments)))[0][:5]
        # 因为我们排除了前8个segments，所以需要加上8来得到原始的索引
        adjusted_indices = indices + 8
        top_segment_pairs[layer] = adjusted_indices
        
    return top_segment_pairs

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

def plot_and_print_segment_pairs(data, top_segment_pairs):
    for layer, pairs in top_segment_pairs.items():
        print(f'Layer {layer} - Spectral Centroid Differences for Top 5 Segment Pairs:')
        fig, axs = plt.subplots(5, 2, figsize=(15, 15))
        fig.suptitle(f'Layer {layer} - Top 5 Segment Pairs')

        for i, (idx1, idx2) in enumerate(pairs):
            segment_info_1 = data[layer]['segments'][idx1]
            segment_info_2 = data[layer]['segments'][idx2]

            # FFT and Spectral Centroid Calculation
            xf1, yf1 = compute_fft(segment_info_1['audio_feature'])
            xf2, yf2 = compute_fft(segment_info_2['audio_feature'])
            centroid1 = spectral_centroid(xf1, yf1)
            centroid2 = spectral_centroid(xf2, yf2)
            centroid_diff = np.abs(centroid1 - centroid2)

            # 打印频谱质心的差值
            print(f'Segment Pair {i+1}: {idx1} vs {idx2}, Centroid Difference: {centroid_diff:.2f}')

            # FFT plots
            axs[i, 0].plot(xf1, yf1, label=f'Segment {idx1}')
            axs[i, 0].plot(xf2, yf2, label=f'Segment {idx2}')
            axs[i, 0].set_title(f'FFT Audio Comparison: Segments {idx1} vs {idx2}')
            axs[i, 0].legend()

            # Height plots
            axs[i, 1].plot(segment_info_1['height_segment'], label=f'Segment {idx1}')
            axs[i, 1].plot(segment_info_2['height_segment'], label=f'Segment {idx2}')
            axs[i, 1].set_title(f'Height Data Comparison: Segments {idx1} vs {idx2}')
            axs[i, 1].legend()

        plt.tight_layout()
        plt.show()
# 主函数
def main():
    file_path = 'Test_data/ER4043_model_150ipm_2023_10_08_08_23_41.pkl'  # 更新为您的文件路径
    data = load_data(file_path)
    top_segments = find_top_segment_pairs(data)
    plot_and_print_segment_pairs(data, top_segments)
    for layer, indices in top_segments.items():
        print(f'{layer}: Top segments indices: {indices}')

if __name__ == "__main__":
    main()
