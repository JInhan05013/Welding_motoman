import numpy as np
import os, re, pickle, wave, sys
import matplotlib.pyplot as plt
import scipy.signal as signal

# Main data path
parent_dir = '../data/wall_weld_test/'
test_dir = 'ER4043_model_150ipm_2023_10_08_08_23_41'
file_dir = parent_dir + test_dir + '/'
layers_data = {}

for n in range(0,24):
 
    data_dir=f'{file_dir}layer_{n}/'

    ## microphone data
    wavfile = wave.open(f'{data_dir}mic_recording.wav', 'rb')
    samplerate = 44000
    channels = 1
    audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
    freq, mic_ts, Sxx = signal.spectrogram(audio_data, samplerate)
    mean_freq = np.sum(Sxx * np.arange(Sxx.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx, axis=0)
    # Kick off time index of aduio data
    start_index = np.searchsorted(mic_ts, mic_ts[0] + 4)  # Welding pre-chopped
    end_index = np.searchsorted(mic_ts, mic_ts[-1] - 1)   
    # Filter time index
    selected_mic_ts = mic_ts[start_index:end_index]
    selected_mean_freq = mean_freq[start_index:end_index]
    # Main frequency filter
    selected_freq_indices = np.flatnonzero(selected_mean_freq > 10)

    # Calculate welding starting and ending time
    if len(selected_freq_indices) > 0:
        start_welding_mic = selected_mic_ts[selected_freq_indices[0]]
        end_welding_mic = selected_mic_ts[selected_freq_indices[-1]]
        start_time_sec = start_welding_mic  
        end_time_sec = end_welding_mic  
        start_sample_index = int(start_time_sec * samplerate)
        end_sample_index = int(end_time_sec * samplerate)
        welding_duration_mic = end_welding_mic - start_welding_mic
        welding_audio_data = audio_data[start_sample_index:end_sample_index]
        if n % 2 == 1:  # 如果是奇数层，则翻转audio_data
            welding_audio_data = np.flip(welding_audio_data, axis=0)  # 假设我们沿着时间轴翻转数据

        # print('np.size(welding_audio_data)',np.size(welding_audio_data))
        # np.save(data_dir+'welding_audio_data.npy',welding_audio_data)
    else:
        # 如果没有找到合适的频率，可能需要其他逻辑来处理
        print("No signal captured")

    print('------------------------------------------------')
    print(f'welding signal of {n} layer:')
    print('welding started:', start_welding_mic)
    print('welding ended:', end_welding_mic)
    print('welding duration time:', welding_duration_mic)
    freq_welding, mic_ts_welding, Sxx_welding = signal.spectrogram(welding_audio_data, samplerate)
    selected_mic_ts_welding = mic_ts_welding[start_index:end_index]
    welding_duration_mic_ext=mic_ts_welding[-1]-mic_ts_welding[0]
    mean_freq_welding = np.sum(Sxx_welding * np.arange(Sxx_welding.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx_welding, axis=0)

    ##height profile
    profile_height_current = np.load(data_dir + 'scans/height_profile.npy')
    # Load the previous layer's height profile if n > 1
    if n > 0:
        profile_height_previous = np.load(data_dir.replace(f'layer_{n}/', f'layer_{n-1}/') + 'scans/height_profile.npy')
        # Ensure both arrays are of the same length
        min_length = min(len(profile_height_current), len(profile_height_previous))
        profile_height_current = profile_height_current[:min_length]
        profile_height_previous = profile_height_previous[:min_length]
        # Calculate the difference in height
        height_difference = profile_height_current[:, 1] - profile_height_previous[:, 1]
    else:
        # For the first layer, there is no previous layer to compare to
        height_difference = profile_height_current[:, 1]  # Default to current height

    # print('np.size(height_difference)',np.size(height_difference))
    # np.save(data_dir+'height_difference.npy',height_difference)
    # 将处理好的数据添加到字典
    layers_data[f'layer_{n}'] = {
        'welding_audio_data': welding_audio_data,  # 或者是经过一些处理的音频数据变量
        'height_difference': height_difference,
        'segments': []  # 正确初始化'segments'键
    }
    # 高度差信号拆分
    segment_length_mm = 1.5  # 每个segment的长度为15mm
    samples_per_segment_height = int(segment_length_mm / 0.1)  # 每个高度差segment包含的样本数
    total_segments_height = int(np.ceil(len(height_difference) / samples_per_segment_height))  # 高度差信号拆分成的段数

    # 声音信号拆分
    # 计算声音信号应拆分为相同数量的segments
    samples_per_segment_audio = int(np.floor(len(welding_audio_data) / total_segments_height))  # 声音信号每个segment的样本数

    # 初始化存储segments的列表
    height_segments = []
    audio_segments = []

    # 拆分高度差信号和声音信号
    for i in range(total_segments_height):
        start_index_height = i * samples_per_segment_height
        end_index_height = start_index_height + samples_per_segment_height
        segment_height = height_difference[start_index_height:end_index_height]

        start_index_audio = i * samples_per_segment_audio
        end_index_audio = start_index_audio + samples_per_segment_audio
        segment_audio = welding_audio_data[start_index_audio:end_index_audio]

        # 确保最后一个segment包含剩余所有样本
        if i == total_segments_height - 1:
            segment_height = height_difference[start_index_height:]
            segment_audio = welding_audio_data[start_index_audio:]

        height_segments.append(segment_height)
        audio_segments.append(segment_audio)
        if len(segment_audio) >= 2200:
        # 找到中间的索引
            mid_index = len(segment_audio) // 2
        # 从中间向两侧各取1100个样本
            audio_feature = segment_audio[mid_index-1100:mid_index+1100]
        else:
        # 如果audio_segment的长度小于2200，则可以选择填充或者其他逻辑处理
        # 这里为了演示，我们简单地使用现有样本作为特征
            audio_feature = segment_audio

        # 将拆分后的segments存入字典
        layers_data[f'layer_{n}']['segments'].append({
            'segment_number': i,
            'audio_segment': segment_audio,
            'audio_feature': audio_feature,
            'height_segment': segment_height
        })

# 目标目录用于存储字典
target_dir = 'Test_data/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 保存字典到文件
dict_file_path = os.path.join(target_dir, f'{test_dir}.pkl')
with open(dict_file_path, 'wb') as file:
    pickle.dump(layers_data, file)

print(f"Data has been saved to {dict_file_path}")
