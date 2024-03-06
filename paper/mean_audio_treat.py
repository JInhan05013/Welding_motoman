import numpy as np
import matplotlib.pyplot as plt
import wave

def moving_average(signal, window_size):
    """对信号应用移动平均滤波器"""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, 'same')

for n in range(2,24):
 
    data_dir=f'../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41/layer_{n}/'
    # Microphone data
    wavfile = wave.open(data_dir + 'mic_recording.wav', 'rb')
    samplerate = 44000  # Fixed sample rate
    audio_data = np.frombuffer(wavfile.readframes(wavfile.getnframes()), dtype=np.int16)
    wavfile.close()  # It's a good practice to close the file

    # 应用移动平均滤波器
    window_size = 441  # 示例窗口大小，可以根据需要调整
    smoothed_audio_data = moving_average(audio_data, window_size)

    # 绘制原始和平滑后的信号波形
    plt.figure(figsize=(15, 6))
    plt.plot(audio_data, label='Original Audio Signal')
    plt.plot(smoothed_audio_data, label='Smoothed Audio Signal', linewidth=2)
    plt.title(f'Audio Signal Waveform of layer{n}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()