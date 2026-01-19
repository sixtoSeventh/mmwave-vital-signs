# -*- coding: utf-8 -*-
"""
毫米波雷达生命体征检测系统 - 独立版本
Standalone Vital Signs Detection using mmWave Radar

基于 TI IWR1843BOOST + DCA1000
功能: 呼吸检测 (6-30 BPM) + 心跳检测 (60-120 BPM)

使用方法:
    python vital_signs_detection.py
    或
    python vital_signs_detection.py --bin_file "your_data.bin"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============ 雷达配置参数 ============
# 根据 mmWaveStudio 配置修改这些参数

CONFIG = {
    # ADC 采样点数
    'numADCSamples': 256,
    
    # 发射天线数 (TX0 和 TX1)
    'numTxAntennas': 2,
    
    # 接收天线数 (4个RX)
    'numRxAntennas': 4,
    
    # 每帧循环数 (numLoops)
    'numLoopsPerFrame': 2,
    
    # 频率斜率 (MHz/us)
    'freqSlope_MHz_us': 70.006,
    
    # ADC 采样率 (ksps)
    'adcSampleRate_ksps': 10000,
    
    # 帧周期 (ms) - 决定帧率
    'framePeriod_ms': 50.0,
    
    # 默认bin文件路径
    'bin_file_path': r"E:\TI\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin",
}

# 计算派生参数
SPEED_OF_LIGHT = 3e8
adcSamplingTime_us = CONFIG['numADCSamples'] / (CONFIG['adcSampleRate_ksps'] / 1e3)
bandwidth_MHz = CONFIG['freqSlope_MHz_us'] * adcSamplingTime_us
CONFIG['rangeResolution_m'] = SPEED_OF_LIGHT / (2 * bandwidth_MHz * 1e6)
CONFIG['frameRate_Hz'] = 1000 / CONFIG['framePeriod_ms']


class BinFileReader:
    """读取 TI mmWave 雷达的 bin 文件"""
    
    def __init__(self, filename):
        self.filename = filename
        self.numADCSamples = CONFIG['numADCSamples']
        self.numTxAntennas = CONFIG['numTxAntennas']
        self.numRxAntennas = CONFIG['numRxAntennas']
        self.numLoopsPerFrame = CONFIG['numLoopsPerFrame']
        
        # 每帧的采样点数 (复数数据，所以乘以2)
        self.samples_per_frame = (self.numRxAntennas * self.numLoopsPerFrame * 
                                   self.numTxAntennas * self.numADCSamples * 2)
        
        self._load_data()
        self.current_frame = 0
        
    def _load_data(self):
        """加载 bin 文件数据"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"找不到文件: {self.filename}")
        
        self.raw_data = np.fromfile(self.filename, dtype=np.int16)
        self.n_frames = int(len(self.raw_data) / self.samples_per_frame)
        
        print(f"=== bin 文件信息 ===")
        print(f"文件: {os.path.basename(self.filename)}")
        print(f"大小: {len(self.raw_data) * 2 / 1024 / 1024:.2f} MB")
        print(f"总帧数: {self.n_frames}")
        print(f"时长: {self.n_frames / CONFIG['frameRate_Hz']:.1f} 秒")
        
    def get_frame(self, frame_idx):
        """获取指定帧的数据"""
        if frame_idx >= self.n_frames:
            raise IndexError(f"帧索引 {frame_idx} 超出范围")
        
        start_idx = frame_idx * self.samples_per_frame
        end_idx = start_idx + self.samples_per_frame
        frame_data = self.raw_data[start_idx:end_idx]
        
        # 重组为复数格式
        complex_data = frame_data[0::2] + 1j * frame_data[1::2]
        
        # 重塑为数据立方体 (numRx, numChirps, numADCSamples)
        data_cube = complex_data.reshape(
            (self.numLoopsPerFrame * self.numTxAntennas, 
             self.numRxAntennas, 
             self.numADCSamples)
        )
        data_cube = np.transpose(data_cube, (1, 0, 2))
        return data_cube


class VitalSignsDetector:
    """生命体征检测器"""
    
    def __init__(self, bin_file):
        print("=" * 60)
        print("   毫米波雷达生命体征检测系统")
        print("   Respiration: 0.1-0.5 Hz | Heart: 1.0-2.0 Hz")
        print("=" * 60)
        
        self.bin_file = bin_file
        self.range_resolution = CONFIG['rangeResolution_m']
        self.frame_rate = CONFIG['frameRate_Hz']
        self.num_range_bins = CONFIG['numADCSamples'] // 2
        
        print(f"\n雷达参数:")
        print(f"  距离分辨率: {self.range_resolution*100:.2f} cm")
        print(f"  帧率: {self.frame_rate:.1f} Hz")
        
    def load_data(self):
        """加载雷达数据"""
        print("\n[1] 加载雷达数据...")
        self.reader = BinFileReader(self.bin_file)
        self.num_frames = self.reader.n_frames
        
    def range_fft(self):
        """距离维FFT"""
        print("\n[2] 距离维FFT...")
        
        # 累积多帧获取稳定的距离谱
        range_profile_acc = np.zeros(self.num_range_bins)
        for i in range(min(50, self.num_frames)):
            frame = self.reader.get_frame(i)
            range_fft = np.fft.fft(frame, axis=2)[:, :, :self.num_range_bins]
            range_profile_acc += np.mean(np.abs(range_fft), axis=(0, 1))
        
        self.range_profile = range_profile_acc
        
        # 保存每帧FFT数据用于相位提取
        self.fft_data = np.zeros((self.num_frames, CONFIG['numADCSamples']), dtype=np.complex128)
        for i in range(self.num_frames):
            frame = self.reader.get_frame(i)
            self.fft_data[i, :] = np.fft.fft(frame[0, 0, :])
        
    def find_target_range_bin(self):
        """交互式选择目标距离"""
        print("\n[3] 选择目标位置...")
        
        distances = np.arange(self.num_range_bins) * self.range_resolution
        energy_db = 20*np.log10(self.range_profile + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(distances, energy_db, 'b-', linewidth=1.5)
        ax.set_xlabel('距离 (m)', fontsize=12)
        ax.set_ylabel('能量 (dB)', fontsize=12)
        ax.set_title('点击选择目标距离 (关闭窗口确认)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(10, distances[-1])])
        
        selected_distance = [None]
        selected_line = [None]
        
        def onclick(event):
            if event.inaxes != ax:
                return
            selected_distance[0] = event.xdata
            if selected_line[0] is not None:
                selected_line[0].remove()
            selected_line[0] = ax.axvline(x=event.xdata, color='r', linestyle='--', 
                                          linewidth=2, label=f'选择: {event.xdata:.2f}m')
            ax.legend(loc='upper right')
            fig.canvas.draw()
            print(f"  已选择距离: {event.xdata:.2f} m")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        print("请在图中点击选择目标位置...")
        plt.show()
        
        if selected_distance[0] is None:
            self.target_bin = np.argmax(self.range_profile[2:]) + 2
        else:
            self.target_bin = round(selected_distance[0] / self.range_resolution)
            self.target_bin = max(2, min(self.target_bin, self.num_range_bins - 1))
        
        self.target_distance = self.target_bin * self.range_resolution
        print(f"  目标距离: {self.target_distance:.2f} m")
        
    def extract_phase(self):
        """提取相位信号"""
        print("\n[4] 提取相位信号...")
        target_signal = self.fft_data[:, self.target_bin]
        raw_phase = np.angle(target_signal)
        self.unwrapped_phase = np.unwrap(raw_phase)
        self.unwrapped_phase = signal.detrend(self.unwrapped_phase)
        
    def bandpass_filter(self, data, low_freq, high_freq, order=4):
        """带通滤波器"""
        nyq = self.frame_rate / 2
        low = max(0.01, min(low_freq / nyq, 0.99))
        high = max(low + 0.01, min(high_freq / nyq, 0.99))
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def extract_vital_signs(self):
        """分离呼吸和心跳信号"""
        print("\n[5] 分离呼吸和心跳信号...")
        
        # 呼吸信号: 0.1-0.5 Hz
        self.breath_signal = self.bandpass_filter(self.unwrapped_phase, 0.1, 0.5)
        
        # 心跳信号: 1.0-2.0 Hz (先去除呼吸分量)
        phase_no_breath = self.unwrapped_phase - self.breath_signal
        self.heart_signal = self.bandpass_filter(phase_no_breath, 1.0, 2.0)
        
    def estimate_rate(self, data, min_freq, max_freq):
        """FFT频率估计"""
        n = len(data)
        freq = np.fft.fftfreq(n, 1/self.frame_rate)
        fft = np.abs(np.fft.fft(data))
        pos_mask = (freq >= min_freq) & (freq <= max_freq)
        if np.sum(pos_mask) == 0:
            return 0, freq, fft
        freq_range = freq[pos_mask]
        fft_range = fft[pos_mask]
        peak_freq = freq_range[np.argmax(fft_range)]
        return peak_freq * 60, freq, fft
    
    def calculate_vital_rates(self):
        """计算生命体征频率"""
        print("\n[6] 计算生命体征频率...")
        
        self.breath_rate, self.breath_freq, self.breath_fft = \
            self.estimate_rate(self.breath_signal, 0.1, 0.5)
        self.heart_rate, self.heart_freq, self.heart_fft = \
            self.estimate_rate(self.heart_signal, 1.0, 2.0)
        
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  [呼吸] 频率: {self.breath_rate:5.1f} 次/分钟          ║")
        print(f"  ║  [心跳] 频率: {self.heart_rate:5.1f} 次/分钟          ║")
        print(f"  ╚══════════════════════════════════════╝")
        
    def plot_results(self):
        """绘制结果"""
        print("\n[7] 生成可视化结果...")
        
        time_axis = np.arange(self.num_frames) / self.frame_rate
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # 原始相位
        axes[0, 0].plot(time_axis, self.unwrapped_phase, 'b-', linewidth=0.5)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('相位 (rad)')
        axes[0, 0].set_title(f'原始相位信号 (距离: {self.target_distance:.2f}m)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 呼吸时域
        axes[1, 0].plot(time_axis, self.breath_signal, 'g-', linewidth=1)
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('幅度')
        axes[1, 0].set_title(f'[呼吸信号] - {self.breath_rate:.1f} BPM')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 呼吸频谱
        pos_mask = self.breath_freq >= 0
        axes[1, 1].plot(self.breath_freq[pos_mask] * 60, np.abs(self.breath_fft[pos_mask]), 'g-')
        axes[1, 1].axvline(x=self.breath_rate, color='r', linestyle='--', label=f'{self.breath_rate:.1f} BPM')
        axes[1, 1].set_xlabel('频率 (次/分钟)')
        axes[1, 1].set_ylabel('幅度')
        axes[1, 1].set_title('呼吸频谱')
        axes[1, 1].set_xlim([0, 40])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 心跳时域
        axes[2, 0].plot(time_axis, self.heart_signal, 'r-', linewidth=1)
        axes[2, 0].set_xlabel('时间 (s)')
        axes[2, 0].set_ylabel('幅度')
        axes[2, 0].set_title(f'[心跳信号] - {self.heart_rate:.1f} BPM')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 心跳频谱
        axes[2, 1].plot(self.heart_freq[pos_mask] * 60, np.abs(self.heart_fft[pos_mask]), 'r-')
        axes[2, 1].axvline(x=self.heart_rate, color='b', linestyle='--', label=f'{self.heart_rate:.1f} BPM')
        axes[2, 1].set_xlabel('频率 (次/分钟)')
        axes[2, 1].set_ylabel('幅度')
        axes[2, 1].set_title('心跳频谱')
        axes[2, 1].set_xlim([40, 140])
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 结果摘要
        axes[0, 1].axis('off')
        info_text = f"""
================================
   Vital Signs Detection Result
================================

  Distance: {self.target_distance:.2f} m
  Respiration: {self.breath_rate:.1f} BPM
  Heart Rate: {self.heart_rate:.1f} BPM
  Duration: {self.num_frames / self.frame_rate:.1f} s
  Sample Rate: {self.frame_rate:.1f} Hz
================================
        """
        axes[0, 1].text(0.1, 0.5, info_text, transform=axes[0, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       fontfamily='DejaVu Sans Mono',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_file = 'vital_signs_result.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n结果已保存到: {output_file}")
        
    def run(self):
        """运行完整流程"""
        self.load_data()
        self.range_fft()
        self.find_target_range_bin()
        self.extract_phase()
        self.extract_vital_signs()
        self.calculate_vital_rates()
        self.plot_results()
        print("\n检测完成!")


def main():
    parser = argparse.ArgumentParser(description='毫米波雷达生命体征检测')
    parser.add_argument('--bin_file', type=str, default=CONFIG['bin_file_path'],
                        help='bin文件路径')
    args = parser.parse_args()
    
    detector = VitalSignsDetector(args.bin_file)
    detector.run()


if __name__ == "__main__":
    main()
