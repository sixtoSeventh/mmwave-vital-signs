# -*- coding: utf-8 -*-
"""
生命体征检测Demo - Python版本

参考: PKU-Millimeter-Wave-Radar-Tutorial 生命体征检测demo
基于IWR1843BOOST + DCA1000

功能:
1. 手动选择人体位置（点击距离谱）
2. 提取相位信号并解缠
3. 带通滤波分离呼吸和心跳信号
4. FFT谱估计计算频率
"""

import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from bin_processor.bin_reader import BinFileReader
from bin_processor.config import (
    numADCSamples, numRxAntennas, numTxAntennas, numLoopsPerFrame,
    freqSlope_MHz_us, adcSampleRate_ksps, framePeriod_ms, rangeResolution_m
)


class VitalSignsDetector:
    """生命体征检测器"""
    
    def __init__(self):
        print("=" * 60)
        print("生命体征检测 (呼吸 + 心跳)")
        print("参考: PKU-Millimeter-Wave-Radar-Tutorial")
        print("=" * 60)
        
        # 雷达参数 (从config.py统一导入)
        self.Fs = adcSampleRate_ksps * 1e3  # ADC采样率
        self.range_resolution = rangeResolution_m  # 距离分辨率 (从config导入)
        
        # 帧率 (信号采样率)
        self.frame_rate = 1000 / framePeriod_ms  # Hz
        
        # Range FFT点数 (使用原始ADC采样数，不做零填充)
        self.num_range_bins = numADCSamples // 2  # 只取正频率部分
        
        print(f"\n雷达参数:")
        print(f"  ADC采样率: {self.Fs/1e6:.2f} MHz")
        print(f"  距离分辨率: {self.range_resolution*100:.2f} cm")
        print(f"  帧率(生命体征采样率): {self.frame_rate:.2f} Hz")
        
    def load_data(self):
        """加载并处理雷达数据"""
        print("\n[1] 加载雷达数据...")
        
        self.reader = BinFileReader()
        print(f"  总帧数: {self.reader.n_frames}")
        print(f"  数据时长: {self.reader.n_frames / self.frame_rate:.1f} 秒")
        
        self.num_frames = self.reader.n_frames
        print(f"  处理帧数: {self.num_frames}")
        
    def range_fft(self):
        """距离维FFT - 对所有天线和chirps取平均"""
        print("\n[2] 距离维FFT...")
        
        # 非相干累积多帧以获取稳定的距离谱
        range_profile_acc = np.zeros(self.num_range_bins)
        
        for i in range(min(50, self.num_frames)):  # 累积前50帧
            frame = self.reader.get_frame(i)  # (numRx, numChirps, numADCSamples)
            # 对所有天线和chirps做FFT
            range_fft = np.fft.fft(frame, axis=2)
            # 只取正频率部分
            range_fft = range_fft[:, :, :self.num_range_bins]
            # 对所有天线和chirps取平均
            range_profile_acc += np.mean(np.abs(range_fft), axis=(0, 1))
        
        self.range_profile = range_profile_acc
        
        # 保存每帧第一个天线第一个chirp的FFT数据（用于后续相位提取）
        self.fft_data = np.zeros((self.num_frames, numADCSamples), dtype=np.complex128)
        for i in range(self.num_frames):
            frame = self.reader.get_frame(i)
            # 取第一个RX天线，第一个chirp做FFT
            self.fft_data[i, :] = np.fft.fft(frame[0, 0, :])
        
        print(f"  FFT点数: {numADCSamples}")
        
    def find_target_range_bin(self):
        """交互式选择目标距离（点击距离谱选择）"""
        print("\n[3] 手动选择目标位置...")
        
        # 有效距离范围
        start_bin = 2  # 从约0.4m开始
        end_bin = self.num_range_bins
        
        # 绘制距离谱
        distances = np.arange(end_bin) * self.range_resolution
        energy_db = 20*np.log10(self.range_profile + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(distances, energy_db, 'b-', linewidth=1.5)
        ax.set_xlabel('距离 (m)', fontsize=12)
        ax.set_ylabel('能量 (dB)', fontsize=12)
        ax.set_title('点击选择目标距离 (关闭窗口确认)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(10, distances[-1])])
        
        # 初始化选择
        selected_distance = [None]
        selected_line = [None]
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            selected_distance[0] = event.xdata
            
            # 更新红色竖线
            if selected_line[0] is not None:
                selected_line[0].remove()
            selected_line[0] = ax.axvline(x=event.xdata, color='r', linestyle='--', 
                                          linewidth=2, label=f'选择: {event.xdata:.2f}m')
            ax.legend(loc='upper right')
            fig.canvas.draw()
            print(f"  已选择距离: {event.xdata:.2f} m")
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        print("请在图中点击选择目标位置，然后关闭窗口确认...")
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        
        # 处理选择结果
        if selected_distance[0] is None:
            # 如果未选择，使用能量最大的bin
            print("  未选择距离，使用自动检测")
            self.target_bin = np.argmax(self.range_profile[start_bin:end_bin]) + start_bin
        else:
            self.target_bin = round(selected_distance[0] / self.range_resolution)
            self.target_bin = max(start_bin, min(self.target_bin, end_bin - 1))
        
        self.target_distance = self.target_bin * self.range_resolution
        print(f"  目标Range bin: {self.target_bin}")
        print(f"  目标距离: {self.target_distance:.2f} m")
        
    def extract_phase(self):
        """提取目标距离处的相位"""
        print("\n[4] 提取相位信号...")
        
        # 取目标距离bin的复数信号
        target_signal = self.fft_data[:, self.target_bin]
        
        # 计算相位
        raw_phase = np.angle(target_signal)
        
        # 相位解缠
        self.unwrapped_phase = np.unwrap(raw_phase)
        
        # 去除线性趋势（DC漂移）
        self.unwrapped_phase = signal.detrend(self.unwrapped_phase)
        
        print(f"  相位数据点数: {len(self.unwrapped_phase)}")
        
    def bandpass_filter(self, data, low_freq, high_freq, order=4):
        """带通滤波器"""
        nyq = self.frame_rate / 2
        low = low_freq / nyq
        high = high_freq / nyq
        
        # 确保频率在有效范围内
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered
    
    def extract_vital_signs(self):
        """提取呼吸和心跳信号"""
        print("\n[5] 分离呼吸和心跳信号...")
        
        # 呼吸信号: 0.1-0.5 Hz (6-30次/分钟)
        self.breath_signal = self.bandpass_filter(self.unwrapped_phase, 0.1, 0.5)
        
        # 心跳信号: 1.0-2.0 Hz (60-120次/分钟)
        # 提高下限到1.0Hz避免呼吸谐波干扰 (17BPM×3=51BPM=0.85Hz)
        # 先从相位中减去呼吸分量
        phase_no_breath = self.unwrapped_phase - self.breath_signal
        self.heart_signal = self.bandpass_filter(phase_no_breath, 1.0, 2.0)
        
        print(f"  呼吸滤波: 0.1-0.5 Hz (6-30 BPM)")
        print(f"  心跳滤波: 1.0-2.0 Hz (60-120 BPM，已去除呼吸分量)")
        
    def estimate_rate(self, data, min_freq, max_freq):
        """使用FFT估计频率"""
        n = len(data)
        freq = np.fft.fftfreq(n, 1/self.frame_rate)
        fft = np.abs(np.fft.fft(data))
        
        # 只取正频率
        pos_mask = (freq >= min_freq) & (freq <= max_freq)
        
        if np.sum(pos_mask) == 0:
            return 0
            
        freq_range = freq[pos_mask]
        fft_range = fft[pos_mask]
        
        # 找峰值
        peak_idx = np.argmax(fft_range)
        peak_freq = freq_range[peak_idx]
        
        # 转换为次/分钟
        rate_bpm = peak_freq * 60
        
        return rate_bpm, freq, fft
    
    def calculate_vital_rates(self):
        """计算呼吸和心跳频率"""
        print("\n[6] 计算生命体征频率...")
        
        # 呼吸频率
        self.breath_rate, self.breath_freq, self.breath_fft = \
            self.estimate_rate(self.breath_signal, 0.1, 0.5)
        
        # 心跳频率 (1.0-2.0 Hz = 60-120 BPM)
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
        
        # 1. 原始相位信号
        axes[0, 0].plot(time_axis, self.unwrapped_phase, 'b-', linewidth=0.5)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('相位 (rad)')
        axes[0, 0].set_title(f'原始相位信号 (距离: {self.target_distance:.2f}m)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 呼吸信号时域
        axes[1, 0].plot(time_axis, self.breath_signal, 'g-', linewidth=1)
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('幅度')
        axes[1, 0].set_title(f'[呼吸信号] (0.1-0.6Hz) - {self.breath_rate:.1f} 次/分钟')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. 呼吸信号频域
        pos_mask = self.breath_freq >= 0
        axes[1, 1].plot(self.breath_freq[pos_mask] * 60, 
                       np.abs(self.breath_fft[pos_mask]), 'g-')
        axes[1, 1].axvline(x=self.breath_rate, color='r', linestyle='--', 
                          label=f'{self.breath_rate:.1f} 次/分')
        axes[1, 1].set_xlabel('频率 (次/分钟)')
        axes[1, 1].set_ylabel('幅度')
        axes[1, 1].set_title('呼吸频谱')
        axes[1, 1].set_xlim([0, 40])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 4. 心跳信号时域
        axes[2, 0].plot(time_axis, self.heart_signal, 'r-', linewidth=1)
        axes[2, 0].set_xlabel('时间 (s)')
        axes[2, 0].set_ylabel('幅度')
        axes[2, 0].set_title(f'[心跳信号] (0.8-2.0Hz) - {self.heart_rate:.1f} 次/分钟')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 5. 心跳信号频域
        axes[2, 1].plot(self.heart_freq[pos_mask] * 60, 
                       np.abs(self.heart_fft[pos_mask]), 'r-')
        axes[2, 1].axvline(x=self.heart_rate, color='b', linestyle='--',
                          label=f'{self.heart_rate:.1f} 次/分')
        axes[2, 1].set_xlabel('频率 (次/分钟)')
        axes[2, 1].set_ylabel('幅度')
        axes[2, 1].set_title('心跳频谱')
        axes[2, 1].set_xlim([40, 140])
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 6. 综合信息
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
        plt.savefig(os.path.join(PROJECT_ROOT, 'output', 'vital_signs_result.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n结果已保存到: output/vital_signs_result.png")
        
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
    detector = VitalSignsDetector()
    detector.run()


if __name__ == "__main__":
    main()
