# -*- coding: utf-8 -*-
"""
波束成形生命体征检测Demo

结合了：
- demo_multi_angle.py - 多角度波束成形选择
- demo_vital_signs.py - 生命体征检测（呼吸+心跳）

功能：
1. 显示距离谱 → 用户点击选择目标距离
2. 显示角度谱 → 用户点击选择目标角度
3. 使用波束成形提取指定角度的相位信号
4. 带通滤波分离呼吸和心跳
5. FFT估计频率并可视化结果
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
from bin_processor.config import framePeriod_ms, rangeResolution_m
from msense.msense_processor import MSenseProcessor


class BeamformedVitalSigns:
    """波束成形生命体征检测器"""
    
    def __init__(self):
        print("=" * 60)
        print("波束成形生命体征检测")
        print("结合角度选择 + 呼吸心跳检测")
        print("=" * 60)
        
        # 初始化处理器
        self.processor = MSenseProcessor(num_virtual_antennas=8)
        self.reader = BinFileReader()
        
        # 参数
        self.fps = 1000 / framePeriod_ms
        self.range_resolution = rangeResolution_m
        
        print(f"\n雷达参数:")
        print(f"  帧率: {self.fps:.2f} Hz")
        print(f"  距离分辨率: {self.range_resolution*100:.2f} cm")
        print(f"  总帧数: {self.reader.n_frames}")
        print(f"  数据时长: {self.reader.n_frames / self.fps:.1f} 秒")
        
    def compute_range_profile(self):
        """计算距离谱（非相干累积多帧）"""
        print("\n[1] 计算距离谱...")
        
        frame_data = self.reader.get_frame(0)
        range_fft = np.fft.fft(frame_data, axis=2)
        num_range_bins = frame_data.shape[2] // 2
        range_fft = range_fft[:, :, :num_range_bins]
        self.range_profile = np.mean(np.abs(range_fft), axis=(0, 1))
        
        # 非相干累积更多帧
        for i in range(1, min(50, self.reader.n_frames)):
            frame = self.reader.get_frame(i)
            range_fft = np.fft.fft(frame, axis=2)[:, :, :num_range_bins]
            self.range_profile += np.mean(np.abs(range_fft), axis=(0, 1))
            
        self.num_range_bins = num_range_bins
        self.distances = np.arange(num_range_bins) * self.range_resolution
        
    def select_target_distance(self):
        """交互式选择目标距离"""
        print("\n[2] 选择目标距离...")
        
        # 转换为dB
        magnitude_db = 20 * np.log10(np.abs(self.range_profile) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.distances, magnitude_db, 'b-', linewidth=1.5)
        ax.set_xlabel('距离 (m)', fontsize=12)
        ax.set_ylabel('幅度 (dB)', fontsize=12)
        ax.set_title('点击选择目标距离 (关闭窗口确认)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(10, self.distances[-1])])
        
        selected = [None]
        marker = [None]
        
        def onclick(event):
            if event.inaxes != ax:
                return
            selected[0] = event.xdata
            if marker[0] is not None:
                marker[0].remove()
            marker[0] = ax.axvline(x=event.xdata, color='r', linestyle='--', 
                                   linewidth=2, label=f'选择: {event.xdata:.2f}m')
            ax.legend(loc='upper right')
            fig.canvas.draw()
            print(f"  已选择距离: {event.xdata:.2f} m")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        print("请在图中点击选择目标距离，然后关闭窗口...")
        plt.show()
        
        if selected[0] is None:
            # 默认使用最大峰值
            self.target_range_bin = np.argmax(self.range_profile[2:]) + 2
        else:
            self.target_range_bin = round(selected[0] / self.range_resolution)
            self.target_range_bin = max(2, min(self.target_range_bin, self.num_range_bins - 1))
        
        self.target_distance = self.target_range_bin * self.range_resolution
        print(f"  确认: Range bin {self.target_range_bin}, 距离 {self.target_distance:.2f} m")
        
    def compute_angle_spectrum(self):
        """计算指定距离的角度谱"""
        print("\n[3] 计算角度谱...")
        
        # 获取第一帧数据进行角度估计
        frame_data = self.reader.get_frame(0)
        virtual_data = self.processor.build_virtual_array(frame_data)
        range_fft_data = self.processor.range_fft(virtual_data)
        
        # 提取目标range bin的数据
        target_data = range_fft_data[:, :, self.target_range_bin]
        
        # MVDR角度估计
        self.angle_spectrum, _ = self.processor.beamformer.estimate_angle_spectrum(target_data)
        self.angles = self.processor.beamformer.angles
        
    def select_target_angle(self):
        """交互式选择目标角度"""
        print("\n[4] 选择目标角度...")
        
        spectrum_db = 10 * np.log10(np.abs(self.angle_spectrum) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.angles, spectrum_db, 'b-', linewidth=1.5)
        ax.set_xlabel('角度 (°)', fontsize=12)
        ax.set_ylabel('功率谱 (dB)', fontsize=12)
        ax.set_title(f'点击选择目标角度 - 距离 {self.target_distance:.2f}m (关闭窗口确认)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 标记阈值
        threshold = np.mean(spectrum_db)
        ax.axhline(y=threshold, color='g', linestyle='-', alpha=0.5, label=f'阈值: {threshold:.1f}dB')
        ax.legend()
        
        selected = [None]
        marker = [None]
        
        def onclick(event):
            if event.inaxes != ax:
                return
            selected[0] = event.xdata
            if marker[0] is not None:
                marker[0].remove()
            marker[0] = ax.axvline(x=event.xdata, color='r', linestyle='--', 
                                   linewidth=2, label=f'选择: {event.xdata:.1f}°')
            ax.legend(loc='upper right')
            fig.canvas.draw()
            print(f"  已选择角度: {event.xdata:.1f}°")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        print("请在图中点击选择目标角度，然后关闭窗口...")
        plt.show()
        
        if selected[0] is None:
            # 默认使用最大峰值
            self.target_angle = self.angles[np.argmax(self.angle_spectrum)]
        else:
            # 找最近的角度
            idx = np.argmin(np.abs(self.angles - selected[0]))
            self.target_angle = self.angles[idx]
        
        print(f"  确认: 目标角度 {self.target_angle:.1f}°")
        
    def extract_beamformed_phase(self):
        """使用波束成形提取指定角度的相位信号"""
        print("\n[5] 提取波束成形相位信号...")
        
        # 处理所有帧
        phase_sequences, metadata = self.processor.process_multiple_frames(
            self.reader, 
            num_frames=self.reader.n_frames,
            target_range_bin=self.target_range_bin
        )
        
        # 找到最接近的候选角度
        candidate_angles = metadata['candidate_angles']
        if len(candidate_angles) == 0:
            print("  警告: 未检测到候选角度，使用0°")
            closest_angle = 0
        else:
            closest_angle = min(candidate_angles, key=lambda x: abs(x - self.target_angle))
        
        print(f"  目标角度: {self.target_angle:.1f}° -> 最近候选: {closest_angle:.1f}°")
        
        if closest_angle in phase_sequences:
            self.beamformed_signal = phase_sequences[closest_angle]
        else:
            # 回退：使用第一个可用的角度
            first_angle = list(phase_sequences.keys())[0]
            print(f"  回退到角度: {first_angle:.1f}°")
            self.beamformed_signal = phase_sequences[first_angle]
            closest_angle = first_angle
        
        self.used_angle = closest_angle
        
        # 提取相位
        raw_phase = np.angle(self.beamformed_signal)
        self.unwrapped_phase = np.unwrap(raw_phase)
        self.unwrapped_phase = signal.detrend(self.unwrapped_phase)
        
        print(f"  相位数据点数: {len(self.unwrapped_phase)}")
        
    def bandpass_filter(self, data, low_freq, high_freq, order=4):
        """带通滤波器"""
        nyq = self.fps / 2
        low = max(0.01, min(low_freq / nyq, 0.99))
        high = max(low + 0.01, min(high_freq / nyq, 0.99))
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def detect_vital_signs(self):
        """检测呼吸和心跳"""
        print("\n[6] 检测生命体征...")
        
        # 呼吸信号: 0.1-0.6 Hz
        self.breath_signal = self.bandpass_filter(self.unwrapped_phase, 0.1, 0.6)
        
        # 心跳信号: 0.8-2.0 Hz
        self.heart_signal = self.bandpass_filter(self.unwrapped_phase, 0.8, 2.0)
        
        # FFT估计频率
        self.breath_rate, self.breath_freq, self.breath_fft = self._estimate_rate(
            self.breath_signal, 0.1, 0.6)
        self.heart_rate, self.heart_freq, self.heart_fft = self._estimate_rate(
            self.heart_signal, 0.8, 2.0)
        
        print(f"\n  ╔══════════════════════════════════════════════╗")
        print(f"  ║  目标: {self.target_distance:.2f}m @ {self.used_angle:.1f}°")
        print(f"  ║  ────────────────────────────────────────────")
        print(f"  ║  [呼吸] 频率: {self.breath_rate:5.1f} 次/分钟")
        print(f"  ║  [心跳] 频率: {self.heart_rate:5.1f} 次/分钟")
        print(f"  ╚══════════════════════════════════════════════╝")
        
    def _estimate_rate(self, data, min_freq, max_freq):
        """使用FFT估计频率"""
        n = len(data)
        freq = np.fft.fftfreq(n, 1/self.fps)
        fft = np.abs(np.fft.fft(data))
        
        pos_mask = (freq >= min_freq) & (freq <= max_freq)
        if np.sum(pos_mask) == 0:
            return 0, freq, fft
            
        freq_range = freq[pos_mask]
        fft_range = fft[pos_mask]
        peak_idx = np.argmax(fft_range)
        rate_bpm = freq_range[peak_idx] * 60
        
        return rate_bpm, freq, fft
    
    def plot_results(self):
        """可视化结果"""
        print("\n[7] 生成可视化结果...")
        
        time_axis = np.arange(len(self.unwrapped_phase)) / self.fps
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 距离谱
        ax1 = fig.add_subplot(3, 2, 1)
        magnitude_db = 20 * np.log10(np.abs(self.range_profile) + 1e-10)
        ax1.plot(self.distances, magnitude_db, 'b-', linewidth=1)
        ax1.axvline(x=self.target_distance, color='r', linestyle='--', 
                   linewidth=2, label=f'目标: {self.target_distance:.2f}m')
        ax1.set_xlabel('距离 (m)')
        ax1.set_ylabel('幅度 (dB)')
        ax1.set_title('距离谱')
        ax1.set_xlim([0, min(10, self.distances[-1])])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 角度谱
        ax2 = fig.add_subplot(3, 2, 2)
        spectrum_db = 10 * np.log10(np.abs(self.angle_spectrum) + 1e-10)
        ax2.plot(self.angles, spectrum_db, 'b-', linewidth=1)
        ax2.axvline(x=self.used_angle, color='r', linestyle='--', 
                   linewidth=2, label=f'目标: {self.used_angle:.1f}°')
        ax2.set_xlabel('角度 (°)')
        ax2.set_ylabel('功率谱 (dB)')
        ax2.set_title(f'角度谱 @ {self.target_distance:.2f}m')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 原始相位信号
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(time_axis, self.unwrapped_phase, 'b-', linewidth=0.5)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('相位 (rad)')
        ax3.set_title(f'波束成形相位信号 ({self.target_distance:.2f}m @ {self.used_angle:.1f}°)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 呼吸信号
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(time_axis, self.breath_signal, 'g-', linewidth=1)
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('幅度')
        ax4.set_title(f'[呼吸信号] (0.1-0.6Hz) - {self.breath_rate:.1f} 次/分钟')
        ax4.grid(True, alpha=0.3)
        
        # 5. 心跳信号
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(time_axis, self.heart_signal, 'r-', linewidth=1)
        ax5.set_xlabel('时间 (s)')
        ax5.set_ylabel('幅度')
        ax5.set_title(f'[心跳信号] (0.8-2.0Hz) - {self.heart_rate:.1f} 次/分钟')
        ax5.grid(True, alpha=0.3)
        
        # 6. 频谱对比
        ax6 = fig.add_subplot(3, 2, 6)
        pos_mask = self.breath_freq >= 0
        ax6.plot(self.breath_freq[pos_mask] * 60, np.abs(self.breath_fft[pos_mask]), 
                'g-', label='呼吸', alpha=0.7)
        ax6.plot(self.heart_freq[pos_mask] * 60, np.abs(self.heart_fft[pos_mask]), 
                'r-', label='心跳', alpha=0.7)
        ax6.axvline(x=self.breath_rate, color='g', linestyle='--', alpha=0.5)
        ax6.axvline(x=self.heart_rate, color='r', linestyle='--', alpha=0.5)
        ax6.set_xlabel('频率 (次/分钟)')
        ax6.set_ylabel('幅度')
        ax6.set_title('生命体征频谱')
        ax6.set_xlim([0, 150])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(PROJECT_ROOT, 'output', 'beamformed_vital_signs.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n结果已保存到: {output_path}")
        
        plt.show()
        
    def run(self):
        """运行完整流程"""
        self.compute_range_profile()
        self.select_target_distance()
        self.compute_angle_spectrum()
        self.select_target_angle()
        self.extract_beamformed_phase()
        self.detect_vital_signs()
        self.plot_results()
        print("\n检测完成!")


def main():
    detector = BeamformedVitalSigns()
    detector.run()


if __name__ == "__main__":
    main()
