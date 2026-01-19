# -*- coding: utf-8 -*-
"""
多距离多角度信号波形对比演示 (动态模式)

支持：
1. 动态添加距离 - 选择一个距离立即处理
2. 每个距离可选择多个角度
3. 可随时添加新的距离
"""

import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from bin_processor.bin_reader import BinFileReader
from bin_processor.config import framePeriod_ms, numADCSamples
from msense.msense_processor import MSenseProcessor


class DynamicMultiViewer:
    """动态多距离多角度波形查看器"""
    
    def __init__(self):
        print("=" * 60)
        print("多距离多角度信号波形对比 (动态模式)")
        print("=" * 60)
        
        # 初始化处理器
        print("\n[1] 初始化...")
        self.processor = MSenseProcessor(num_virtual_antennas=8)
        self.reader = BinFileReader()
        self.fps = 1000 / framePeriod_ms
        
        print(f"帧率: {self.fps:.2f} fps")
        print(f"总帧数: {self.reader.n_frames}")
        print(f"距离分辨率: {self.processor.range_resolution*100:.2f} cm")
        
        # 存储: {distance_bin: {angle: signal, ...}}
        self.all_signals = {}
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.color_idx = 0
        
        # 波形数据
        self.wave_lines = []
        self.selected_items = []  # [(range_bin, angle, color), ...]
        
    def compute_range_profile(self):
        """计算距离谱"""
        print("\n[2] 计算距离谱...")
        
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
        self.distances = np.arange(num_range_bins) * self.processor.range_resolution
        
    def process_distance(self, range_bin):
        """处理指定距离"""
        if range_bin in self.all_signals:
            return self.all_signals[range_bin]
            
        distance = range_bin * self.processor.range_resolution
        print(f"  处理距离 {distance:.2f}m (bin {range_bin})...")
        
        phase_sequences, metadata = self.processor.process_multiple_frames(
            self.reader, 
            num_frames=self.reader.n_frames,
            target_range_bin=range_bin
        )
        
        self.all_signals[range_bin] = {
            'distance': distance,
            'phase_sequences': phase_sequences,
            'metadata': metadata,
            'angle_spectrum': metadata['angle_spectrum'],
            'candidate_angles': metadata['candidate_angles'],
            'color': self.colors[self.color_idx % len(self.colors)]
        }
        self.color_idx += 1
        
        print(f"  完成! 候选角度: {len(metadata['candidate_angles'])}个")
        return self.all_signals[range_bin]
        
    def run_interactive_viewer(self):
        """运行交互式查看器"""
        print("\n[3] 启动交互式查看器")
        print("    左上: 距离谱 - 点击选择距离")
        print("    右上: 角度谱 - 点击添加波形")  
        print("    下方: 波形对比")
        
        # 创建图形布局
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1, 1.5], 
                     width_ratios=[1, 1], hspace=0.3, wspace=0.25)
        
        # 左上: 距离谱
        self.ax_range = self.fig.add_subplot(gs[0, 0])
        magnitude_db = 20 * np.log10(np.abs(self.range_profile) + 1e-10)
        self.ax_range.plot(self.distances, magnitude_db, 'b-', linewidth=1.5)
        self.ax_range.set_xlabel('距离 (m)')
        self.ax_range.set_ylabel('幅度 (dB)')
        self.ax_range.set_title('点击选择距离 (可多次点击)')
        self.ax_range.grid(True, alpha=0.3)
        self.ax_range.set_xlim([0, min(10, self.distances[-1])])
        self.range_markers = []
        
        # 右上: 角度谱 (初始为空)
        self.ax_angle = self.fig.add_subplot(gs[0, 1])
        self.ax_angle.set_xlabel('角度 (°)')
        self.ax_angle.set_ylabel('功率谱 (dB)')
        self.ax_angle.set_title('选择距离后显示角度谱')
        self.ax_angle.grid(True, alpha=0.3)
        self.current_range_bin = None
        
        # 下方: 波形显示
        self.ax_wave = self.fig.add_subplot(gs[1, :])
        self.ax_wave.set_xlabel('时间 (s)')
        self.ax_wave.set_ylabel('位移 (mm)')
        self.ax_wave.set_title('信号波形对比 (点击角度谱添加)')
        self.ax_wave.grid(True, alpha=0.3)
        self.ax_wave.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 绑定点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
        
        # 打印结果
        if self.selected_items:
            print("\n已选择的信号:")
            for rb, angle, color in self.selected_items:
                dist = self.all_signals[rb]['distance']
                print(f"  - {dist:.2f}m @ {angle:.1f}°")
        
    def on_click(self, event):
        """处理点击事件"""
        if event.inaxes == self.ax_range:
            self.on_range_click(event)
        elif event.inaxes == self.ax_angle:
            self.on_angle_click(event)
            
    def on_range_click(self, event):
        """距离谱点击"""
        distance = event.xdata
        if distance is None:
            return
            
        range_bin = round(distance / self.processor.range_resolution)
        range_bin = max(1, min(range_bin, self.num_range_bins - 1))
        actual_distance = range_bin * self.processor.range_resolution
        
        # 先在距离谱上标记（立即显示反馈）
        color = self.colors[self.color_idx % len(self.colors)]
        marker = self.ax_range.axvline(x=actual_distance, color=color, 
                                        linestyle='--', linewidth=2)
        self.range_markers.append(marker)
        
        # 显示处理提示
        self.ax_angle.clear()
        self.ax_angle.text(0.5, 0.5, f'正在处理 {actual_distance:.2f}m...', 
                          transform=self.ax_angle.transAxes,
                          ha='center', va='center', fontsize=14)
        self.ax_angle.set_title('处理中...')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # 强制刷新显示
        
        # 处理该距离
        data = self.process_distance(range_bin)
        self.current_range_bin = range_bin
        
        # 更新角度谱
        self.update_angle_spectrum(range_bin)
        self.fig.canvas.draw()
        
    def update_angle_spectrum(self, range_bin):
        """更新角度谱显示"""
        self.ax_angle.clear()
        
        data = self.all_signals[range_bin]
        angles = self.processor.beamformer.angles
        spectrum = data['angle_spectrum']
        spectrum_db = 10 * np.log10(np.abs(spectrum) + 1e-10)
        
        self.ax_angle.plot(angles, spectrum_db, 'b-', linewidth=1.5)
        threshold = np.mean(spectrum_db)
        self.ax_angle.axhline(y=threshold, color='g', linestyle='-', alpha=0.5)
        self.ax_angle.fill_between(angles, threshold, spectrum_db, 
                                   where=(spectrum_db > threshold), 
                                   color='orange', alpha=0.3)
        
        # 标记已选择的角度
        for rb, angle, color in self.selected_items:
            if rb == range_bin:
                idx = np.argmin(np.abs(angles - angle))
                self.ax_angle.plot(angle, spectrum_db[idx], 'o', 
                                  color=color, markersize=10)
        
        distance = data['distance']
        self.ax_angle.set_xlabel('角度 (°)')
        self.ax_angle.set_ylabel('功率谱 (dB)')
        self.ax_angle.set_title(f"角度谱 - {distance:.2f}m (点击添加波形)")
        self.ax_angle.grid(True, alpha=0.3)
        
    def on_angle_click(self, event):
        """角度谱点击"""
        if self.current_range_bin is None:
            print("  请先在距离谱中选择一个距离")
            return
            
        clicked_angle = event.xdata
        if clicked_angle is None:
            return
            
        range_bin = self.current_range_bin
        data = self.all_signals[range_bin]
        
        # 找最近的候选角度
        closest_angle = min(data['candidate_angles'], 
                           key=lambda x: abs(x - clicked_angle))
        
        # 检查是否已添加
        for rb, angle, _ in self.selected_items:
            if rb == range_bin and abs(angle - closest_angle) < 1:
                print(f"  {data['distance']:.2f}m @ {closest_angle:.1f}° 已存在")
                return
        
        if closest_angle not in data['phase_sequences']:
            print(f"  角度 {closest_angle:.1f}° 无数据")
            return
        
        # 添加波形
        color = data['color']
        self.selected_items.append((range_bin, closest_angle, color))
        
        # 提取信号并转换为位移
        signal = data['phase_sequences'][closest_angle]
        phase = np.unwrap(np.angle(signal))
        wavelength = 3.9e-3
        displacement = phase / (4 * np.pi / wavelength) * 1000
        displacement = displacement - np.mean(displacement)
        
        time_axis = np.arange(len(displacement)) / self.fps
        
        # 使用不同线型区分同一距离的不同角度
        linestyles = ['-', '--', '-.', ':']
        same_dist_count = sum(1 for rb, _, _ in self.selected_items[:-1] if rb == range_bin)
        linestyle = linestyles[same_dist_count % len(linestyles)]
        
        distance = data['distance']
        line, = self.ax_wave.plot(time_axis, displacement, linestyle, 
                                  color=color, linewidth=1.5, 
                                  label=f'{distance:.2f}m @ {closest_angle:.1f}°')
        self.wave_lines.append(line)
        
        # 更新
        self.ax_wave.legend(loc='upper right', ncol=min(4, len(self.wave_lines)), fontsize=8)
        self.ax_wave.set_title(f'已选择 {len(self.selected_items)} 个信号')
        self.ax_wave.relim()
        self.ax_wave.autoscale_view()
        
        self.update_angle_spectrum(range_bin)
        self.fig.canvas.draw()
        
        print(f"  已添加: {distance:.2f}m @ {closest_angle:.1f}°")


def main():
    viewer = DynamicMultiViewer()
    viewer.compute_range_profile()
    viewer.run_interactive_viewer()
    print("\n演示完成!")


if __name__ == "__main__":
    main()
