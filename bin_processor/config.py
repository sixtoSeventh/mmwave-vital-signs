# -*- coding: utf-8 -*-
"""
雷达参数配置文件
根据 mmWaveStudio 的 18xx.mmwave.json 配置提取
更新时间: 2026-01-17
"""

# ============ 雷达配置参数 ============

# ADC 采样点数
numADCSamples = 256

# 发射天线数 (chirpEndIdx=1, 使用TX0和TX1)
numTxAntennas = 2

# 接收天线数 (rxChannelEn = 0xF)
numRxAntennas = 4

# 每帧循环数 (numLoops) - 生命体征检测: 2
numLoopsPerFrame = 2

# 距离 bin 数
numRangeBins = int(numADCSamples / 2)

# 总帧数 - 生命体征检测: 1200 (60秒 @ 20Hz)
numFrames = 1200

# ============ 文件路径配置 ============
BIN_FILE_PATH = r"E:\TI\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin"

# ============ 雷达物理参数 ============
# 起始频率 (GHz)
startFreq_GHz = 77.0

# 频率斜率 (MHz/us) - 生命体征检测: 70.006
freqSlope_MHz_us = 70.006

# ADC 采样率 (ksps) - digOutSampleRate: 10000
adcSampleRate_ksps = 10000

# 空闲时间 (us)
idleTime_us = 100.0

# 斜坡结束时间 (us) - 生命体征检测: 55.0
rampEndTime_us = 55.0

# 帧周期 (ms) - 生命体征检测: 50.0 (20Hz帧率)
framePeriod_ms = 50.0

# ============ 计算派生参数 ============
SPEED_OF_LIGHT = 3e8

# ADC采样时间 (us)
adcSamplingTime_us = numADCSamples / (adcSampleRate_ksps / 1e3)  # = 256 / 10 = 25.6 us

# 有效带宽 (MHz) - 使用实际ADC采样窗口对应的带宽
# 这与MATLAB rawDataReader的计算方式一致
bandwidth_MHz = freqSlope_MHz_us * adcSamplingTime_us  # = 70.006 * 25.6 = 1792.15 MHz

# 距离分辨率 (m) - 约 8.4 cm
rangeResolution_m = SPEED_OF_LIGHT / (2 * bandwidth_MHz * 1e6)

# 最大距离 (m) - 约 2.1 m (使用FFT的一半)
maxRange_m = (numADCSamples / 2) * rangeResolution_m

# Chirp 周期
chirpTime_us = idleTime_us + rampEndTime_us

# 最大速度 (m/s) - 约 1 m/s
lambda_m = SPEED_OF_LIGHT / (startFreq_GHz * 1e9)
maxVelocity_mps = lambda_m / (4 * chirpTime_us * 1e-6 * numTxAntennas)

print(f"=== 雷达配置信息 (pymmw 兼容) ===")
print(f"ADC 采样点数: {numADCSamples}")
print(f"TX/RX 天线: {numTxAntennas}/{numRxAntennas}")
print(f"每帧循环数: {numLoopsPerFrame}")
print(f"距离分辨率: {rangeResolution_m*100:.2f} cm")
print(f"最大距离: {maxRange_m:.2f} m")
print(f"最大速度: {maxVelocity_mps:.2f} m/s")
