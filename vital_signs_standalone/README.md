# mmWave Vital Signs Detection - Standalone Version

毫米波雷达生命体征检测系统 - 独立版本

## 功能

- 🫁 呼吸检测 (6-30 BPM)
- 💓 心跳检测 (60-120 BPM)
- 📊 可视化结果

## 硬件

- TI IWR1843BOOST + DCA1000
- mmWave Studio

## 安装

```bash
pip install numpy scipy matplotlib
```

## 使用

```bash
# 使用默认bin文件路径
python vital_signs_detection.py

# 指定bin文件
python vital_signs_detection.py --bin_file "your_data.bin"
```

## 配置

编辑 `vital_signs_detection.py` 中的 `CONFIG` 字典修改雷达参数：

```python
CONFIG = {
    'numADCSamples': 256,
    'numTxAntennas': 2,
    'numRxAntennas': 4,
    'numLoopsPerFrame': 2,
    'freqSlope_MHz_us': 70.006,
    'adcSampleRate_ksps': 10000,
    'framePeriod_ms': 50.0,
    'bin_file_path': "your_default_path.bin",
}
```

## License

MIT License
