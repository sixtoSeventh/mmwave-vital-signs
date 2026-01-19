# mmWave Vital Signs Detection

基于TI IWR1843BOOST + DCA1000的毫米波雷达生命体征检测系统

## 功能特性

- 🫁 **呼吸检测**: 0.1-0.5 Hz (6-30 BPM)
- 💓 **心跳检测**: 1.0-2.0 Hz (60-120 BPM)
- 📊 **可视化**: 实时相位信号、时域波形、频谱分析
- 🎯 **交互式目标选择**: 点击距离谱选择检测目标

## 硬件要求

- TI IWR1843BOOST 毫米波雷达模块
- TI DCA1000 数据采集板
- mmWave Studio 软件

## 安装

```bash
# 克隆仓库
git clone https://github.com/sixtoSeventh/mmwave-vital-signs.git
cd mmwave-vital-signs

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 配置雷达参数

编辑 `bin_processor/config.py`，设置与mmWave Studio一致的参数：

```python
numADCSamples = 256
numLoopsPerFrame = 2
freqSlope_MHz_us = 70.006
framePeriod_ms = 50.0  # 20Hz帧率
```

### 2. 采集数据

使用mmWave Studio采集数据，保存为 `adc_data.bin`

推荐配置：
- 帧周期: 50ms (20Hz)
- 帧数: 1200+ (60秒以上)
- ADC采样点: 256

### 3. 运行检测

```bash
python demo/demo_vital_signs.py
```

程序会显示距离谱图，点击目标位置后自动检测呼吸和心跳。

## 项目结构

```
mmwave-vital-signs/
├── bin_processor/          # 数据处理模块
│   ├── config.py           # 雷达参数配置
│   ├── bin_reader.py       # bin文件读取器
│   └── pymmw_style_processor.py
├── demo/                   # 演示脚本
│   ├── demo_vital_signs.py # 生命体征检测
│   ├── demo_beamformed_vital_signs.py  # 波束成形版本
│   └── demo_multi_angle.py # 多角度检测
├── matlabcode/             # MATLAB参考代码
│   ├── Humansensing.m
│   └── rawDataReader.m
├── config/                 # mmWave Studio配置文件
└── output/                 # 输出结果
```

## 算法原理

```
ADC数据 → Range-FFT → 目标距离检测 → 相位提取 → 相位解缠
                                          ↓
心跳频率 ← FFT频谱分析 ← 带通滤波(1.0-2.0Hz) ← 去除呼吸分量
呼吸频率 ← FFT频谱分析 ← 带通滤波(0.1-0.5Hz)
```

## 参考资源

- [PKU Millimeter Wave Radar Tutorial](https://github.com/DeepWiSe888/PKU-Millimeter-Wave-Radar-Tutorial)
- [TI mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK)

## License

MIT License
