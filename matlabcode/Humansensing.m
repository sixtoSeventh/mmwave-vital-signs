%==========================================================================
% TI 毫米波雷达数据处理示例：心跳和呼吸检测 (基于相位分析)
%
% 输入: 从 mmWaveStudio 的 rawDataReader.m 脚本生成的 radarCube.mat 文件
% 输出: 各种相位波形图、位移波形图、心跳和呼吸频率频谱图
%
% 关键修改点:
% 1. 加载 radarCube 数据，不再需要原始 ADC 数据 'rawData'。
% 2. 移除原始代码中的 '距离维FFT' 部分，直接使用 radarCube (已是 1D FFT 结果)。
% 3. 所有雷达参数 (Nadc, Ts, K, rangeRes等) 从 radarCube.rfParams 中动态提取或推算。
% 4. 避免使用 .moe 文件加载滤波器系数，改为在代码中直接设计巴特沃斯滤波器。
% 5. 目标距离通过用户输入而非鼠标点击来指定。
%==========================================================================

clear;
close all;
clc;

%% 1. 加载 radarcube 数据
% 假设你的 radarCube 数据被保存在 'radarCube.mat' 文件中
% 使用脚本所在位置自动定位数据文件夹
scriptPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptPath, '..', 'data', 'HeartBeat_Breathe1.mat');
load(dataPath, 'radarCube');

% 检查数据是否为空
if isempty(radarCube.data)
    error('雷达立方体数据为空，请检查 radarCube.data 文件内容。');
end

% 从 radarCube 结构中获取必要的参数
rfParams = radarCube.rfParams; % 射频参数
numFrames = length(radarCube.data); % 总帧数
% radarCube.data 的维度是 {帧数} x [numChirpsPerFrame, numRxChan, numRangeBins]
NchirpPerFrame = size(radarCube.data{1}, 1); % 每个帧的Chirp数 (原代码中的 Nchirp 在这里含义模糊，现在统一为总帧数)
numRxChan = size(radarCube.data{1}, 2);
numRangeBins = size(radarCube.data{1}, 3);

% 从 rfParams 提取雷达配置参数 (用于计算波长, 帧率等)
startFreq_GHz = rfParams.startFreq; % 起始频率 (GHz)
freqSlope_MHz_usec = rfParams.freqSlope; % 频率斜率 (MHz/us)
sampleRate_Msps = rfParams.sampleRate; % ADC采样率 (Msps)
rangeRes = rfParams.rangeResolutionsInMeters; % 距离分辨率 (m)
framePeriodicity_msec = rfParams.framePeriodicity; % 帧周期 (ms)
% 这里的 nFFTtime 应该等于 numRangeBins，因为 radarCube 已经是 1D FFT 结果
nFFTtime = numRangeBins;

% 计算 derived 雷达参数
C = 3e8; % 光速 (m/s)
lambda = C / (startFreq_GHz * 1e9); % 波长 (m)
FS = 1000 / framePeriodicity_msec; % 帧率 (Hz) - 这就是慢时间采样率 FS

fprintf('已加载雷达立方体数据。\n');
fprintf('  总帧数: %d\n', numFrames);
fprintf('  每帧Chirp数: %d\n', NchirpPerFrame);
fprintf('  RX 通道数: %d\n', numRxChan);
fprintf('  距离单元数 (1D FFT点数): %d\n', numRangeBins);
fprintf('\n雷达系统参数：\n');
fprintf('  起始频率: %.2f GHz\n', startFreq_GHz);
fprintf('  波长: %.4f m\n', lambda);
fprintf('  帧率 (慢时间采样率 FS): %.2f Hz\n', FS);
fprintf('  距离分辨率: %.4f m\n', rangeRes);


% 为了进行后续处理，将所有帧的 radarCube 数据堆叠起来
% 这里假设我们只关心第一个 RX 通道和第一个 Chirp 进行相位提取
% 更复杂的场景可能需要对多个 RX 通道进行相干叠加或选择不同的 Chirp
rxChannelToAnalyze = 1; % 默认使用第一个 RX 通道
chirpToAnalyze = 1;     % 默认使用第一个 Chirp

% 将所有帧的 radarCube 数据在选定的 Chirp 和 RX 通道上提取并堆叠
% 形成一个 [numFrames, numRangeBins] 的矩阵，作为新的 'rangeData'
% 这相当于原代码中经过 1D FFT 后的 rangeData，但现在是跨所有帧的
rangeData = zeros(numFrames, numRangeBins, 'single');
for fIdx = 1:numFrames
    rangeData(fIdx, :) = squeeze(radarCube.data{fIdx}(chirpToAnalyze, rxChannelToAnalyze, :));
end

% 原代码中的 Nchirp 在这里被替换为 numFrames，表示慢时间维度上的采样点数
Nchirp = numFrames;
% 原代码中的 Nadc 不再直接使用，因为我们直接从 radarCube 开始。
% 但其值可以从 rfParams.numAdcSamples 得到，这里不需要。


%% 1. 绘制 1D FFT 结果 (等同于原代码的 '距离维FFT' 作图，但数据来源不同)
% 这里我们使用第一帧的 rangeData 来作图
current_range_data_for_plot = rangeData(1, :); % 提取第一帧的距离剖面

rangeAxis = rangeRes * (0:nFFTtime-1); % 距离轴的计算，简化了 Nadc/nFFTtime 因子，因为 rangeRes 已是最终分辨率
[X_mesh, Y_mesh] = meshgrid(rangeAxis, (1:Nchirp)); % Nchirp 此时是 numFrames

figure('Name', '距离维-1DFFT结果');
mesh(X_mesh, Y_mesh, abs(rangeData)); % 绘制所有帧的距离剖面
xlabel('距离(m)');
ylabel('帧数（慢时间）'); % 将“脉冲chirp数”改为“帧数”更准确
zlabel('幅度');
title('距离维-1DFFT结果 (所有帧)');
drawnow;


%% 2. 提取相位
angleData = angle(rangeData);

% detaR 的计算与 rangeRes 相同，因为 rangeRes 已经是每个距离单元的物理间隔
detaR = rangeRes;

% Range-bin tracking: 找出能量最大的点，即人体的位置
rangeAbs = abs(rangeData);

% 对所有帧的距离剖面幅度进行平均，以找到稳定的目标
rangeSum = mean(rangeAbs, 1); % 对第一个维度（帧）求平均

% 限定检测距离
minRange = 0.3; % 默认值，可根据需要调整
maxRange = 1.5; % 默认值，可根据需要调整

% 将物理距离门限转换为距离单元索引
minIndex = floor(minRange / detaR) + 1; % +1 因为 MATLAB 索引从 1 开始
maxIndex = ceil(maxRange / detaR) + 1;

% 确保索引在有效范围内
minIndex = max(1, minIndex);
maxIndex = min(numRangeBins, maxIndex);

% 排除门限外的距离单元
if minIndex <= numRangeBins
    rangeSum(1:minIndex-1) = 0;
end
if maxIndex >= 1
    rangeSum(maxIndex+1:end) = 0;
end


% 用户输入目标距离 (代替原代码的 max 搜索，更精确)
figure('Name', '1D FFT 距离剖面 - 请参考此图输入目标距离');
plot(rangeAxis, 10*log10(rangeSum)); % 绘制平均距离剖面供用户参考
grid on;
xlabel('距离 (m)');
ylabel('平均幅度 (dB)');
title('平均 1D FFT 距离剖面 (所有帧平均)');
drawnow; % 确保图窗在等待用户输入前显示

% 提示用户输入目标距离
fprintf('\n请在命令窗口输入离雷达最近的强反射目标的距离 (单位: 米)，然后按 Enter 键。\n');
fprintf('  当前有效检测距离范围: %.2f m 到 %.2f m\n', rangeAxis(minIndex), rangeAxis(maxIndex));

% 使用 input 函数获取用户输入的距离
targetRange_m = input('  请输入目标距离 (m): ');

% 验证用户输入是否在有效范围内
if targetRange_m < rangeAxis(minIndex) || targetRange_m > rangeAxis(maxIndex)
    error('输入的距离 (%.2f m) 超出有效检测距离范围 (%.2f m 到 %.2f m)。', ...
        targetRange_m, rangeAxis(minIndex), rangeAxis(maxIndex));
end

% 根据输入的距离找到最接近的距离单元索引
[~, index] = min(abs(rangeAxis - targetRange_m));

fprintf('  用户输入的距离: %.2f m\n', targetRange_m);
fprintf('  选定的目标距离单元索引: %d (对应实际距离: %.2f m)\n', ...
    index, rangeAxis(index));
close; % 关闭辅助图窗


%% 3. 取出能量最大点的相位  extract phase from selected range bin
angleTarget = angleData(:,index); % angleData 的维度是 [numFrames, numRangeBins]
% 提取相位信号（原始）
figure('Name', '未展开相位信号');
plot((0:Nchirp-1)/FS, angleTarget); % Nchirp 此时是 numFrames
xlabel('时间 (秒)'); % 横轴是时间
ylabel('相位');
title('未展开相位信号');
drawnow;

phi = angleTarget;


%% 4. 进行相位解缠
% unwrap函数：
phi = unwrap(phi);
figure('Name', '解缠后的相位');
plot((0:Nchirp-1)/FS, phi); % Nchirp 此时是 numFrames
xlabel('时间 (秒)'); % 横轴是时间
ylabel('相位（rad）');
title('解缠后的相位');
drawnow;
angle_fft_last = phi;


%% 5. phase difference 相位差分
% 通过减去连续的相位值，对展开的相位执行相位差运算，
angle_fft_last2 = zeros(Nchirp, 1); % 维度与 Nchirp (numFrames) 匹配
for i = 2:Nchirp
    angle_fft_last2(i) = angle_fft_last(i) - angle_fft_last(i-1);
end
% 第一个点的相位差通常设为0或与第二个点相同，这里保持0
angle_fft_last2(1) = 0;

figure('Name', '相位差分后信号');
plot((0:Nchirp-1)/FS, angle_fft_last2); % Nchirp 此时是 numFrames
xlabel('时间 (秒)'); % 横轴是时间
ylabel('相位（rad）');
title('相位差分后信号');
drawnow;

phi = angle_fft_last2;


%% 6. 脉冲噪声去除：滑动平均滤波
%   去除由于测试环境引起的脉冲噪声
%   窗口长度为5，对应 5 * (1/FS) = 5 * (1/20) = 0.25 秒
moving_average_window = 5;
phi = smoothdata(phi, 'movmean', moving_average_window);
figure('Name', '滑动平均滤波相位信号');
plot((0:Nchirp-1)/FS, phi); % Nchirp 此时是 numFrames
xlabel('时间 (秒)'); % 横轴是时间
ylabel('幅度');
title('滑动平均滤波后的相位差分信号');
drawnow;


%% 7. 对相位差分信号作FFT
N_fft = length(phi); % FFT 的样本点数，与 phi 的长度相同

FFT_result = abs(fft(phi, N_fft)); % 进行 FFT 并取模
f_axis = (0:N_fft-1)*(FS/N_fft); % 频率轴

figure('Name', '相位信号FFT');
% 通常只关心正频率部分，且频率上限限制在奈奎斯特频率 FS/2
plot(f_axis(1:N_fft/2+1), 10*log10(FFT_result(1:N_fft/2+1))); % 转换为 dB
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
title('相位差分信号FFT');
xlim([0 FS/2]); % 频率范围从 0 到奈奎斯特频率
grid on;
drawnow;


%% 8. IIR带通滤波 (0.1-0.5Hz)，输出呼吸信号
% 设计 IIR 4阶巴特沃斯带通滤波器

lowCutoff_breath_Hz = 0.1;
highCutoff_breath_Hz = 0.5;
order_breath = 4; % 滤波器阶数

% 检查采样率是否足够高以支持滤波
if FS / 2 < highCutoff_breath_Hz
    error('帧率 (%.2f Hz) 过低，无法有效分离呼吸信号 (奈奎斯特频率 %.2f Hz < 呼吸上限 %.2f Hz)。', FS, FS/2, highCutoff_breath_Hz);
end

[b_breath, a_breath] = butter(order_breath, ...
    [lowCutoff_breath_Hz, highCutoff_breath_Hz] / (FS / 2), 'bandpass');
% 使用 filtfilt 避免相位失真
breath_data = filtfilt(b_breath, a_breath, phi);

figure('Name', '呼吸时域波形');
plot((0:Nchirp-1)/FS, breath_data); % Nchirp 此时是 numFrames
xlabel('时间 (秒)');
ylabel('幅度');
title(sprintf('呼吸时域波形 (%.1f-%.1f Hz)', lowCutoff_breath_Hz, highCutoff_breath_Hz));
grid on;
drawnow;

%% 9. 呼吸信号FFT-Peak
breath_fft = abs(fft(breath_data, N_fft));
P1_breath = breath_fft(1:N_fft/2+1);
P1_breath(2:end-1) = 2*P1_breath(2:end-1); % 单边谱

figure('Name', '呼吸信号FFT');
plot(f_axis(1:N_fft/2+1), 10*log10(P1_breath)); % 显示 dB 幅度
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
title('呼吸信号FFT');
xlim([0 FS/2]); % 频率范围从 0 到奈奎斯特频率
grid on;
drawnow;

% 在指定的呼吸频率范围内找出主导峰值
freq_range_breath_indices = find(f_axis(1:N_fft/2+1) >= lowCutoff_breath_Hz & f_axis(1:N_fft/2+1) <= highCutoff_breath_Hz);
if isempty(freq_range_breath_indices)
    warning('在指定的呼吸频率范围 (%.1f-%.1f Hz) 内未找到有效频率点。', lowCutoff_breath_Hz, highCutoff_breath_Hz);
    breath_freq_Hz = NaN;
    breath_count_BPM = NaN;
else
    [~, breath_peak_idx_in_range] = max(P1_breath(freq_range_breath_indices));
    breath_index = freq_range_breath_indices(breath_peak_idx_in_range); % 原始 FFT 索引
    
    breath_freq_Hz = f_axis(breath_index);
    breath_count_BPM = breath_freq_Hz * 60;        % 呼吸频率解算 (BPM)
end

fprintf('\n检测到的呼吸频率: %.2f Hz (%.1f BPM)\n', breath_freq_Hz, breath_count_BPM);


%% 10. IIR带通滤波 (0.8-2Hz)，输出心跳数据
% 设计 IIR 8阶巴特沃斯带通滤波器

lowCutoff_heart_Hz = 0.8;
highCutoff_heart_Hz = 2.0; % 根据经验值或您的配置
order_heart = 8; % 滤波器阶数

% 检查采样率是否足够高以支持滤波
if FS / 2 < highCutoff_heart_Hz
    error('帧率 (%.2f Hz) 过低，无法有效分离心跳信号 (奈奎斯特频率 %.2f Hz < 心跳上限 %.2f Hz)。', FS, FS/2, highCutoff_heart_Hz);
end

[b_heart, a_heart] = butter(order_heart, ...
    [lowCutoff_heart_Hz, highCutoff_heart_Hz] / (FS / 2), 'bandpass');
% 使用 filtfilt 避免相位失真
heart_data = filtfilt(b_heart, a_heart, phi);

figure('Name', '心跳时域波形');
plot((0:Nchirp-1)/FS, heart_data); % Nchirp 此时是 numFrames
xlabel('时间 (秒)');
ylabel('幅度');
title(sprintf('心跳时域波形 (%.1f-%.1f Hz)', lowCutoff_heart_Hz, highCutoff_heart_Hz));
grid on;
drawnow;

%% 11. 心跳信号FFT-Peak
heart_fft = abs(fft(heart_data, N_fft));
P1_heart = heart_fft(1:N_fft/2+1);
P1_heart(2:end-1) = 2*P1_heart(2:end-1); % 单边谱

figure('Name', '心跳信号FFT');
plot(f_axis(1:N_fft/2+1), 10*log10(P1_heart)); % 显示 dB 幅度
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
title('心跳信号FFT');
xlim([0 FS/2]); % 频率范围从 0 到奈奎斯特频率
grid on;
drawnow;

% 在指定的心跳频率范围内找出主导峰值
freq_range_heart_indices = find(f_axis(1:N_fft/2+1) >= lowCutoff_heart_Hz & f_axis(1:N_fft/2+1) <= highCutoff_heart_Hz);
if isempty(freq_range_heart_indices)
    warning('在指定的心跳频率范围 (%.1f-%.1f Hz) 内未找到有效频率点。', lowCutoff_heart_Hz, highCutoff_heart_Hz);
    heart_freq_Hz = NaN;
    heart_count_BPM = NaN;
else
    [~, heart_peak_idx_in_range] = max(P1_heart(freq_range_heart_indices));
    heart_index = freq_range_heart_indices(heart_peak_idx_in_range); % 原始 FFT 索引
    
    heart_freq_Hz = f_axis(heart_index);
    heart_count_BPM = heart_freq_Hz * 60; % 心跳频率解算 (BPM)
end

fprintf('\n检测到的心跳频率: %.2f Hz (%.1f BPM)\n', heart_freq_Hz, heart_count_BPM);