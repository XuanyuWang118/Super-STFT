from math import exp
import os
import yaml
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# 引入你的模型定义
from super_res_spec import STFTConfig, SuperResEncoder

def resample_spec_to_target_time(spec, original_hop, sr, target_times):
    """
    最标准的对齐方式：将声谱图插值到指定的目标时间点上。
    
    spec: (Freq, Time)
    original_hop: 原始 Hop 长度 (samples)
    sr: 采样率
    target_times: 目标时间点数组 (秒)
    """
    n_freq, n_time = spec.shape
    # 1. 构建原始时间轴
    # 每一帧的中心时间点或起始时间点。STFT center=True 时，第i帧对应 i * hop / sr
    original_times = np.arange(n_time) * original_hop / sr
    
    # 2. 创建插值函数
    # axis=1 表示沿时间轴插值
    # kind='nearest': 保持原始像素块状结构，不模糊 (最适合展示不同分辨率的对比)
    # kind='linear': 平滑过渡
    # fill_value="extrapolate": 防止边缘极微小的浮点误差导致 NaN
    interpolator = interp1d(original_times, spec, kind='nearest', axis=1, fill_value="extrapolate")
    
    # 3. 计算目标时间点的值
    aligned_spec = interpolator(target_times)
    
    return aligned_spec

def plot_aligned_spectrogram(ax, spec, title, sr, start_time, end_time):
    """
    绘制已经对齐好数据的声谱图
    """
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    
    # 因为数据已经严格对齐到 [start_time, end_time]，我们可以直接指定 extent
    extent = [start_time, end_time, 0, sr / 2]
    
    img = ax.imshow(spec_db, origin='lower', aspect='auto', cmap='magma', extent=extent, interpolation='none')
    
    ax.set_title(title, fontsize=10, weight='bold')
    ax.set_ylabel('Freq (Hz)')
    return img

def main():
    # ================= 1. Load Config =================
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = "exp/config.yaml"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    vis_cfg = config_dict['visualization']
    train_cfg = config_dict['train']
    stft_dict = config_dict['stft']
    
    # Paths
    exp_dir = train_cfg['exp_dir']
    model_path = os.path.join(exp_dir, "best_valid_loss.pth")
    save_dir = os.path.join(exp_dir, "image")
    os.makedirs(save_dir, exist_ok=True)
    
    test_audio_path = vis_cfg['test_audio_path']
    
    # Zoom Setting
    zoom_start = vis_cfg.get('zoom_start', 0.5)
    zoom_end = vis_cfg.get('zoom_end', 1.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 2. Load Model & Data =================
    stft_cfg = STFTConfig(cfg_dict=config_dict)
    encoder = SuperResEncoder(stft_cfg).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"[Warning] Model not found at {model_path}, utilizing random init.")
    
    encoder.eval()

    print(f"Processing {test_audio_path}...")
    audio, sr = sf.read(test_audio_path, dtype='float32')
    if audio.ndim > 1: audio = audio[:, 0]
    
    # 截取范围稍微大一点，保证插值时两头都有数据
    buffer_s = 0.5
    start_sample = max(0, int((zoom_start - buffer_s) * sr))
    end_sample = min(len(audio), int((zoom_end + buffer_s) * sr))
    
    audio_segment = audio[start_sample:end_sample]
    # 记录这段截取音频相对于原始文件的起始时间偏移
    offset_s = start_sample / sr
    
    x = torch.from_numpy(audio_segment).unsqueeze(0).unsqueeze(0).to(device)

    # ================= 3. Compute Spectrograms =================
    specs_to_plot = [] 
    
    # A. Super-Resolution
    with torch.no_grad():
        z_complex = encoder(x) 
        z_mag = torch.norm(z_complex, dim=-1).squeeze(0).cpu().numpy()
    
    super_hop = stft_cfg.base_hop_len
    specs_to_plot.append((z_mag, "Super-Res-Encoder", super_hop))
    
    # B. Target Resolutions
    resolutions = stft_dict['target_resolutions']
    
    for win_ms, hop_ms in resolutions:
        n_fft = int(sr * win_ms / 1000)
        hop_len = int(sr * hop_ms / 1000)
        
        S = librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_len, window='hann', center=True)
        S_mag = np.abs(S)
        
        title = f"STFT (Win={win_ms}ms, Hop={hop_ms}ms)"
        specs_to_plot.append((S_mag, title, hop_len))

    # ================= 4. Data Alignment Logic (The Standard Way) =================
    
    # 定义公共的绘图时间轴 (Target Time Axis)
    # 我们只关心 zoom_start 到 zoom_end 这一段
    # 分辨率设为极高，例如 1000个点，或者对应 0.5ms 一个点
    # 注意：这里的 zoom_start 是绝对时间，需要减去 offset_s 变成相对于 audio_segment 的时间
    rel_zoom_start = zoom_start - offset_s
    rel_zoom_end = zoom_end - offset_s
    
    # 检查边界安全性
    rel_zoom_start = max(0, rel_zoom_start)
    rel_zoom_end = min(len(audio_segment)/sr, rel_zoom_end)
    
    # 生成 1000 个均匀分布的时间点用于插值
    target_times = np.linspace(rel_zoom_start, rel_zoom_end, num=800)
    
    # ================= 5. Plotting =================
    num_plots = len(specs_to_plot)
    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(12, 3 * num_plots))
    gs = GridSpec(num_plots, 2, width_ratios=[1, 1]) 

    print(f"Plotting aligned range: {zoom_start:.3f}s to {zoom_end:.3f}s")

    for i, (spec, title, hop) in enumerate(specs_to_plot):
        # --- Column 0: Full View (Original Data, Linear mapping) ---
        # 全图我们不需要插值，直接画原始数据即可，用 extent 映射物理时间
        ax_full = fig.add_subplot(gs[i, 0])
        full_start = offset_s
        full_end = offset_s + spec.shape[1] * hop / sr
        
        # 简单显示截取的这一整段
        plot_aligned_spectrogram(ax_full, spec, f"{title} [Segment]", sr, full_start, full_end)
        ax_full.set_xlabel('Time (s)')
        # 画个框标出 Zoom 区域
        ax_full.axvline(x=zoom_start, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_full.axvline(x=zoom_end, color='white', linestyle='--', linewidth=1, alpha=0.7)
        
        # --- Column 1: Zoom View (Resampled & Strictly Aligned) ---
        ax_zoom = fig.add_subplot(gs[i, 1])
        
        # 核心步骤：插值对齐
        # 将当前 spec 插值到 target_times 这个统一的时间轴上
        aligned_spec_data = resample_spec_to_target_time(spec, hop, sr, target_times)
        
        # 绘图：现在所有图的 extent 都是严格的 [zoom_start, zoom_end]
        plot_aligned_spectrogram(ax_zoom, aligned_spec_data, f"{title} [Zoom]", sr, zoom_start, zoom_end)
        
        # 设置 X 轴标签格式
        ax_zoom.set_xlabel('Time (s)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, vis_cfg['save_filename'])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()