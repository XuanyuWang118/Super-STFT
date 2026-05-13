import os
import re
import yaml
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from super_res_spec import STFTConfig, SuperResEncoder, HybridSupervisionLoss

def resample_spec_to_target_time(spec, original_hop, sr, target_times):
    n_freq, n_time = spec.shape
    original_times = np.arange(n_time) * original_hop / sr
    interpolator = interp1d(original_times, spec, kind='nearest', axis=1, fill_value="extrapolate")
    aligned_spec = interpolator(target_times)
    return aligned_spec

def plot_aligned_spectrogram(ax, spec, title, sr, start_time, end_time):
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    extent = [start_time, end_time, 0, sr / 2]
    img = ax.imshow(spec_db, origin='lower', aspect='auto', cmap='magma', extent=extent, interpolation='none')
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_ylabel('Freq (Hz)')
    return img

def load_best_or_latest_model(encoder, exp_dir, device):
    """尝试加载 best_valid_loss.pth，否则加载最新的 *epoch.pth"""
    best_path = os.path.join(exp_dir, "best_valid_loss.pth")
    if os.path.exists(best_path):
        print(f"Loaded BEST model from {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        return True

    print(f"Best model not found. Searching for latest epoch in {exp_dir}...")
    ckpt_list = [f for f in os.listdir(exp_dir) if f.endswith('epoch.pth')]
    if not ckpt_list:
        print("[Warning] No model found, utilizing random init.")
        return False
        
    ckpt_list.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
    latest_path = os.path.join(exp_dir, ckpt_list[-1])
    print(f"Loaded LATEST model from {latest_path}")
    ckpt = torch.load(latest_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    return True

def compute_and_cache_features(audio_segment, sr, stft_cfg, device, cache_path):
    """一次性计算目标、原始STFT和投影，并保存为文件"""
    print(f"--- Computing and caching features to {cache_path} ---")

    super_hop = stft_cfg.base_hop_len
    x = torch.from_numpy(audio_segment).unsqueeze(0).unsqueeze(0).to(device)

    cache_dict = {
        'stft': [],     # List of (mag_np, win_ms, hop_ms, hop_len)
        'proj': [],     # List of (mag_proj_np, win_ms, hop_ms, super_hop)
        'target': None  # (target_mag_np, super_hop)
    }

    # 原始 STFT（native hop，用于绘图 Section C）
    for win_ms, hop_ms in stft_cfg.target_resolutions:
        n_fft_curr = int(sr * win_ms / 1000)
        target_hop_len = int(sr * hop_ms / 1000)
        S = librosa.stft(audio_segment, n_fft=n_fft_curr, hop_length=target_hop_len, window='hann', center=True)
        cache_dict['stft'].append((np.abs(S), win_ms, hop_ms, target_hop_len))

    # 交集目标和归一化投影：直接复用模型的 compute_intersection，保证逻辑完全一致
    loss_fn = HybridSupervisionLoss(stft_cfg).to(device)
    with torch.no_grad():
        intersection_mag, mag_stack_normed, _, __ = loss_fn.compute_intersection(x, verbose=False)

    cache_dict['target'] = (intersection_mag.squeeze(0).cpu().numpy(), super_hop)

    for (win_ms, hop_ms), mag_normed in zip(stft_cfg.target_resolutions, mag_stack_normed):
        cache_dict['proj'].append((mag_normed.squeeze(0).cpu().numpy(), win_ms, hop_ms, super_hop))

    torch.save(cache_dict, cache_path)
    print(f"Caching completed successfully!")
    return cache_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='plot', choices=['plot', 'cache_only'], 
                        help='plot: normal plotting, cache_only: just compute and save features without plotting.')
    args = parser.parse_args()

    # ================= 1. Load Config =================
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = "exp/config.yaml"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    vis_cfg = config_dict.get('visualization', {})
    train_cfg = config_dict.get('train', {})
    
    exp_dir = train_cfg.get('exp_dir', 'exp')
    save_dir = os.path.join(exp_dir, "image")
    os.makedirs(save_dir, exist_ok=True)
    
    test_audio_path = vis_cfg.get('test_audio_path', '')
    zoom_start = vis_cfg.get('zoom_start', 0.5)
    zoom_end = vis_cfg.get('zoom_end', 0.8)
    plot_items = vis_cfg.get('plot_items', ['pred', 'target', 'stft', 'proj']) # 从配置读取需要画的项
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= 2. Prepare Data =================
    print(f"Processing {test_audio_path}...")
    audio, sr = sf.read(test_audio_path, dtype='float32')
    if audio.ndim > 1: audio = audio[:, 0]
    
    buffer_s = 0.5
    start_sample = max(0, int((zoom_start - buffer_s) * sr))
    end_sample = min(len(audio), int((zoom_end + buffer_s) * sr))
    
    audio_segment = audio[start_sample:end_sample]
    offset_s = start_sample / sr
    x = torch.from_numpy(audio_segment).unsqueeze(0).unsqueeze(0).to(device)

    # ================= 3. Handle Cache for Fixed Features =================
    stft_cfg = STFTConfig(cfg_dict=config_dict)
    audio_name = os.path.splitext(os.path.basename(test_audio_path))[0]
    cache_path = os.path.join(exp_dir, f"{audio_name}_stft_cache.pt")

    if not os.path.exists(cache_path):
        cache_dict = compute_and_cache_features(audio_segment, sr, stft_cfg, device, cache_path)
    else:
        print(f"Loading cached features from {cache_path}...")
        cache_dict = torch.load(cache_path, weights_only=False)

    if args.mode == 'cache_only':
        return # 如果只要求缓存，到此结束

    # ================= 4. Prepare Plotting Items =================
    specs_to_plot = [] 
    
    # A. Super-Resolution Prediction
    if 'pred' in plot_items:
        encoder = SuperResEncoder(stft_cfg).to(device)
        load_best_or_latest_model(encoder, exp_dir, device)
        encoder.eval()
        with torch.no_grad():
            z_complex = encoder(x) 
            z_mag = torch.norm(z_complex, dim=-1).squeeze(0).cpu().numpy()
        specs_to_plot.append((z_mag, "Super-Res-Encoder [Prediction]", stft_cfg.base_hop_len))

    # B. Target
    if 'target' in plot_items:
        mag, hop = cache_dict['target']
        specs_to_plot.append((mag, "Ultimate Target [Min Intersection]", hop))

    # C. Original STFTs
    if 'stft' in plot_items:
        for mag, win_ms, hop_ms, hop in cache_dict['stft']:
            specs_to_plot.append((mag, f"STFT (Win={win_ms}ms, Hop={hop_ms}ms)", hop))

    # D. Projected STFTs
    if 'proj' in plot_items:
        for mag, win_ms, hop_ms, hop in cache_dict['proj']:
            specs_to_plot.append((mag, f"Proj STFT (Win={win_ms}ms, Hop={hop_ms}ms)", hop))

    # ================= 5. Data Alignment & Plotting =================
    if not specs_to_plot:
        print("Nothing to plot based on 'plot_items' config.")
        return

    rel_zoom_start = max(0, zoom_start - offset_s)
    rel_zoom_end = min(len(audio_segment)/sr, zoom_end - offset_s)
    target_times = np.linspace(rel_zoom_start, rel_zoom_end, num=800)
    
    num_plots = len(specs_to_plot)
    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(14, 3 * num_plots))
    gs = GridSpec(num_plots, 2, width_ratios=[1, 1]) 

    print(f"Plotting aligned range: {zoom_start:.3f}s to {zoom_end:.3f}s")

    for i, (spec, title, hop) in enumerate(specs_to_plot):
        # Column 0: Full View
        ax_full = fig.add_subplot(gs[i, 0])
        full_start = offset_s
        full_end = offset_s + spec.shape[1] * hop / sr
        plot_aligned_spectrogram(ax_full, spec, f"{title} [Segment]", sr, full_start, full_end)
        ax_full.set_xlabel('Time (s)')
        ax_full.axvline(x=zoom_start, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_full.axvline(x=zoom_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Column 1: Zoom View
        ax_zoom = fig.add_subplot(gs[i, 1])
        aligned_spec_data = resample_spec_to_target_time(spec, hop, sr, target_times)
        plot_aligned_spectrogram(ax_zoom, aligned_spec_data, f"{title} [Zoom]", sr, zoom_start, zoom_end)
        ax_zoom.set_xlabel('Time (s)')

    plt.tight_layout()
    save_filename = vis_cfg.get('save_filename', f"{audio_name}_spec.png")
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()