import os
from sympy import use
import yaml
import csv
import torch
import soundfile as sf
import numpy as np
import librosa
import onnxruntime as ort
from tqdm import tqdm
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from pesq import pesq
from pystoi import stoi

from espnet2.enh.layers.dnsmos import DNSMOS_local
from super_res_spec import STFTConfig, SuperResEncoder, SuperResDecoder

# ==========================================
# DNSMOS Evaluator (Local ONNX)
# ==========================================
class DNSMOSEvaluator:
    def __init__(self, model_dir, use_gpu=False):
        """
        model_dir: 包含 model_v8.onnx 和 sig_bak_ovr.onnx 的目录
        """
        primary_model = os.path.join(model_dir, 'sig_bak_ovr.onnx')
        p808_model = os.path.join(model_dir, 'model_v8.onnx')
        
        if not os.path.exists(primary_model) or not os.path.exists(p808_model):
            print(f"[Error] DNSMOS models not found in {model_dir}")
            self.dnsmos = None
            return

        # 实例化 ESPnet 提供的 DNSMOS_local
        # 注意：这里我们使用 onnx 模式 (convert_to_torch=False)
        self.dnsmos = DNSMOS_local(
            primary_model_path=primary_model,
            p808_model_path=p808_model,
            use_gpu=use_gpu,
            convert_to_torch=False
        )
        self.sr = 16000

    def compute(self, audio_array):
        """
        audio_array: (T,) numpy float32
        """
        if self.dnsmos is None:
            return {'SIG': 0, 'BAK': 0, 'OVRL': 0, 'P808_MOS': 0}

        # DNSMOS_local.__call__ 接收音频和采样率
        # 它内部会自动处理分段、补齐以及平均值计算
        scores = self.dnsmos(audio_array, self.sr)
        
        # 返回格式统一化
        return {
            'SIG': scores['SIG'],
            'BAK': scores['BAK'],
            'OVRL': scores['OVRL'],
            'P808_MOS': scores['P808_MOS']
        }


# ==========================================
# Metric Helper
# ==========================================
def compute_metrics(ref_wav, est_wav, sr, dnsmos_evaluator=None):
    # 1. SI-SNR (保持原有逻辑)
    ref_t = torch.from_numpy(ref_wav).unsqueeze(0)
    est_t = torch.from_numpy(est_wav).unsqueeze(0)
    sisnr_val = si_snr(est_t, ref_t).item()

    # 2. PESQ (Wideband)
    try:
        pesq_val = pesq(sr, ref_wav, est_wav, 'wb')
    except:
        pesq_val = 0.0

    # 3. STOI
    stoi_val = stoi(ref_wav, est_wav, sr, extended=False)

    # 4. DNSMOS (调用新封装)
    m = {'SIG': 0.0, 'BAK': 0.0, 'OVRL': 0.0, 'P808_MOS': 0.0}
    if dnsmos_evaluator:
        m = dnsmos_evaluator.compute(est_wav)

    return {
        "SI-SNR": sisnr_val,
        "PESQ": pesq_val,
        "STOI": stoi_val,
        "SIG": m['SIG'],
        "BAK": m['BAK'],
        "OVRL": m['OVRL'],
        "P808_MOS": m['P808_MOS']
    }

# ==========================================
# Main
# ==========================================
def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = "exp/config.yaml"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Config Objects
    inf_cfg = config_dict['inference']
    data_cfg = config_dict['dataset']
    train_cfg = config_dict['train']
    
    # Paths
    exp_dir = train_cfg['exp_dir']
    model_path = os.path.join(exp_dir, "best_valid_loss.pth")
    output_csv_path = os.path.join(exp_dir, inf_cfg['output_filename'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on {device}")
    
    # 1. Load Model
    stft_cfg = STFTConfig(cfg_dict=config_dict)
    encoder = SuperResEncoder(stft_cfg).to(device)
    decoder = SuperResDecoder(stft_cfg).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
    else:
        print(f"[Error] Model not found at {model_path}")
        return

    encoder.eval()
    decoder.eval()

    # 2. Init DNSMOS
    use_gpu = torch.cuda.is_available()
    dnsmos_eval = DNSMOSEvaluator(inf_cfg['dnsmos_dir'], use_gpu=use_gpu)

    # 3. Load Test Data
    with open(data_cfg['test_scp'], 'r') as f:
        lines = f.readlines()
    # Parse SCP: "id /path/to/wav"
    file_list = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            file_list.append((parts[0], parts[1]))
            
    print(f"Found {len(file_list)} test files.")

    # 4. Inference Loop
    results = [] # List of dicts
    
    # 准备 CSV Header
    headers = ["Filename", "SI-SNR", "PESQ", "STOI", "SIG", "BAK", "OVRL", "P808_MOS"]
    
    # 打开 CSV 文件准备写入 (实时写入防止中断丢失)
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        with torch.no_grad():
            for filename, wav_path in tqdm(file_list):
                # Read Audio
                if not os.path.exists(wav_path):
                    wav_path = os.path.join("/exp/xuanyu.wang/espnet_20250624/egs2/vctk_noisy/enh1", wav_path)
                audio, sr = sf.read(wav_path, dtype='float32')
                if audio.ndim > 1: audio = audio[:, 0]
                
                x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward
                z_complex = encoder(x)
                x_recon = decoder(z_complex)
                
                # Post-process
                est_wav = x_recon.squeeze().cpu().numpy()
                
                # Align lengths
                min_len = min(len(audio), len(est_wav))
                ref_wav = audio[:min_len]
                est_wav_trunc = est_wav[:min_len]

                # Ground Truth Truncation (如果需要保持原有逻辑，可以注释掉下面这行)
                # est_wav_trunc = ref_wav
                
                # Compute Metrics
                m = compute_metrics(ref_wav, est_wav_trunc, sr, dnsmos_eval)
                
                # Record
                row = {"Filename": filename}
                row.update(m)
                results.append(m)
                
                # Write to CSV immediately
                writer.writerow(row)
                
                # Optional: Save Audio
                if inf_cfg.get('save_audio', False):
                    save_wav_dir = os.path.join(exp_dir, "test_wavs")
                    os.makedirs(save_wav_dir, exist_ok=True)
                    sf.write(os.path.join(save_wav_dir, f"{filename}_rec.wav"), est_wav_trunc, sr)

    # 5. Summary
    print("\n" + "="*40)
    print("Average Results:")
    avg_res = {}
    for key in headers[1:]:
        vals = [r[key] for r in results]
        avg_res[key] = np.mean(vals)
        print(f"{key}: {avg_res[key]:.4f}")
    print("="*40)
    print(f"Detailed results saved to {output_csv_path}")

if __name__ == "__main__":
    main()