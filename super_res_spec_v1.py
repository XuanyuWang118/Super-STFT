import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple

class STFTConfig:
    def __init__(self, sample_rate: int = 16000, target_resolutions: List[Tuple[int, int]] = None, super_hop_ms: float = 1.0):
        self.sr = sample_rate
        self.super_hop_ms = super_hop_ms
        
        if target_resolutions is None:
            self.target_resolutions = [(64, 32), (32, 16), (16, 8), (8, 4), (4, 2), (2, 1)]
        else:
            self.target_resolutions = target_resolutions

        self.loss_weight = [1.0 for _ in self.target_resolutions]

        self.max_win_ms = max([r[0] for r in self.target_resolutions])
        
        self.base_win_len = int(self.sr * self.max_win_ms / 1000)
        self.base_hop_len = int(self.sr * self.super_hop_ms / 1000)
        if self.base_win_len % 2 != 0:
            self.base_win_len += 1

    def __repr__(self):
        return (f"<STFTConfig SR={self.sr}, "
                f"BaseWin={self.base_win_len}({self.max_win_ms}ms), "
                f"BaseHop={self.base_hop_len}({self.super_hop_ms}ms), "
                f"Targets={len(self.target_resolutions)} configs>")


class SuperResEncoder(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.base_win_len
        self.out_channels = (self.n_fft // 2 + 1) * 2
        self.enc = nn.Conv1d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=config.base_win_len,
            stride=config.base_hop_len,
            bias=False,
            padding=0
        )
        self._init_weights()

    def _init_weights(self):
        n = np.arange(self.n_fft)
        k = np.arange(self.n_fft // 2 + 1)[:, None]
        window = np.hanning(self.n_fft)
        real_part = np.cos(2 * np.pi * k * n / self.n_fft) * window
        imag_part = np.sin(2 * np.pi * k * n / self.n_fft) * window
        weight = np.concatenate([real_part, imag_part], axis=0)
        self.enc.weight.data = torch.from_numpy(weight).float().unsqueeze(1)
        self.enc.weight.requires_grad = True

    def forward(self, x):
        """x: (Batch, 1, Time) - 原始音频"""
        pad_amount = self.config.base_win_len // 2
        x_padded = F.pad(x, (pad_amount, pad_amount), mode='constant', value=0.0)
        feat = self.enc(x_padded)
        
        cutoff = self.n_fft // 2 + 1
        real = feat[:, :cutoff, :]
        imag = feat[:, cutoff:, :]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        return mag

class SuperResDecoder(nn.Module):
### Todo: 实现 SuperResDecoder



class MultiResConsistencyLoss(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        
    def forward(self, pred_super_mag, raw_audio, verbose=False):
        """
        pred_super_mag: Encoder 的输出
        raw_audio: 原始音频
        verbose: 是否打印中间变量
        """
        total_loss = 0.0
        details = {}
        
        base_win_len = self.config.base_win_len
        base_hop_len = self.config.base_hop_len
        
        if verbose:
            print("\n" + "="*60)
            print(f" [Loss Calculation Debug] Base Shape: {pred_super_mag.shape}")
            print(f" [Audio Length]: {raw_audio.shape[-1]} samples")
            print("-" * 60)

        for i, (win_ms, hop_ms) in enumerate(self.config.target_resolutions):
            target_win_len = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)
            
            # 计算 Ground Truth Magnitude
            gt_complex = torch.stft(
                raw_audio.squeeze(1), 
                n_fft=target_win_len, 
                hop_length=target_hop_len, 
                win_length=target_win_len, 
                window=torch.hann_window(target_win_len).to(raw_audio.device),
                center=True,
                return_complex=True
            )
            gt_mag = torch.abs(gt_complex)
            
            # 下采样预测的 Super Magnitude
            curr_pred = pred_super_mag
            
            # 1. 频率下采样
            freq_factor = base_win_len // target_win_len
            if freq_factor > 1:
                curr_pred = curr_pred.permute(0, 2, 1)
                curr_pred = F.avg_pool1d(curr_pred, kernel_size=freq_factor, stride=freq_factor)
                curr_pred = curr_pred.permute(0, 2, 1)
            
            print(target_win_len, torch.fft.rfftfreq(target_win_len, 1.0 / 16000)[1])

            # 2. 时间下采样
            time_factor = target_hop_len // base_hop_len
            time_factor = max(1, time_factor)
            curr_pred = curr_pred[:, :, ::time_factor]

            # if time_factor > 1:
            #     curr_pred = F.avg_pool1d(curr_pred, kernel_size=time_factor, stride=time_factor)
            
            # 裁剪到相同尺寸
            min_f = min(curr_pred.shape[1], gt_mag.shape[1])
            min_t = min(curr_pred.shape[2], gt_mag.shape[2])
            
            pred_crop = curr_pred[:, :min_f, :min_t]
            gt_crop = gt_mag[:, :min_f, :min_t]
            
            # 计算 Loss
            loss_val = F.mse_loss(torch.log(pred_crop + 1e-6), torch.log(gt_crop + 1e-6))
            loss_val = loss_val * self.config.loss_weight[i]
            total_loss += loss_val
            details[f"{win_ms}ms_{hop_ms}ms"] = loss_val.item()
            
            if verbose:
                print(f" Target [{i+1}]: Win={win_ms}ms({target_win_len}), Hop={hop_ms}ms({target_hop_len})")
                print(f"   -> Freq Factor: {freq_factor} (Pool)")
                print(f"   -> Time Factor: {time_factor} (Slice)")
                print(f"   -> Shape Trans: {pred_super_mag.shape} => {curr_pred.shape}")
                print(f"   -> GT Shape:    {gt_mag.shape}")
                print(f"   -> Final Crop:  ({min_f}, {min_t})")
                print(f"   -> MSE Loss:    {loss_val.item():.5f}")
                print("-" * 30)

        if verbose:
            print("="*60 + "\n")
            
        return total_loss, details


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    cfg = STFTConfig(sample_rate=16000, target_resolutions=[(64, 32), (32, 16), (16, 8), (8, 4), (4, 2), (2, 1)]) 
    print(f"Config: {cfg}")
    
    model = SuperResEncoder(cfg).to(device)
    criterion = MultiResConsistencyLoss(cfg).to(device)
    
    rand_len = int(16000 * 1.00)
    x_random = torch.randn(1, 1, rand_len).to(device)
    print(f"Input Shape (Raw): {x_random.shape}")
    
    y_super = model(x_random)
    print(f"Super Spec Output: {y_super.shape}")
    
    loss, details = criterion(y_super, x_random, verbose=True)
    print(f"Total Loss: {loss.item():.4f}\n")
    print(f"Loss Details: {details}\n")
