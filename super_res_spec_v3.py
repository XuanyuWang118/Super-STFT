import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple

from gompsnr.phase_related_losses import WeightedOmniPhaseLoss, CoupledOmniRILoss


def get_dft_bases(n_fft, round_pow_of_two=True):
    # Ref: https://en.wikipedia.org/wiki/DFT_matrix#Definition
    # FFT points
    N = 2 ** math.ceil(math.log2(n_fft)) if round_pow_of_two else n_fft
    # DFT{ δ[n - n0] } = exp(-j 2π k n0 / N), where N is n_fft
    delayed_delta = torch.eye(N)
    # (n_fft, N, 2)
    dft_bases = torch.view_as_real(torch.fft.fft(delayed_delta))
    return dft_bases


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
        window = torch.hann_window(self.n_fft).float()
        weight = get_dft_bases(self.n_fft)[:, : self.n_fft // 2 + 1].float()
        weight = weight.reshape(weight.size(0), -1).transpose(0, 1) * window.unsqueeze(0)
        # (n_fft + 2, 1, n_fft)
        self.enc.weight.data = weight.unsqueeze(1)
        self.enc.weight.requires_grad = True

    def forward(self, x):
        """
        x: (Batch, 1, Time)
        return: (Batch, Freq_bin, Time_frame, 2) [Real, Imag]
        """
        pad_amount = self.config.base_win_len // 2
        x_padded = F.pad(x, (pad_amount, pad_amount), mode='constant', value=0.0)
        feat = self.enc(x_padded) # (B, 2*F, T)
        cutoff = self.n_fft // 2 + 1
        real = feat[:, :cutoff, :] # (B, F, T)
        imag = feat[:, cutoff:, :] # (B, F, T)
        spec = torch.stack([real, imag], dim=-1)
        return spec


class SuperResDecoder(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.base_win_len
        self.in_channels = (self.n_fft // 2 + 1) * 2
        self.dec = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=config.base_win_len,
            stride=config.base_hop_len,
            bias=False,
            padding=0
        )

    def forward(self, spec):
        """
        spec: (Batch, Freq_bin, Time_frame, 2)
        return: (Batch, 1, Time)
        """
        real = spec[..., 0]
        imag = spec[..., 1]
        feat = torch.cat([real, imag], dim=1)
        waveform = self.dec(feat)
        pad_amount = self.config.base_win_len // 2
        if pad_amount > 0:
            waveform = waveform[:, :, pad_amount:-pad_amount]
        return waveform


class MultiResConsistencyLoss(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        
        # 创建可学习参数，键是 time_factor，值是权重向量
        self.kernel_win = nn.ParameterDict()
        for win_ms, hop_ms in self.config.target_resolutions:
            target_hop_len = int(self.config.sr * hop_ms / 1000)
            base_hop_len = self.config.base_hop_len
            
            time_factor = target_hop_len // base_hop_len
            time_factor = max(1, time_factor)
            
            key = str(time_factor)
            if time_factor > 1 and key not in self.kernel_win:
                weight = torch.ones(time_factor)
                self.kernel_win[key] = nn.Parameter(weight)

    def forward(self, pred_super_complex, raw_audio, verbose=False):
        """
        pred_super_complex: (B, F, T, 2) 复数频谱
        raw_audio: (B, 1, L)
        """
        total_loss = 0.0
        details = {}
        device = pred_super_complex.device
        
        base_win_len = self.config.base_win_len
        base_hop_len = self.config.base_hop_len
        
        if verbose:
            print("\n" + "="*60)
            print(f" [Loss Calculation] Base Complex Shape: {pred_super_complex.shape}")

        for i, (win_ms, hop_ms) in enumerate(self.config.target_resolutions):
            target_win_len = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)
            
            # 1. 计算 Ground Truth
            gt_complex_tensor = torch.stft(
                raw_audio.squeeze(1), 
                n_fft=target_win_len, 
                hop_length=target_hop_len, 
                win_length=target_win_len, 
                window=torch.hann_window(target_win_len).to(device),
                center=True,
                return_complex=True
            )
            gt_view = torch.view_as_real(gt_complex_tensor) # (B, F_gt, T_gt, 2) 
            
            curr_pred = pred_super_complex # (B, F, T, 2)
            
            # 2. 频率下采样
            freq_factor = base_win_len // target_win_len
            
            if freq_factor > 1:
                curr_pred = curr_pred.permute(0, 2, 3, 1) # (B, F, T, 2) -> (B, T, 2, F)

                B_temp, T_temp, C_temp, F_in = curr_pred.shape 
                curr_pred = curr_pred.reshape(-1, C_temp, F_in) # (B, T, 2, F) -> (B*T, 2, F)

                target_n_freq = target_win_len // 2 + 1
                expected_len = target_n_freq * freq_factor
                pad_amt = expected_len - F_in
                
                if pad_amt > 0:
                    curr_pred = F.pad(curr_pred, (0, pad_amt), mode='replicate')
                
                curr_pred = F.avg_pool1d(curr_pred, kernel_size=freq_factor, stride=freq_factor)
                curr_pred = curr_pred.view(B_temp, T_temp, C_temp, -1).permute(0, 3, 1, 2) # (B*T, 2, F_new) -> (B, T, 2, F_new) -> (B, F_new, T, 2)

            # 3. 时间下采样（相位对齐 + 可学习系数）
            time_factor = target_hop_len // base_hop_len
            time_factor = max(1, time_factor)
            
            if time_factor > 1:
                # curr_pred shape: (B, F, T, 2) -> Complex (B, F, T)
                curr_pred_complex = torch.view_as_complex(curr_pred.contiguous())
                B, n_freq, n_time = curr_pred_complex.shape
                
                remainder = n_time % time_factor
                if remainder != 0:
                    pad_amount = time_factor - remainder
                    curr_pred_complex = F.pad(curr_pred_complex, (0, pad_amount), mode='constant', value=0)

                n_time_padded = curr_pred_complex.shape[-1]
                n_blocks = n_time_padded // time_factor
                curr_pred_unfolded = curr_pred_complex.view(B, n_freq, n_blocks, time_factor) # (B, F, n_blocks, time_factor)
                
                freqs = torch.fft.rfftfreq(target_win_len, d=1.0/self.config.sr).to(device) # freqs: (F,)
                t_indices = torch.arange(time_factor).to(device) # t_indices: (time_factor,) -> [0, 1, ..., k-1]
                
                base_hop_sec = base_hop_len / self.config.sr
                # 广播 (F, 1) * (1, time_factor) -> (F, time_factor)
                theta = -2 * torch.pi * freqs.unsqueeze(1) * t_indices.unsqueeze(0) * base_hop_sec 
                
                # 生成复数旋转权重 e^(j*theta)
                rot_complex = torch.polar(torch.ones_like(theta), theta)
                
                # 可学习的时间权重
                kernel = self.kernel_win[str(time_factor)] # (time_factor,)
                weighted_spec = curr_pred_unfolded * rot_complex.unsqueeze(0).unsqueeze(2) * kernel.view(1, 1, 1, -1) # (B, F, n_blocks, time_factor)
                
                curr_pred_reduced = weighted_spec.mean(dim=3) # (B, F, n_blocks)
                curr_pred = torch.view_as_real(curr_pred_reduced) # (B, F, n_blocks, 2)
                
            # 4. 计算损失
            # curr_pred: (B, F, T_down, 2), gt_view: (B, F, T_gt, 2)
            assert curr_pred.shape == gt_view.shape, f"curr_pred.shape: {curr_pred.shape}, gt_view.shape: {gt_view.shape}"
            min_f = min(curr_pred.shape[1], gt_view.shape[1])
            min_t = min(curr_pred.shape[2], gt_view.shape[2])
            pred_crop = curr_pred[:, :min_f, :min_t, :]
            gt_crop = gt_view[:, :min_f, :min_t, :]
            
            # Real Part Loss
            pred_real = pred_crop[..., 0]
            gt_real = gt_crop[..., 0]
            loss_real = F.mse_loss(pred_real, gt_real)
            
            # Imag Part Loss
            pred_imag = pred_crop[..., 1]
            gt_imag = gt_crop[..., 1]
            loss_imag = F.mse_loss(pred_imag, gt_imag)
            
            # Log-Magnitude Loss
            pred_mag = torch.norm(pred_crop, dim=-1)
            gt_mag = torch.norm(gt_crop, dim=-1)
            loss_mag = F.mse_loss(torch.log(pred_mag + 1e-6), torch.log(gt_mag + 1e-6))
            
            current_total_loss = loss_real + loss_imag + loss_mag
            current_total_loss = current_total_loss * self.config.loss_weight[i]
            total_loss += current_total_loss
            details[f"{win_ms}ms_{hop_ms}ms"] = current_total_loss.item()
    
            if verbose:
                print(f" Target [{i}]: Win={win_ms}ms, Hop={hop_ms}ms, T_factor={time_factor}")
                print(f"   -> Aligned Shape: {curr_pred.shape}")
                print(f"   -> GT Shape:      {gt_view.shape}")
                print(f"   -> Loss Real:     {loss_real.item():.5f}")
                print(f"   -> Loss Imag:     {loss_imag.item():.5f}")
                print(f"   -> Loss Mag:      {loss_mag.item():.5f}")
                print(f"   -> Sum:           {current_total_loss.item():.5f}")

        return total_loss, details


class GOMPSNRLoss(nn.Module):
    def __init__(self, config: STFTConfig, wop_weight=1.0, cori_weight=1.0):
        super().__init__()
        self.config = config
        self.wop_weight = wop_weight
        self.cori_weight = cori_weight
        
        self.wop_loss = WeightedOmniPhaseLoss(alpha=100) 
        self.cori_loss = CoupledOmniRILoss(mag_dist_type="L1")

        self.win_len = config.base_win_len
        self.hop_len = config.base_hop_len
        self.n_fft = config.base_win_len
        self.register_buffer('window', torch.hann_window(self.win_len))

    def forward(self, x_recon, x_gt):
        """
        x_recon: (Batch, 1, Time) or (Batch, Time) - Decoder 生成的波形
        x_gt:    (Batch, 1, Time) or (Batch, Time) - 真实波形
        """
        if x_recon.dim() == 3:
            x_recon = x_recon.squeeze(1)
        if x_gt.dim() == 3:
            x_gt = x_gt.squeeze(1)

        # 1. 执行 STFT 变换
        spec_recon = torch.stft(
            x_recon, n_fft=self.n_fft, hop_length=self.hop_len, 
            win_length=self.win_len, window=self.window, 
            return_complex=True
        )
        spec_gt = torch.stft(
            x_gt, n_fft=self.n_fft, hop_length=self.hop_len, 
            win_length=self.win_len, window=self.window, 
            return_complex=True
        )

        # 2. 提取论文 Loss 所需的特征
        rea_g, imag_g = spec_recon.real, spec_recon.imag
        rea, imag = spec_gt.real, spec_gt.imag
        mag = torch.sqrt(rea**2 + imag**2 + 1e-8)
        pha_g = torch.atan2(imag_g, rea_g)
        pha = torch.atan2(imag, rea)

        loss_dict = {}
        total_loss = 0.0

        # 3. 计算 WOP Loss (加权全向相位损失)
        if self.wop_weight > 0:
            # WOP forward: (phase_target, phase_estimate, mag_target)
            l_wop = self.wop_loss(pha, pha_g, mag)
            total_loss += l_wop * self.wop_weight
            loss_dict['loss_wop'] = l_wop.item()

        # 4. 计算 CORI Loss (耦合全向实虚部损失)
        if self.cori_weight > 0:
            # CORI forward: (rea_target, imag_target, rea_est, imag_est)
            l_cori = self.cori_loss(rea, imag, rea_g, imag_g)
            total_loss += l_cori * self.cori_weight
            loss_dict['loss_cori'] = l_cori.item()

        return total_loss, loss_dict
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    # 1. Config
    cfg = STFTConfig()
    print(f"Config: {cfg}")
    
    # 2. Models
    encoder = SuperResEncoder(cfg).to(device)
    decoder = SuperResDecoder(cfg).to(device)
    criterion = MultiResConsistencyLoss(cfg).to(device)
    decoder_criterion = GOMPSNRLoss(cfg).to(device)
    
    # 3. Data
    rand_len = int(16000 * 1.0)
    x = torch.randn(1, 1, rand_len).to(device)
    print(f"\nInput Audio: {x.shape}")
    
    # 4. Encoder Forward
    z_complex = encoder(x)
    print(f"Encoder Output (Complex): {z_complex.shape}")
    
    # 5. Loss Calculation
    multi_loss, multi_details = criterion(z_complex, x, verbose=True)
    print(f"\nMultiResConsistency Loss: {multi_loss.item():.6f}, Details: {multi_details}")
    
    # 6. Decoder Forward
    x_recon = decoder(z_complex)
    print(f"\nDecoder Output (Waveform): {x_recon.shape}")
    
    # 7. Check Reconstruction
    recon_loss = F.mse_loss(x, x_recon)
    print(f"\nReconstruction MSE Loss: {recon_loss.item():.6f}")

    # 8. GOMPSNR Loss Calculation
    gompsnr_loss, gompsnr_details = decoder_criterion(x_recon, x)
    print(f"\nGOMPSNR Loss: {gompsnr_loss.item():.6f}, Details: {gompsnr_details}")
    
    # 9. Test Gradients
    optim = torch.optim.Adam(list(encoder.parameters()) + 
                             list(decoder.parameters()) + 
                             list(criterion.parameters()), lr=1e-3)
    optim.zero_grad()
    total_loss = multi_loss + recon_loss + gompsnr_loss
    total_loss.backward()
    optim.step()
    print("\nBackprop successful. Learnable kernels updated.")
    # print("Learned Time Downsample Kernels:")
    # for k, v in criterion.kernel_win.items():
    #     print(f"  Factor {k}: {v.data.cpu().numpy()}")