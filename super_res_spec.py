from email.mime import audio

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import yaml
from typing import List, Tuple, Dict, Any

from stft.gompsnr.phase_related_losses import WeightedOmniPhaseLoss, CoupledOmniRILoss

def get_dft_bases(n_fft, round_pow_of_two=True):
    N = 2 ** math.ceil(math.log2(n_fft)) if round_pow_of_two else n_fft
    delayed_delta = torch.eye(N)
    dft_bases = torch.view_as_real(torch.fft.fft(delayed_delta))
    return dft_bases


def create_triangular_filterbank(n_freq_in, n_freq_out, sr):
    """
    创建从高分辨率(n_freq_in)到低分辨率(n_freq_out)的三角滤波器组矩阵
    返回矩阵形状: (n_freq_out, n_freq_in)
    """
    # 1. 定义频率轴 (Hz)
    # n_freq_in 对应 0 ~ sr/2
    f_in = torch.linspace(0, sr / 2, n_freq_in)
    f_out = torch.linspace(0, sr / 2, n_freq_out)
    
    # 2. 计算低分辨率的带宽 (Bin Width)
    # f_out 是均匀分布的，间距即为带宽
    if n_freq_out > 1:
        delta_f = f_out[1] - f_out[0]
    else:
        delta_f = sr / 2 # 只有一个点的情况
    
    # 3. 利用广播计算权重矩阵 (F_out, F_in)
    # shape: (F_out, 1) - (1, F_in) -> (F_out, F_in)
    diff = torch.abs(f_out.unsqueeze(1) - f_in.unsqueeze(0))
    
    # 三角核公式: max(0, 1 - |f_in - f_out| / delta_f)
    weights = torch.clamp(1 - diff / delta_f, min=0)
    
    # 4. 归一化 (按行归一化，确保每个目标频点的能量守恒)
    # 加上 eps 防止除以 0
    row_sums = weights.sum(dim=1, keepdim=True)
    weights = weights / (row_sums + 1e-8)
    
    return weights


class STFTConfig:
    def __init__(self, cfg_dict: Dict[str, Any] = None, yaml_path: str = None):
        """
        初始化配置。可以通过字典传入，也可以通过 yaml 路径加载。
        """
        if yaml_path:
            with open(yaml_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
        
        if cfg_dict is None:
            cfg_dict = {
                'audio': {'sample_rate': 16000},
                'stft': {'target_resolutions': [(64, 32), (32, 16), (16, 8), (8, 4), (4, 2), (2, 1)], 'super_hop_ms': 1.0},
                'model': {'encoder_init': 'dft', 'decoder_init': 'idft'},
                'loss': {}
            }

        # 1. Audio Params
        self.sr = cfg_dict.get('audio', {}).get('sample_rate', 16000)

        # 2. STFT Params
        stft_cfg = cfg_dict.get('stft', {})
        self.target_resolutions = stft_cfg.get('target_resolutions', [(64, 32)])
        self.super_hop_ms = stft_cfg.get('super_hop_ms', 1.0)
        
        # 3. Model Params
        model_cfg = cfg_dict.get('model', {})
        self.encoder_init = model_cfg.get('encoder_init', 'dft')
        self.decoder_init = model_cfg.get('decoder_init', 'idft')

        # 4. Loss Params
        loss_cfg = cfg_dict.get('loss', {})

        # (1) Multi-Resolution Consistency Loss Weights
        multi_res_weight = loss_cfg.get('multi_res_weight', {})
        self.multi_res_weight = multi_res_weight.get('weight', 1.0)

        r_w = multi_res_weight.get('real_weight', 1.0)
        i_w = multi_res_weight.get('imag_weight', 1.0)
        m_w = multi_res_weight.get('mag_weight', 1.0)
        w_sum = r_w + i_w + m_w + 1e-8
        self.real_weight = r_w / w_sum
        self.imag_weight = i_w / w_sum
        self.mag_weight = m_w / w_sum

        res_weights = multi_res_weight.get('resolution_weights', [])
        if res_weights and len(res_weights) == len(self.target_resolutions):
            total = sum(res_weights) + 1e-8
            self.resolution_weights = [w / total for w in res_weights]
        else:
            self.resolution_weights = [1.0 / len(self.target_resolutions)] * len(self.target_resolutions)

        # (2) GOMPSNR Loss Weights 
        gompsnr_weight = loss_cfg.get('gompsnr_weight', {})
        self.gompsnr_weight = gompsnr_weight.get('weight', 1.0)
        w_w = gompsnr_weight.get('wop_weight', 1.0)
        c_w = gompsnr_weight.get('cori_weight', 1.0)
        w_sum_g = w_w + c_w + 1e-8
        self.wop_weight = w_w / w_sum_g
        self.cori_weight = c_w / w_sum_g
        self.wop_alpha = gompsnr_weight.get('wop_alpha', 100)
        self.mag_dist_type = gompsnr_weight.get('mag_dist_type', 'L1')

        # (3) Reconstruction Loss Weight
        self.recon_weight = loss_cfg.get('recon_weight', 1.0)
        self.sisnr_weight = loss_cfg.get('sisnr_weight', 1.0)

        # 5. Derived Params (Calculated)
        self.max_win_ms = max([r[0] for r in self.target_resolutions])
        self.base_win_len = int(self.sr * self.max_win_ms / 1000)
        self.base_hop_len = int(self.sr * self.super_hop_ms / 1000)
        
        if self.base_win_len % 2 != 0:
            self.base_win_len += 1

    def __repr__(self):
        return (f"STFTConfig(sr={self.sr}, target_resolutions={self.target_resolutions}, "
                f"super_hop_ms={self.super_hop_ms}, encoder_init='{self.encoder_init}', decoder_init='{self.decoder_init}', "
                f"multi_res_weight={self.multi_res_weight}, resolution_weights={self.resolution_weights}, "
                f"real_weight={self.real_weight}, imag_weight={self.imag_weight}, mag_weight={self.mag_weight}, "
                f"gompsnr_weight={self.gompsnr_weight}, wop_weight={self.wop_weight}, cori_weight={self.cori_weight}, wop_alpha={self.wop_alpha}, mag_dist_type='{self.mag_dist_type}', "
                f"recon_weight={self.recon_weight}, sisnr_weight={self.sisnr_weight}, base_win_len={self.base_win_len}, base_hop_len={self.base_hop_len})")


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
        if self.config.encoder_init == 'dft':
            window = torch.hann_window(self.n_fft).float()
            weight = get_dft_bases(self.n_fft)[:, : self.n_fft // 2 + 1].float()
            weight = weight.reshape(weight.size(0), -1).transpose(0, 1) * window.unsqueeze(0)
            self.enc.weight.data = weight.unsqueeze(1)
        else:
            nn.init.kaiming_normal_(self.enc.weight)
        
        self.enc.weight.requires_grad = True

    def forward(self, x):
        """
        x: (Batch, 1, Time)
        return: (Batch, Freq_bin, Time_frame, 2) [Real, Imag]
        """
        pad_amount = self.config.base_win_len // 2
        x_padded = F.pad(x, (pad_amount, pad_amount), mode='constant', value=0.0)
        feat = self.enc(x_padded) 
        cutoff = self.n_fft // 2 + 1
        real = feat[:, :cutoff, :] 
        imag = feat[:, cutoff:, :] 
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

    # def forward(self, spec):
    #     """
    #     spec: (Batch, Freq_bin, Time_frame, 2)
    #     return: (Batch, 1, Time)
    #     """
    #     real = spec[..., 0]
    #     imag = spec[..., 1]
    #     feat = torch.cat([real, imag], dim=1)
    #     waveform = self.dec(feat)
    #     pad_amount = self.config.base_win_len // 2
    #     if pad_amount > 0:
    #         waveform = waveform[:, :, pad_amount:-pad_amount]
    #     return waveform
    
    def forward(self, spec, target_len=None):
        """
        spec: (Batch, Freq_bin, Time_frame, 2)
        target_len: (Batch,) Tensor or int or None
        return: (Batch, 1, Time) 
        """
        real = spec[..., 0]
        imag = spec[..., 1]
        feat = torch.cat([real, imag], dim=1)
        
        conv_target_len = None
        if target_len is not None:
            if torch.is_tensor(target_len):
                actual_target_len = int(target_len.max().item())
            else:
                actual_target_len = int(target_len)
            conv_target_len = (actual_target_len + self.n_fft,)
        
        waveform = self.dec(feat, output_size=conv_target_len)
        pad_amount = self.n_fft // 2
        if pad_amount > 0:
            if target_len is not None:
                waveform = waveform[:, :, pad_amount : pad_amount + actual_target_len]
            else:
                waveform = waveform[:, :, pad_amount : -pad_amount]
        return waveform


class MultiResConsistencyLoss(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.kernel_win = nn.ParameterDict()
        self.freq_downsamplers = nn.ModuleDict()

        for win_ms, hop_ms in self.config.target_resolutions:

            # 1. Time Downsampling Kernels
            target_hop_len = int(self.config.sr * hop_ms / 1000)
            base_hop_len = self.config.base_hop_len
            time_factor = target_hop_len // base_hop_len
            time_factor = max(1, time_factor)
            key = str(time_factor)
            if time_factor > 1 and key not in self.kernel_win:
                weight = torch.ones(time_factor)
                self.kernel_win[key] = nn.Parameter(weight)
            
            # 2. Frequency Downsampling Matrices
            target_win_len = int(self.config.sr * win_ms / 1000)
            base_win_len = self.config.base_win_len
            if base_win_len > target_win_len:
                n_freq_in = base_win_len // 2 + 1
                n_freq_out = target_win_len // 2 + 1
                mat = create_triangular_filterbank(n_freq_in, n_freq_out, self.config.sr)
                key_freq = f"mat_{win_ms}_{hop_ms}"
                self.register_buffer(key_freq, mat)

    def forward(self, pred_super_complex, raw_audio, verbose=False):
        total_loss = 0.0
        details = {}
        device = pred_super_complex.device
        
        base_win_len = self.config.base_win_len
        base_hop_len = self.config.base_hop_len
        
        if verbose:
            print(f"\n[Loss Debug] Base Complex Shape: {pred_super_complex.shape}")

        for i, (win_ms, hop_ms) in enumerate(self.config.target_resolutions):
            target_win_len = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)
            
            # 1. GT
            gt_complex_tensor = torch.stft(
                raw_audio.squeeze(1), 
                n_fft=target_win_len, 
                hop_length=target_hop_len, 
                win_length=target_win_len, 
                window=torch.hann_window(target_win_len).to(device),
                center=True,
                return_complex=True
            )
            gt_view = torch.view_as_real(gt_complex_tensor) 
            
            curr_pred = pred_super_complex
            
            # 2. Freq Downsampling
            if base_win_len > target_win_len:
                key_freq = f"mat_{win_ms}_{hop_ms}"
                W = getattr(self, key_freq)
                curr_pred = torch.einsum('bitc,oi->botc', curr_pred, W)

            # 3. Time Downsampling
            time_factor = target_hop_len // base_hop_len
            time_factor = max(1, time_factor)
            
            if time_factor > 1:
                curr_pred_complex = torch.view_as_complex(curr_pred.contiguous())
                B, n_freq, n_time = curr_pred_complex.shape
                
                remainder = n_time % time_factor
                if remainder != 0:
                    pad_amount = time_factor - remainder
                    curr_pred_complex = F.pad(curr_pred_complex, (0, pad_amount), mode='constant', value=0)

                n_time_padded = curr_pred_complex.shape[-1]
                n_blocks = n_time_padded // time_factor
                curr_pred_unfolded = curr_pred_complex.view(B, n_freq, n_blocks, time_factor)
                
                freqs = torch.fft.rfftfreq(target_win_len, d=1.0/self.config.sr).to(device)
                t_indices = torch.arange(time_factor).to(device)
                
                base_hop_sec = base_hop_len / self.config.sr
                theta = -2 * torch.pi * freqs.unsqueeze(1) * t_indices.unsqueeze(0) * base_hop_sec 
                
                rot_complex = torch.polar(torch.ones_like(theta), theta)
                kernel = self.kernel_win[str(time_factor)]
                weighted_spec = curr_pred_unfolded * rot_complex.unsqueeze(0).unsqueeze(2) * kernel.view(1, 1, 1, -1)
                
                curr_pred_reduced = weighted_spec.mean(dim=3)
                curr_pred = torch.view_as_real(curr_pred_reduced)
                
            # 4. Loss
            assert curr_pred.shape == gt_view.shape, f"Shape mismatch: Pred {curr_pred.shape}, GT {gt_view.shape}"
            min_f = min(curr_pred.shape[1], gt_view.shape[1])
            min_t = min(curr_pred.shape[2], gt_view.shape[2])
            pred_crop = curr_pred[:, :min_f, :min_t, :]
            gt_crop = gt_view[:, :min_f, :min_t, :]
            
            pred_real = pred_crop[..., 0]
            gt_real = gt_crop[..., 0]
            loss_real = F.mse_loss(pred_real, gt_real)
            
            pred_imag = pred_crop[..., 1]
            gt_imag = gt_crop[..., 1]
            loss_imag = F.mse_loss(pred_imag, gt_imag)
            
            pred_mag = torch.norm(pred_crop, dim=-1)
            gt_mag = torch.norm(gt_crop, dim=-1)
            pred_mag = torch.clamp(pred_mag, min=1e-6)
            gt_mag = torch.clamp(gt_mag, min=1e-6)
            loss_mag = F.mse_loss(torch.log(pred_mag), torch.log(gt_mag))
            
            current_total_loss = loss_real * self.config.real_weight + loss_imag * self.config.imag_weight + loss_mag * self.config.mag_weight
            current_total_loss = current_total_loss * self.config.resolution_weights[i]
            
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
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        
        self.wop_loss = WeightedOmniPhaseLoss(alpha=config.wop_alpha) 
        self.cori_loss = CoupledOmniRILoss(mag_dist_type=config.mag_dist_type)

        self.win_len = config.base_win_len
        self.hop_len = config.base_hop_len
        self.n_fft = config.base_win_len
        self.register_buffer('window', torch.hann_window(self.win_len))

    def forward(self, x_recon, x_gt):
        if x_recon.dim() == 3: x_recon = x_recon.squeeze(1)
        if x_gt.dim() == 3: x_gt = x_gt.squeeze(1)

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

        rea_g, imag_g = spec_recon.real, spec_recon.imag
        rea, imag = spec_gt.real, spec_gt.imag
        eps = 1e-6
        rea_g = rea_g + eps
        imag_g = imag_g + eps
        rea = rea + eps
        imag = imag + eps
        mag = torch.sqrt(rea**2 + imag**2 + 1e-8)
        pha_g = torch.atan2(imag_g, rea_g)
        pha = torch.atan2(imag, rea)

        loss_dict = {}
        total_loss = 0.0

        if self.config.wop_weight > 0:
            l_wop = self.wop_loss(pha, pha_g, mag)
            total_loss += l_wop * self.config.wop_weight
            loss_dict['loss_wop'] = l_wop.item()

        if self.config.cori_weight > 0:
            l_cori = self.cori_loss(rea, imag, rea_g, imag_g)
            total_loss += l_cori * self.config.cori_weight
            loss_dict['loss_cori'] = l_cori.item()

        return total_loss, loss_dict
    

class SISNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        """
        Scale-Invariant Signal-to-Noise Ratio Loss
        """
        super().__init__()
        self.eps = eps

    def forward(self, preds, target):
        """
        preds: (Batch, 1, Time) - Decoder 重构的波形
        target: (Batch, 1, Time) - 原始参考波形
        """
        # 去掉 Channel 维度
        preds = preds.squeeze(1)
        target = target.squeeze(1)

        # 1. 中心化 (Zero-mean)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

        # 2. 计算目标投影: target_proj = <preds, target> * target / ||target||^2
        dot_product = torch.sum(preds * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target**2, dim=-1, keepdim=True) + self.eps
        target_projected = (dot_product / target_energy) * target

        # 3. 计算噪声能量: noise = preds - target_proj
        noise = preds - target_projected
        
        sig_energy = torch.sum(target_projected**2, dim=-1)
        noise_energy = torch.sum(noise**2, dim=-1)

        # 4. 计算 SI-SNR: 10 * log10(sig_energy / noise_energy)
        si_snr = 10 * torch.log10(sig_energy / (noise_energy + self.eps) + self.eps)

        # 返回负均值作为 Loss
        return -torch.mean(si_snr)


# class SuperResEncoderWrapper(nn.Module):
#     def __init__(self, config, checkpoint_path, device="cuda"):
#         super().__init__()
#         self.config = config
#         # 1. 实例化你原本的 Encoder
#         # from super_res_spec import SuperResEncoder
#         self.model = SuperResEncoder(config)
        
#         # 2. 加载预训练权重
#         print(f"Loading Super-Res Encoder from {checkpoint_path}")
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         self.model.load_state_dict(ckpt['encoder'])
        
#         # 3. 冻结参数
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model.eval()

#     def forward(self, speech, lengths, fs=None):
#         """
#         speech: (B, L)
#         lengths: (B,)
#         """
#         # 适配输入维度 (B, 1, L)
#         if speech.dim() == 2:
#             speech = speech.unsqueeze(1)
            
#         # 调用原始 Encoder: 输出 (B, F, T, 2)
#         z_complex_real_view = self.model(speech)
        
#         # 维度转换: (B, F, T, 2) -> (B, T, F, 2)
#         z = z_complex_real_view.permute(0, 2, 1, 3).contiguous()
        
#         # 转换为 ComplexTensor (B, T, F) 以适配 BSRNN 默认行为
#         z_complex = torch.view_as_complex(z)
        
#         # 计算特征长度 (对应 1ms 的 hop)
#         # 根据你的 padding 逻辑，长度通常是 L // hop + 1
#         hop = self.config.base_hop_len
#         flens = torch.div(lengths, hop, rounding_mode='floor') + 1
        
#         return z_complex, flens

# class SuperResDecoderWrapper(nn.Module):
#     def __init__(self, config, checkpoint_path, device="cuda"):
#         super().__init__()
#         self.config = config
#         # from super_res_spec import SuperResDecoder
#         self.model = SuperResDecoder(config)
        
#         # 加载并冻结
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         self.model.load_state_dict(ckpt['decoder'])
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model.eval()

#     def forward(self, feature, lengths, fs=None):
#         """
#         feature: (B, T, F) complex
#         lengths: (B,) 原始音频点数长度
#         """
#         # 1. 转换为实数视图: (B, T, F, 2)
#         z_real_view = torch.view_as_real(feature)
        
#         # 2. 还原回你 Decoder 期望的维度: (B, F, T, 2)
#         z = z_real_view.permute(0, 2, 1, 3).contiguous()
        
#         # 3. 调用原始 Decoder: 输出 (B, 1, L)
#         recon_wav = self.model(z)

#         # 根本上解决长度问题
#         # target_len = lengths.max().item()
#         # recon_wav = self.model(z, target_len=target_len)

#         # 暂时解决长度问题
#         target_len = lengths.max().item()
#         current_len = recon_wav.shape[-1]
#         if current_len < target_len:
#             # 如果生成的音频短了，在尾部补零
#             diff = target_len - current_len
#             recon_wav = F.pad(recon_wav, (0, diff), mode='constant', value=0.0)
#         elif current_len > target_len:
#             # 如果生成的音频长了（通常不会），进行截断
#             recon_wav = recon_wav[:, :, :target_len]
        
#         # 4. 裁剪长度以匹配原始输入
#         # 注意：转置卷积可能会产生稍微多出的采样点
#         # recon_wav = recon_wav[:, 0, :lengths.max()] # 简单做法
        
#         return recon_wav.squeeze(1), lengths # 返回 (B, L)


# class SuperResEncoderWrapper(nn.Module):
#     def __init__(self, config, checkpoint_path, device="cuda"):
#         super().__init__()
#         self.config = config
#         # 1. 实例化 Encoder (假设 SuperResEncoder 已在同文件中定义)
#         self.model = SuperResEncoder(config)
        
#         # 2. 加载预训练权重
#         print(f"Loading Super-Res Encoder from {checkpoint_path}")
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         self.model.load_state_dict(ckpt['encoder'])
        
#         # 3. 冻结参数
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model.eval()

#     def forward(self, speech, lengths, fs=None):
#         """
#         speech: (Batch, Time)
#         lengths: (Batch,)
#         """
#         # 适配输入维度 (B, 1, L)
#         if speech.dim() == 2:
#             speech = speech.unsqueeze(1)
            
#         # 1. 调用原始 Encoder: 输出 (Batch, Freq, Time, 2)
#         z_real_view = self.model(speech)
        
#         # 2. 维度转换: (B, F, T, 2) -> (B, T, F, 2)
#         z = z_real_view.permute(0, 2, 1, 3).contiguous()
        
#         # 3. 转换为 ComplexTensor (B, T, F) 以适配 ESPnet BSRNN 接口
#         z_complex = torch.view_as_complex(z)
        
#         # 4. 计算特征长度 (根据 Stride=16 计算帧数)
#         hop = self.config.base_hop_len
#         # 由于 Encoder 内部做了 Win//2 的 padding， flens = L // hop + 1
#         flens = torch.div(lengths, hop, rounding_mode='floor') + 1
        
#         return z_complex, flens

# class SuperResDecoderWrapper(nn.Module):
#     def __init__(self, config, checkpoint_path, device="cuda"):
#         super().__init__()
#         self.config = config
#         # 1. 实例化 Decoder (假设 SuperResDecoder 已在同文件中定义)
#         self.model = SuperResDecoder(config)
        
#         # 2. 加载并冻结参数
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         self.model.load_state_dict(ckpt['decoder'])
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model.eval()

#     def forward(self, feature, lengths, fs=None):
#         """
#         feature: (Batch, Time_frame, Freq_bin) complex
#         lengths: (Batch,) 原始音频采样点数
#         """
#         # 1. 转换为实数视图并还原内部维度: (B, T, F, 2) -> (B, F, T, 2)
#         z_real_view = torch.view_as_real(feature)
#         z = z_real_view.permute(0, 2, 1, 3).contiguous()
        
#         # 2. 调用优化后的 Decoder (通过 target_len 参数在底层卷积实现完美对齐)
#         # 这种方式不仅满足了 ESPnet 的 shape 校验，还保留了卷积边缘的信息恢复
#         recon_wav = self.model(z, target_len=lengths)
        
#         # 返回 (Batch, Time) 格式
#         return recon_wav.squeeze(1), lengths


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    # 1. 加载配置 (优先加载 yaml)
    import os
    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        cfg = STFTConfig(yaml_path=config_path)
    else:
        print("Config file not found, using defaults")
        cfg = STFTConfig()
        
    print(f"Config: {cfg}")
    
    # 2. Models
    encoder = SuperResEncoder(cfg).to(device)
    decoder = SuperResDecoder(cfg).to(device)
    mrc_loss = MultiResConsistencyLoss(cfg).to(device)
    gompsnr_loss = GOMPSNRLoss(cfg).to(device)
    sisnr_loss = SISNRLoss().to(device)
    
    # 3. Data
    rand_len = int(38791 * 1.0)
    x = torch.randn(1, 1, rand_len).to(device)
    print(f"\nInput Audio: {x.shape}")
    
    # 4. Forward & Backward
    z_complex = encoder(x)
    multi_loss, multi_details = mrc_loss(z_complex, x, verbose=False)
    x_recon = decoder(z_complex, torch.tensor([rand_len]))
    recon_loss = F.mse_loss(x, x_recon)
    gompsnr_loss, gompsnr_details = gompsnr_loss(x_recon, x)
    sisnr_loss = sisnr_loss(x_recon, x)

    print(f"Output audio: {x_recon.shape}")
    
    # Aggregate
    total_loss = multi_loss * cfg.multi_res_weight + recon_loss * cfg.recon_weight + gompsnr_loss * cfg.gompsnr_weight + sisnr_loss * cfg.sisnr_weight
    
    print(f"\nTotal Loss: {total_loss.item():.6f}")
    print(f"  MultiRes: {multi_loss.item():.6f}")
    print(f"  Recon:    {recon_loss.item():.6f}")
    print(f"  GOMPSNR:  {gompsnr_loss.item():.6f}")
    print(f"  SI-SNR:   {sisnr_loss.item():.6f}")

    optim = torch.optim.Adam(list(encoder.parameters()) + 
                             list(decoder.parameters()) + 
                             list(mrc_loss.parameters()), lr=1e-3)
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    print("\nBackprop successful.")