import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from typing import Dict, Any
import soundfile as sf

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
        self.segment_length = cfg_dict.get('audio', {}).get('segment_length', 1.0)

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

        self.mask_type = loss_cfg.get('mask_type', 'none')

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
        self.sisnr_weight = loss_cfg.get('sisnr_weight', 0.0)

        # (4) 其他 Loss Weights 可以在这里添加
        self.entropy_weight = loss_cfg.get('entropy_weight', 0.0)
        self.tv_weight = loss_cfg.get('tv_weight', 0.0)

        # 5. Derived Params (Calculated)
        self.max_win_ms = max([r[0] for r in self.target_resolutions])
        self.base_win_len = int(self.sr * self.max_win_ms / 1000)
        self.base_hop_len = int(self.sr * self.super_hop_ms / 1000)
        
        if self.base_win_len % 2 != 0:
            self.base_win_len += 1

    def __repr__(self):
        return (f"STFTConfig(mask_type='{self.mask_type}', sr={self.sr}, segment_length={self.segment_length}, target_resolutions={self.target_resolutions}, "
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

        # =====================================================================
        # Control Variable: 'none', '2_extreme', or '6_all'
        # Defaulting to '2_extreme' if not explicitly set in config
        # =====================================================================
        mask_type = getattr(self.config, 'mask_type', '2_extreme')
        global_mask = None

        if mask_type != 'none':
            with torch.no_grad():
                if mask_type == '6_all':
                    # [Method 1] 6-Resolution Intersection Mask (log-space geometric mean)
                    num_res = len(self.config.target_resolutions)
                    log_sum = None
                    for win_ms, hop_ms in self.config.target_resolutions:
                        n_fft_curr = int(self.config.sr * win_ms / 1000)
                        stft_curr = torch.stft(
                            raw_audio.squeeze(1), n_fft=n_fft_curr, hop_length=base_hop_len,
                            win_length=n_fft_curr, window=torch.hann_window(n_fft_curr).to(device),
                            center=True, return_complex=True
                        )
                        mag_curr = torch.abs(stft_curr)
                        if base_win_len > n_fft_curr:
                            key_freq = f"mat_{win_ms}_{hop_ms}"
                            W_curr = getattr(self, key_freq)
                            mag_proj = torch.einsum('bot,oi->bit', mag_curr, W_curr)
                        else:
                            mag_proj = mag_curr

                        log_mag = torch.log(mag_proj + 1e-8)
                        log_sum = log_mag if log_sum is None else log_sum + log_mag

                    global_mask = torch.exp(log_sum / num_res)

                elif mask_type == '2_extreme':
                    # [Method 2] 2-Extreme Mask (Max Win & Min Win)
                    max_win_ms, max_hop_ms = self.config.target_resolutions[0]
                    min_win_ms, min_hop_ms = self.config.target_resolutions[-1]
                    
                    n_fft_long = int(self.config.sr * max_win_ms / 1000)
                    stft_long = torch.stft(
                        raw_audio.squeeze(1), n_fft=n_fft_long, hop_length=base_hop_len, win_length=n_fft_long,
                        window=torch.hann_window(n_fft_long).to(device), center=True, return_complex=True
                    )
                    mag_long = torch.abs(stft_long)
                    
                    n_fft_short = int(self.config.sr * min_win_ms / 1000)
                    stft_short = torch.stft(
                        raw_audio.squeeze(1), n_fft=n_fft_short, hop_length=base_hop_len, win_length=n_fft_short,
                        window=torch.hann_window(n_fft_short).to(device), center=True, return_complex=True
                    )
                    mag_short = torch.abs(stft_short)
                    
                    key_freq = f"mat_{min_win_ms}_{min_hop_ms}"
                    W_short = getattr(self, key_freq)
                    mag_short_proj = torch.einsum('bot,oi->bit', mag_short, W_short) 
                    
                    global_mask = torch.sqrt(mag_long * mag_short_proj + 1e-8)

                # Normalize and apply clamp for both mask types
                max_val, _ = global_mask.flatten(1).max(dim=1)
                global_mask = global_mask / (max_val.view(-1, 1, 1) + 1e-8)
                global_mask = torch.clamp(global_mask, min=0.05)
                # print(f"Global Mask Created with type '{mask_type}' and shape {global_mask.shape}")
        # =====================================================================

        if verbose:
            print(f"\n[Loss Debug] Base Complex Shape: {pred_super_complex.shape}, Mask Type: {mask_type}")

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
            curr_mask = global_mask 
            
            # 2. Freq Downsampling
            if base_win_len > target_win_len:
                key_freq = f"mat_{win_ms}_{hop_ms}"
                W = getattr(self, key_freq)
                curr_pred = torch.einsum('bitc,oi->botc', curr_pred, W)
                
                if curr_mask is not None:
                    curr_mask = torch.einsum('bit,oi->bot', curr_mask, W)

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
                    if curr_mask is not None:
                        curr_mask = F.pad(curr_mask, (0, pad_amount), mode='constant', value=0.05)

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
                
                if curr_mask is not None:
                    curr_mask_unfolded = curr_mask.view(B, n_freq, n_blocks, time_factor)
                    curr_mask = curr_mask_unfolded.mean(dim=3)
                
            # 4. Loss Mask Assignment
            assert curr_pred.shape == gt_view.shape, f"Shape mismatch: Pred {curr_pred.shape}, GT {gt_view.shape}"
            min_f = min(curr_pred.shape[1], gt_view.shape[1])
            min_t = min(curr_pred.shape[2], gt_view.shape[2])
            
            pred_crop = curr_pred[:, :min_f, :min_t, :]
            gt_crop = gt_view[:, :min_f, :min_t, :]
            
            if curr_mask is not None:
                mask_crop = curr_mask[:, :min_f, :min_t]
                mask_crop = mask_crop / (mask_crop.mean() + 1e-8)
            else:
                mask_crop = 1.0 # Broadcasting allows seamless unweighted operation
            
            # 5. Loss Calculation
            loss_real_unreduced = F.mse_loss(pred_crop[..., 0], gt_crop[..., 0], reduction='none')
            loss_real = (loss_real_unreduced * mask_crop).mean()
            
            loss_imag_unreduced = F.mse_loss(pred_crop[..., 1], gt_crop[..., 1], reduction='none')
            loss_imag = (loss_imag_unreduced * mask_crop).mean()

            pred_mag = torch.norm(pred_crop, dim=-1)
            gt_mag = torch.norm(gt_crop, dim=-1)
            # pred_mag = torch.clamp(pred_mag, min=1e-6)
            # gt_mag = torch.clamp(gt_mag, min=1e-6)
            pred_mag_compressed = torch.log1p(pred_mag * 10.0)
            gt_mag_compressed = torch.log1p(gt_mag * 10.0)
            loss_mag_unreduced = F.mse_loss(pred_mag_compressed, gt_mag_compressed, reduction='none')
            loss_mag = (loss_mag_unreduced * mask_crop).mean()
            
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


class HybridSupervisionLoss(MultiResConsistencyLoss):
    """
    Hybrid Supervision paradigm:
      - Magnitude (Target-Driven):    single loss against the 6-resolution min-intersection target.
      - Real/Imag (Consistency-Driven): multi-resolution loop (existing downsampling + mask), magnitude term removed.
    """

    def compute_intersection(self, raw_audio, verbose=False):
        """
        Compute the multi-resolution intersection target and cache the per-resolution STFTs.

        For each resolution (win_ms, hop_ms):
          1. STFT at native hop → complex tensor cached for Step 3 reuse
          2. Freq projection: nearest-neighbor → F_base
          3. Time projection: nearest-neighbor upsample → T_base
          4. Scale normalization: align all resolutions' max to the middle resolution
          5. Element-wise min across normalized stack → intersection_mag

        Returns:
          intersection_mag  : (B, F_base, T_base)
          mag_stack_normed  : list of (B, F_base, T_base), one per resolution (normalized)
          cached_gt_stfts   : dict {(win_ms, hop_ms): complex STFT tensor}
          scale_tensors     : list of (B,) tensors, one per resolution
        """
        device = raw_audio.device
        base_win_len = self.config.base_win_len
        n_freq_base = base_win_len // 2 + 1

        cached_gt_stfts = {}
        mag_projs = []
        T_base = None

        for win_ms, hop_ms in self.config.target_resolutions:
            n_fft_curr = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)

            stft_curr = torch.stft(
                raw_audio.squeeze(1), n_fft=n_fft_curr, hop_length=target_hop_len,
                win_length=n_fft_curr, window=torch.hann_window(n_fft_curr).to(device),
                center=True, return_complex=True
            )
            cached_gt_stfts[(win_ms, hop_ms)] = stft_curr

            mag_curr = torch.abs(stft_curr)  # (B, F_curr, T_native)

            # Freq projection: nearest-neighbor → F_base
            if base_win_len > n_fft_curr:
                n_freq_coarse = mag_curr.shape[1]
                fine_idx = torch.arange(n_freq_base, device=device)
                coarse_idx = (fine_idx * (n_freq_coarse - 1) / (n_freq_base - 1)).round().long()
                coarse_idx = coarse_idx.clamp(0, n_freq_coarse - 1)
                mag_proj = mag_curr[:, coarse_idx, :]
            else:
                mag_proj = mag_curr

            T_native = mag_proj.shape[2]
            if T_base is None or T_native > T_base:
                T_base = T_native
            mag_projs.append(mag_proj)

            if verbose:
                print(f"  [Intersection] Win={win_ms:2d}ms Hop={hop_ms:2d}ms | "
                      f"STFT (B,F={n_fft_curr//2+1},T={T_native}) → freq-proj (B,{n_freq_base},{T_native}) | "
                      f"max={mag_curr.max():.4f}  mean={mag_curr.mean():.4f}")

        # Time projection: nearest-neighbor upsample → T_base
        mag_stack_aligned = []
        for (win_ms, hop_ms), mag_proj in zip(self.config.target_resolutions, mag_projs):
            T_native = mag_proj.shape[2]
            if T_native < T_base:
                base_idx = torch.arange(T_base, device=device)
                native_idx = (base_idx * (T_native - 1) / (T_base - 1)).round().long()
                native_idx = native_idx.clamp(0, T_native - 1)
                mag_aligned = mag_proj[:, :, native_idx]
            else:
                mag_aligned = mag_proj
            mag_stack_aligned.append(mag_aligned)
            if verbose:
                print(f"  [Intersection] Win={win_ms}ms | time-proj: {T_native} → {T_base}")

        # Scale normalization: align all resolutions' max to the middle resolution
        n_res = len(mag_stack_aligned)
        ref_idx = n_res // 2
        ref_max = mag_stack_aligned[ref_idx].flatten(1).max(dim=1).values  # (B,)
        mag_stack_normed = []
        scale_tensors = []   # (B,) tensor per resolution, for applying to complex GT in Step 3
        scale_factors = []   # scalar for logging
        for mag_aligned in mag_stack_aligned:
            curr_max = mag_aligned.flatten(1).max(dim=1).values
            scale = ref_max / (curr_max + 1e-8)  # (B,)
            mag_stack_normed.append(mag_aligned * scale.view(-1, 1, 1))
            scale_tensors.append(scale)
            scale_factors.append(scale.mean().item())

        intersection_mag = torch.stack(mag_stack_normed, dim=0).min(dim=0).values  # (B, F_base, T_base)

        if verbose:
            ref_win = self.config.target_resolutions[ref_idx][0]
            print(f"\n  [Intersection] Scale norm ref=Win{ref_win}ms: "
                  + "  ".join(f"Win{r[0]}ms×{s:.3f}" for r, s in
                               zip(self.config.target_resolutions, scale_factors)))
            print(f"  [Intersection] result: {intersection_mag.shape}  "
                  f"max={intersection_mag.max():.4f}  mean={intersection_mag.mean():.4f}")

        return intersection_mag, mag_stack_normed, cached_gt_stfts, scale_tensors

    def forward(self, pred_super_complex, raw_audio, verbose=False):
        total_loss = 0.0
        details = {}
        device = pred_super_complex.device

        base_win_len = self.config.base_win_len
        base_hop_len = self.config.base_hop_len
        n_freq_base = base_win_len // 2 + 1

        if verbose:
            print(f"\n[HybridLoss] pred_super_complex: {pred_super_complex.shape}  "
                  f"(B, F_base={n_freq_base}, T_enc, 2)")

        # =====================================================================
        # Step 1: Compute intersection target + cache GT STFTs for Step 3.
        # =====================================================================
        with torch.no_grad():
            intersection_mag, _, cached_gt_stfts, scale_tensors = \
                self.compute_intersection(raw_audio, verbose=verbose)

            max_val, _ = intersection_mag.flatten(1).max(dim=1)
            global_mask = intersection_mag / (max_val.view(-1, 1, 1) + 1e-8)
            global_mask = torch.clamp(global_mask, min=0.05)

        if verbose:
            print(f"  [Step1] global_mask: min={global_mask.min():.3f}  max={global_mask.max():.3f}")

        # =====================================================================
        # Step 2: Magnitude Target Loss (single, base resolution)
        # =====================================================================
        pred_mag = torch.norm(pred_super_complex, dim=-1)  # (B, F_base, T_enc)
        min_f = min(pred_mag.shape[1], intersection_mag.shape[1])
        min_t = min(pred_mag.shape[2], intersection_mag.shape[2])

        loss_mag_target = F.mse_loss(
            torch.log1p(pred_mag[:, :min_f, :min_t] * 10.0),
            torch.log1p(intersection_mag[:, :min_f, :min_t] * 10.0),
        )
        total_loss += loss_mag_target * self.config.mag_weight
        details['loss_mag'] = loss_mag_target.item()

        if verbose:
            print(f"\n  [Step2] pred_mag: {pred_mag.shape}  crop → (B, {min_f}, {min_t})")
            print(f"  [Step2] Loss Mag Target: {loss_mag_target.item():.5f}")

        # =====================================================================
        # Step 3: Complex Consistency Loss (real + imag only, no mag term).
        # =====================================================================
        if verbose:
            print(f"\n  [Step3] Complex consistency loop:")

        _real_acc = 0.0
        _imag_acc = 0.0
        _res_w_sum = 0.0

        for i, (win_ms, hop_ms) in enumerate(self.config.target_resolutions):
            target_win_len = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)

            # Apply the same scale as magnitude normalization: Re/Im scale identically with mag.
            gt_complex_tensor = cached_gt_stfts[(win_ms, hop_ms)] * scale_tensors[i].view(-1, 1, 1)
            gt_view = torch.view_as_real(gt_complex_tensor)        # (B, F_curr, T_native, 2)

            curr_pred = pred_super_complex
            curr_mask = global_mask

            # Freq downsampling (triangular filterbank, same as parent class)
            if base_win_len > target_win_len:
                key_freq = f"mat_{win_ms}_{hop_ms}"
                W = getattr(self, key_freq)
                curr_pred = torch.einsum('bitc,oi->botc', curr_pred, W)
                curr_mask = torch.einsum('bit,oi->bot', curr_mask, W)

            # Time downsampling (phase-aware weighted average, same as parent class)
            time_factor = max(1, target_hop_len // base_hop_len)
            if time_factor > 1:
                curr_pred_complex = torch.view_as_complex(curr_pred.contiguous())
                B, n_freq, n_time = curr_pred_complex.shape

                remainder = n_time % time_factor
                if remainder != 0:
                    pad_amount = time_factor - remainder
                    curr_pred_complex = F.pad(curr_pred_complex, (0, pad_amount), value=0)
                    curr_mask = F.pad(curr_mask, (0, pad_amount), value=0.05)

                n_blocks = curr_pred_complex.shape[-1] // time_factor
                curr_pred_unfolded = curr_pred_complex.view(B, n_freq, n_blocks, time_factor)

                freqs = torch.fft.rfftfreq(target_win_len, d=1.0 / self.config.sr).to(device)
                t_indices = torch.arange(time_factor).to(device)
                base_hop_sec = base_hop_len / self.config.sr
                theta = -2 * torch.pi * freqs.unsqueeze(1) * t_indices.unsqueeze(0) * base_hop_sec
                rot_complex = torch.polar(torch.ones_like(theta), theta)
                kernel = self.kernel_win[str(time_factor)]
                weighted_spec = curr_pred_unfolded * rot_complex.unsqueeze(0).unsqueeze(2) * kernel.view(1, 1, 1, -1)

                curr_pred = torch.view_as_real(weighted_spec.mean(dim=3))
                curr_mask = curr_mask.view(B, n_freq, n_blocks, time_factor).mean(dim=3)

            if verbose:
                print(f"   [{i}] Win={win_ms}ms Hop={hop_ms}ms T_factor={time_factor} | "
                      f"curr_pred: {curr_pred.shape}  gt_view: {gt_view.shape}")

            min_f = min(curr_pred.shape[1], gt_view.shape[1])
            min_t = min(curr_pred.shape[2], gt_view.shape[2])
            pred_crop = curr_pred[:, :min_f, :min_t, :]
            gt_crop = gt_view[:, :min_f, :min_t, :]
            mask_crop = curr_mask[:, :min_f, :min_t]
            mask_crop = mask_crop / (mask_crop.mean() + 1e-8)

            loss_real = (F.mse_loss(pred_crop[..., 0], gt_crop[..., 0], reduction='none') * mask_crop).mean()
            loss_imag = (F.mse_loss(pred_crop[..., 1], gt_crop[..., 1], reduction='none') * mask_crop).mean()

            res_w = self.config.resolution_weights[i]
            current_loss = (loss_real * self.config.real_weight + loss_imag * self.config.imag_weight) * res_w
            total_loss += current_loss
            _real_acc += loss_real.item() * res_w
            _imag_acc += loss_imag.item() * res_w
            _res_w_sum += res_w

            if verbose:
                print(f"        crop → (B, {min_f}, {min_t}) | Loss Mag: {loss_mag_target.item():.5f}  "
                      f"Loss Real: {loss_real.item():.5f}  Imag: {loss_imag.item():.5f}  "
                      f"Weighted: {current_loss.item():.5f}")

        details['loss_real'] = _real_acc / (_res_w_sum + 1e-8)
        details['loss_imag'] = _imag_acc / (_res_w_sum + 1e-8)

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


def compute_entropy_loss(pred_complex):
    mag = torch.norm(pred_complex, dim=-1)
    p = mag / (torch.sum(mag, dim=(1, 2), keepdim=True) + 1e-8)
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=(1, 2))
    return entropy.mean()

def compute_tv_loss(pred_complex):
    mag = torch.norm(pred_complex, dim=-1)
    tv_f = torch.abs(mag[:, 1:, :] - mag[:, :-1, :]).mean()
    tv_t = torch.abs(mag[:, :, 1:] - mag[:, :, :-1]).mean()
    return tv_f + tv_t


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
    mrc_loss = HybridSupervisionLoss(cfg).to(device)
    gompsnr_loss = GOMPSNRLoss(cfg).to(device)
    sisnr_loss = SISNRLoss().to(device)
    
    # 3. Data
    # rand_len = int(38791 * 1.0)
    # x = torch.randn(1, 1, rand_len).to(device)
    audio_data, sr = sf.read("../egs2/vctk_noisy/enh1/dump/raw/tt_2spk/data/wav/format.1/p232_001.flac", dtype='float32')
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    x = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).to(device)
    rand_len = x.shape[-1]
    print(f"\nInput Audio: {x.shape}")
    
    # 4. Forward & Backward
    z_complex = encoder(x)
    entropy_loss = compute_entropy_loss(z_complex)
    tv_loss = compute_tv_loss(z_complex)
    multi_loss, multi_details = mrc_loss(z_complex, x, verbose=True)
    x_recon = decoder(z_complex, torch.tensor([rand_len]))
    recon_loss = F.mse_loss(x, x_recon)
    gompsnr_loss, gompsnr_details = gompsnr_loss(x_recon, x)
    sisnr_loss = sisnr_loss(x_recon, x)

    print(f"Output audio: {x_recon.shape}")
    
    # Aggregate
    total_loss = multi_loss * cfg.multi_res_weight + recon_loss * cfg.recon_weight + gompsnr_loss * cfg.gompsnr_weight + sisnr_loss * cfg.sisnr_weight + entropy_loss * cfg.entropy_weight + tv_loss * cfg.tv_weight
    
    print(f"\nTotal Loss: {total_loss.item():.6f}")
    print(f"  MultiRes: {multi_loss.item():.6f}")
    print(f"  Recon:    {recon_loss.item():.6f}")
    print(f"  GOMPSNR:  {gompsnr_loss.item():.6f}")
    print(f"  SI-SNR:   {sisnr_loss.item():.6f}")
    print(f"  Entropy:  {entropy_loss.item():.6f}")
    print(f"  TV:       {tv_loss.item():.6f}")

    optim = torch.optim.Adam(list(encoder.parameters()) + 
                             list(decoder.parameters()) + 
                             list(mrc_loss.parameters()), lr=1e-3)
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    print("\nBackprop successful.")