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


def stft_left_aligned(signal, n_fft, hop_length, win_length, window, pad_amount):
    """左对齐 STFT：统一外部补 pad_amount 再 center=False，与 encoder 的 F.pad 约定一致。"""
    padded = F.pad(signal, (pad_amount, pad_amount))
    return torch.stft(padded, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                      window=window, center=False, return_complex=True)


def anti_wrap(x):
    """反缠绕：把相位差映射到 (-pi, pi]。f_aw(x) = x - 2pi*round(x/2pi)。"""
    return x - 2 * math.pi * torch.round(x / (2 * math.pi))


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

        self.sr = cfg_dict.get('audio', {}).get('sample_rate', 16000)
        self.segment_length = cfg_dict.get('audio', {}).get('segment_length', 1.0)

        stft_cfg = cfg_dict.get('stft', {})
        self.target_resolutions = stft_cfg.get('target_resolutions', [(64, 32)])
        self.super_hop_ms = stft_cfg.get('super_hop_ms', 1.0)

        model_cfg = cfg_dict.get('model', {})
        self.encoder_init = model_cfg.get('encoder_init', 'dft')
        self.decoder_init = model_cfg.get('decoder_init', 'idft')

        loss_cfg = cfg_dict.get('loss', {})

        self.mask_type = loss_cfg.get('mask_type', 'none')

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

        gompsnr_weight = loss_cfg.get('gompsnr_weight', {})
        self.gompsnr_weight = gompsnr_weight.get('weight', 1.0)
        w_w = gompsnr_weight.get('wop_weight', 1.0)
        c_w = gompsnr_weight.get('cori_weight', 1.0)
        w_sum_g = w_w + c_w + 1e-8
        self.wop_weight = w_w / w_sum_g
        self.cori_weight = c_w / w_sum_g
        self.wop_alpha = gompsnr_weight.get('wop_alpha', 100)
        self.mag_dist_type = gompsnr_weight.get('mag_dist_type', 'L1')

        self.recon_weight = loss_cfg.get('recon_weight', 1.0)
        self.sisnr_weight = loss_cfg.get('sisnr_weight', 0.0)

        self.max_win_ms = max([r[0] for r in self.target_resolutions])
        self.base_win_len = int(self.sr * self.max_win_ms / 1000)
        self.base_hop_len = int(self.sr * self.super_hop_ms / 1000)

        if self.base_win_len % 2 != 0:
            self.base_win_len += 1

    def __repr__(self):
        return (f"STFTConfig(sr={self.sr}, segment_length={self.segment_length}, "
                f"target_resolutions={self.target_resolutions}, super_hop_ms={self.super_hop_ms}, "
                f"encoder_init='{self.encoder_init}', decoder_init='{self.decoder_init}', "
                f"multi_res_weight={self.multi_res_weight}, resolution_weights={self.resolution_weights}, "
                f"real_weight={self.real_weight}, imag_weight={self.imag_weight}, mag_weight={self.mag_weight}, "
                f"gompsnr_weight={self.gompsnr_weight}, wop_weight={self.wop_weight}, cori_weight={self.cori_weight}, "
                f"wop_alpha={self.wop_alpha}, mag_dist_type='{self.mag_dist_type}', "
                f"recon_weight={self.recon_weight}, sisnr_weight={self.sisnr_weight}, "
                f"base_win_len={self.base_win_len}, base_hop_len={self.base_hop_len})")


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


class ComplexDownsampler(nn.Module):
    """
    把 high-res 复数谱 (B, T_hi, F_hi) 下采样到目标档 (B, T_lo, F_lo)。
    用 2->2 通道 Conv2d 在复数域学习最优下采样核，修复了实数三角窗破坏相位的问题。
    初始化为恒等（中心抽取），便于起步。
    """
    def __init__(self, time_stride, freq_stride, kt=3, kf=7):
        super().__init__()
        self.time_stride = time_stride
        self.freq_stride = freq_stride
        self.conv = nn.Conv2d(2, 2, kernel_size=(kt, kf),
                              stride=(time_stride, freq_stride),
                              padding=(kt // 2, kf // 2), bias=False)
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            self.conv.weight[0, 0, kt // 2, kf // 2] = 1.0
            self.conv.weight[1, 1, kt // 2, kf // 2] = 1.0

    def forward(self, Z):
        """Z: (B, T, F) 复 -> (B, T', F') 复。"""
        x = torch.stack([Z.real, Z.imag], dim=1)   # (B, 2, T, F)
        y = self.conv(x)                            # (B, 2, T', F')
        return torch.complex(y[:, 0], y[:, 1])      # (B, T', F')


class MultiResConsistencyLoss(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.kernel_win = nn.ParameterDict()

        for win_ms, hop_ms in self.config.target_resolutions:
            target_hop_len = int(self.config.sr * hop_ms / 1000)
            base_hop_len = self.config.base_hop_len
            time_factor = max(1, target_hop_len // base_hop_len)
            key = str(time_factor)
            if time_factor > 1 and key not in self.kernel_win:
                self.kernel_win[key] = nn.Parameter(torch.ones(time_factor))

            target_win_len = int(self.config.sr * win_ms / 1000)
            base_win_len = self.config.base_win_len
            if base_win_len > target_win_len:
                n_freq_in = base_win_len // 2 + 1
                n_freq_out = target_win_len // 2 + 1
                mat = create_triangular_filterbank(n_freq_in, n_freq_out, self.config.sr)
                self.register_buffer(f"mat_{win_ms}_{hop_ms}", mat)

    def forward(self, pred_super_complex, raw_audio, verbose=False):
        total_loss = 0.0
        details = {}
        device = pred_super_complex.device

        base_win_len = self.config.base_win_len
        base_hop_len = self.config.base_hop_len
        pad = base_win_len // 2

        if verbose:
            print(f"\n[MultiResLoss] Base Complex Shape: {pred_super_complex.shape}")

        for i, (win_ms, hop_ms) in enumerate(self.config.target_resolutions):
            target_win_len = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)

            # 1. GT STFT（左对齐，统一 pad=512）
            gt_complex_tensor = stft_left_aligned(
                raw_audio.squeeze(1), target_win_len, target_hop_len,
                target_win_len, torch.hann_window(target_win_len).to(device), pad
            )
            gt_view = torch.view_as_real(gt_complex_tensor)

            curr_pred = pred_super_complex

            # 2. 频率下采样（三角滤波器组）
            if base_win_len > target_win_len:
                W = getattr(self, f"mat_{win_ms}_{hop_ms}")
                curr_pred = torch.einsum('bitc,oi->botc', curr_pred, W)

            # 3. 时间下采样（带相位旋转补偿）
            time_factor = max(1, target_hop_len // base_hop_len)
            if time_factor > 1:
                curr_pred_complex = torch.view_as_complex(curr_pred.contiguous())
                B, n_freq, n_time = curr_pred_complex.shape

                remainder = n_time % time_factor
                if remainder != 0:
                    curr_pred_complex = F.pad(curr_pred_complex, (0, time_factor - remainder))

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

            # 4. Loss
            min_f = min(curr_pred.shape[1], gt_view.shape[1])
            min_t = min(curr_pred.shape[2], gt_view.shape[2])
            pred_crop = curr_pred[:, :min_f, :min_t, :]
            gt_crop = gt_view[:, :min_f, :min_t, :]

            loss_real = F.mse_loss(pred_crop[..., 0], gt_crop[..., 0])
            loss_imag = F.mse_loss(pred_crop[..., 1], gt_crop[..., 1])

            pred_mag = torch.norm(pred_crop, dim=-1).clamp(min=1e-6)
            gt_mag = torch.norm(gt_crop, dim=-1).clamp(min=1e-6)
            loss_mag = F.mse_loss(torch.log(pred_mag), torch.log(gt_mag))

            current_total_loss = (
                loss_real * self.config.real_weight +
                loss_imag * self.config.imag_weight +
                loss_mag * self.config.mag_weight
            ) * self.config.resolution_weights[i]

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
      - Real/Imag (Consistency-Driven): multi-resolution loop, magnitude term removed.
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
        pad = base_win_len // 2

        cached_gt_stfts = {}
        mag_projs = []
        T_base = None

        for win_ms, hop_ms in self.config.target_resolutions:
            n_fft_curr = int(self.config.sr * win_ms / 1000)
            target_hop_len = int(self.config.sr * hop_ms / 1000)

            stft_curr = stft_left_aligned(
                raw_audio.squeeze(1), n_fft_curr, target_hop_len,
                n_fft_curr, torch.hann_window(n_fft_curr).to(device), pad
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
                      f"STFT (B,F={n_fft_curr//2+1},T={T_native}) -> freq-proj (B,{n_freq_base},{T_native}) | "
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
                print(f"  [Intersection] Win={win_ms}ms | time-proj: {T_native} -> {T_base}")

        # Scale normalization: align all resolutions' max to the middle resolution
        n_res = len(mag_stack_aligned)
        ref_idx = n_res // 2
        ref_max = mag_stack_aligned[ref_idx].flatten(1).max(dim=1).values
        mag_stack_normed = []
        scale_tensors = []
        scale_factors = []
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

        # Step 1: intersection 目标 + 缓存 GT STFT
        with torch.no_grad():
            intersection_mag, _, cached_gt_stfts, scale_tensors = \
                self.compute_intersection(raw_audio, verbose=verbose)

            max_val, _ = intersection_mag.flatten(1).max(dim=1)
            global_mask = intersection_mag / (max_val.view(-1, 1, 1) + 1e-8)
            global_mask = torch.clamp(global_mask, min=0.05)

        # intersection 用最小 hop 的 STFT 计算，帧数可能多于 encoder 输出，截断对齐
        T_enc = pred_super_complex.shape[2]
        global_mask = global_mask[:, :, :T_enc]

        if verbose:
            print(f"  [Step1] global_mask: min={global_mask.min():.3f}  max={global_mask.max():.3f}")

        # Step 2: 幅度目标 loss
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
            print(f"\n  [Step2] Loss Mag Target: {loss_mag_target.item():.5f}")

        # Step 3: 复数一致性 loss（Re/Im）
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
                W = getattr(self, f"mat_{win_ms}_{hop_ms}")
                curr_pred = torch.einsum('bitc,oi->botc', curr_pred, W)
                curr_mask = torch.einsum('bit,oi->bot', curr_mask, W)

            # Time downsampling (phase-aware weighted average, same as parent class)
            time_factor = max(1, target_hop_len // base_hop_len)
            if time_factor > 1:
                curr_pred_complex = torch.view_as_complex(curr_pred.contiguous())
                B, n_freq, n_time = curr_pred_complex.shape

                remainder = n_time % time_factor
                if remainder != 0:
                    curr_pred_complex = F.pad(curr_pred_complex, (0, time_factor - remainder))
                    curr_mask = F.pad(curr_mask, (0, time_factor - remainder), value=0.05)

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

        details['loss_real'] = _real_acc / (_res_w_sum + 1e-8)
        details['loss_imag'] = _imag_acc / (_res_w_sum + 1e-8)

        return total_loss, details


class MultiScalePhaseAuxLoss(nn.Module):
    """
    多尺度相位辅助损失。

    编码器输出 (B, F, T, 2) 经 ComplexDownsampler 下采样到各档时频网格，
    在复数域对齐到该档真实 STFT 的相位谱，监督 GD（群延迟）、IF（瞬时频率）
    和 IP（瞬时相位）三项反缠绕损失。

    所有 GT STFT 使用统一 512 外部补零 + center=False，与 encoder 的
    F.pad(x, (512, 512)) 约定完全一致。
    """
    def __init__(self, config: STFTConfig,
                 scale_wins_ms=(1, 2, 4, 8, 16, 32),
                 win_name="hann",
                 w_ip=0.2, w_gd=1.0, w_if=1.0,
                 trainable_down=True):
        super().__init__()
        self.config = config
        self.win_name = win_name
        self.w_ip, self.w_gd, self.w_if = w_ip, w_gd, w_if
        self.pad_amount = config.base_win_len // 2   # = 512

        hires_hop = config.base_hop_len              # 16 samples
        hires_nfft = config.base_win_len             # 1024 samples

        self.scales = []
        downs = []
        for w_ms in scale_wins_ms:
            n_fft = int(round(w_ms * 1e-3 * config.sr))
            hop = n_fft // 2
            F_lo = n_fft // 2 + 1
            t_stride = max(1, hop // hires_hop)
            # 频率下采样率：窗长比（DC 对齐，物理频率精确对应）
            f_stride = hires_nfft // n_fft
            self.scales.append(dict(w_ms=w_ms, n_fft=n_fft, hop=hop,
                                    F_lo=F_lo, t_stride=t_stride, f_stride=f_stride))
            d = ComplexDownsampler(t_stride, f_stride)
            for p in d.parameters():
                p.requires_grad = trainable_down
            downs.append(d)
        self.downsamplers = nn.ModuleList(downs)

    @staticmethod
    def _gd(phase):
        """群延迟：沿频率轴差分，反缠绕。(B,T,F) -> (B,T,F-1)。"""
        return anti_wrap(phase[..., 1:] - phase[..., :-1])

    @staticmethod
    def _if(phase):
        """瞬时频率：沿时间轴差分，反缠绕。(B,T,F) -> (B,T-1,F)。"""
        return anti_wrap(phase[:, 1:, :] - phase[:, :-1, :])

    def _phase_terms_loss(self, pred_c, tgt_phase):
        pred_phase = torch.angle(pred_c)
        L_ip = anti_wrap(pred_phase - tgt_phase).abs().mean()
        L_gd = (self._gd(pred_phase) - self._gd(tgt_phase)).abs().mean()
        L_if = (self._if(pred_phase) - self._if(tgt_phase)).abs().mean()
        return L_ip, L_gd, L_if

    def forward(self, enc_ri, raw_audio, verbose=False):
        """
        enc_ri:    (B, F, T, 2)  encoder 输出
        raw_audio: (B, 1, T) 或 (B, T)  原始波形
        verbose:   打印每档维度和损失细节
        """
        # 转换 encoder 输出为 (B, T, F) 复数
        enc_c = torch.view_as_complex(enc_ri.contiguous())  # (B, F, T)
        Z_hires = enc_c.permute(0, 2, 1).contiguous()       # (B, T, F)

        wav = raw_audio.squeeze(1) if raw_audio.dim() == 3 else raw_audio

        if verbose:
            print(f"\n[MultiScalePhaseAuxLoss] enc_ri: {tuple(enc_ri.shape)}  "
                  f"Z_hires: {tuple(Z_hires.shape)}  wav: {tuple(wav.shape)}  "
                  f"pad={self.pad_amount}")

        details = {}
        total = enc_ri.new_zeros(())

        for cfg, down in zip(self.scales, self.downsamplers):
            # 1) 复数域下采样 encoder 输出到该档时频网格
            pred = down(Z_hires)                             # (B, T', F')

            # 2) GT STFT（统一 512 补零 + center=False，帧数若多于 pred 则在步骤 3 截断）
            with torch.no_grad():
                window = getattr(torch, f"{self.win_name}_window")(
                    cfg["n_fft"], periodic=True, device=wav.device, dtype=wav.dtype)
                S_tgt = stft_left_aligned(
                    wav, cfg["n_fft"], cfg["hop"], cfg["n_fft"], window, self.pad_amount
                ).transpose(1, 2).contiguous()              # (B, T_lo, F_lo)
                tgt_phase = torch.angle(S_tgt)

            # 3) 裁到公共 (T, F)
            T = min(pred.shape[1], S_tgt.shape[1])
            Fc = min(pred.shape[2], S_tgt.shape[2])
            pred = pred[:, :T, :Fc]
            tgt_phase = tgt_phase[:, :T, :Fc]

            # 4) 三项反缠绕相位损失
            L_ip, L_gd, L_if = self._phase_terms_loss(pred, tgt_phase)
            L = self.w_ip * L_ip + self.w_gd * L_gd + self.w_if * L_if
            total = total + L
            details[f"{cfg['w_ms']}ms"] = dict(
                ip=L_ip.item(), gd=L_gd.item(), if_=L_if.item(), total=L.item())

            if verbose:
                print(f"  [{cfg['w_ms']:>2}ms] n_fft={cfg['n_fft']:4d}  hop={cfg['hop']:4d}  "
                      f"t_stride={cfg['t_stride']}  f_stride={cfg['f_stride']}  "
                      f"pred:{tuple(pred.shape)}  gt:{tuple(tgt_phase.shape)}  "
                      f"ip={L_ip.item():.4f}  gd={L_gd.item():.4f}  if={L_if.item():.4f}  "
                      f"L={L.item():.4f}")

        if verbose:
            print(f"  [total] {total.item():.4f}  scales={list(details.keys())}")

        return total, details


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

        pad = self.n_fft // 2
        spec_recon = stft_left_aligned(x_recon, self.n_fft, self.hop_len,
                                       self.win_len, self.window, pad)
        spec_gt = stft_left_aligned(x_gt, self.n_fft, self.hop_len,
                                       self.win_len, self.window, pad)

        eps = 1e-6
        rea_g = spec_recon.real + eps
        imag_g = spec_recon.imag + eps
        rea = spec_gt.real + eps
        imag = spec_gt.imag + eps

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}\n")

    import os
    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        cfg = STFTConfig(yaml_path=config_path)
    else:
        print("Config file not found, using defaults")
        cfg = STFTConfig()

    print(f"Config: {cfg}")

    encoder = SuperResEncoder(cfg).to(device)
    decoder = SuperResDecoder(cfg).to(device)
    hybrid_loss = HybridSupervisionLoss(cfg).to(device)
    phase_loss = MultiScalePhaseAuxLoss(cfg).to(device)
    gompsnr_loss = GOMPSNRLoss(cfg).to(device)
    sisnr_loss = SISNRLoss().to(device)

    audio_data, sr = sf.read("../egs2/vctk_noisy/enh1/dump/raw/tt_2spk/data/wav/format.1/p232_001.flac", dtype='float32')
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    x = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).to(device)
    rand_len = x.shape[-1]
    print(f"\nInput Audio: {x.shape}")

    z_complex = encoder(x)

    multi_loss, multi_details = hybrid_loss(z_complex, x, verbose=True)
    phase_l, phase_details = phase_loss(z_complex, x, verbose=True)

    x_recon = decoder(z_complex, torch.tensor([rand_len]))
    recon_loss = F.mse_loss(x, x_recon)
    gompsnr_l, gompsnr_details = gompsnr_loss(x_recon, x)
    sisnr_l = sisnr_loss(x_recon, x)

    print(f"\nOutput audio: {x_recon.shape}")

    total_loss = (multi_loss * cfg.multi_res_weight +
                  recon_loss * cfg.recon_weight      +
                  gompsnr_l  * cfg.gompsnr_weight    +
                  sisnr_l    * cfg.sisnr_weight)

    print(f"\nTotal Loss: {total_loss.item():.6f}")
    print(f"  MultiRes: {multi_loss.item():.6f}")
    print(f"  Phase:    {phase_l.item():.6f}")
    print(f"  Recon:    {recon_loss.item():.6f}")
    print(f"  GOMPSNR:  {gompsnr_l.item():.6f}")
    print(f"  SI-SNR:   {sisnr_l.item():.6f}")

    optim = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(hybrid_loss.parameters()) +
        list(phase_loss.parameters()),
        lr=1e-3
    )
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    print("\nBackprop successful.")
