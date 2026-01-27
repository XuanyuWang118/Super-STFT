import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import yaml
from typing import List, Tuple, Dict, Any

from gompsnr.phase_related_losses import WeightedOmniPhaseLoss, CoupledOmniRILoss

def get_dft_bases(n_fft, round_pow_of_two=True):
    N = 2 ** math.ceil(math.log2(n_fft)) if round_pow_of_two else n_fft
    delayed_delta = torch.eye(N)
    dft_bases = torch.view_as_real(torch.fft.fft(delayed_delta))
    return dft_bases


class STFTConfig:
    def __init__(self, cfg_dict: Dict[str, Any] = None, yaml_path: str = None):
        """
        初始化配置。可以通过字典传入，也可以通过 yaml 路径加载。
        """
        if yaml_path:
            with open(yaml_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
        
        if cfg_dict is None:
            # 默认兜底配置，防止空初始化报错
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
        self.multi_res_weight = loss_cfg.get('multi_res_weight', 1.0)
        
        # Resolution weights logic
        res_weights = loss_cfg.get('resolution_weights', [])
        if not res_weights:
            self.resolution_weights = [1.0] * len(self.target_resolutions)
        else:
            self.resolution_weights = res_weights
            assert len(self.resolution_weights) == len(self.target_resolutions), "Resolution weights length mismatch"

        self.recon_weight = loss_cfg.get('recon_weight', 1.0)
        
        self.gompsnr_weight = loss_cfg.get('gompsnr_weight', 1.0)
        self.wop_weight = loss_cfg.get('wop_weight', 1.0)
        self.cori_weight = loss_cfg.get('cori_weight', 1.0)
        self.wop_alpha = loss_cfg.get('wop_alpha', 100)
        self.mag_dist_type = loss_cfg.get('mag_dist_type', 'L1')

        # 5. Derived Params (Calculated)
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
            freq_factor = base_win_len // target_win_len
            
            if freq_factor > 1:
                curr_pred = curr_pred.permute(0, 2, 3, 1) 
                B_temp, T_temp, C_temp, F_in = curr_pred.shape 
                curr_pred = curr_pred.reshape(-1, C_temp, F_in) 

                target_n_freq = target_win_len // 2 + 1
                expected_len = target_n_freq * freq_factor
                pad_amt = expected_len - F_in
                
                if pad_amt > 0:
                    curr_pred = F.pad(curr_pred, (0, pad_amt), mode='replicate')
                
                curr_pred = F.avg_pool1d(curr_pred, kernel_size=freq_factor, stride=freq_factor)
                curr_pred = curr_pred.view(B_temp, T_temp, C_temp, -1).permute(0, 3, 1, 2)

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
            
            current_total_loss = loss_real + loss_imag + loss_mag
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
    criterion = MultiResConsistencyLoss(cfg).to(device)
    decoder_criterion = GOMPSNRLoss(cfg).to(device)
    
    # 3. Data
    rand_len = int(16000 * 1.0)
    x = torch.randn(1, 1, rand_len).to(device)
    print(f"\nInput Audio: {x.shape}")
    
    # 4. Forward & Backward
    z_complex = encoder(x)
    multi_loss, multi_details = criterion(z_complex, x, verbose=False)
    
    x_recon = decoder(z_complex)
    recon_loss = F.mse_loss(x, x_recon)
    
    gompsnr_loss, gompsnr_details = decoder_criterion(x_recon, x)
    
    # Aggregate
    total_loss = multi_loss * cfg.multi_res_weight + recon_loss * cfg.recon_weight + gompsnr_loss * cfg.gompsnr_weight
    
    print(f"\nTotal Loss: {total_loss.item():.6f}")
    print(f"  MultiRes: {multi_loss.item():.6f}")
    print(f"  Recon:    {recon_loss.item():.6f}")
    print(f"  GOMPSNR:  {gompsnr_loss.item():.6f}")

    optim = torch.optim.Adam(list(encoder.parameters()) + 
                             list(decoder.parameters()) + 
                             list(criterion.parameters()), lr=1e-3)
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    print("\nBackprop successful.")