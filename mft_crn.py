import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Part 1: Backbone (保持不变，省略内部细节以节省篇幅，请确保包含之前的 MFT_CRN_Backbone)
# ============================================================================
class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.time_pad = kernel_size[0] - 1
    def forward(self, x):
        if self.time_pad > 0: x = F.pad(x, (0, 0, self.time_pad, 0))
        return self.conv(x)

class CausalDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=0)
    def forward(self, x): return self.deconv(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_deconv=False):
        super().__init__()
        self.conv = CausalDeconv2d(in_channels, out_channels, kernel_size, stride) if is_deconv else CausalConv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
    def forward(self, x): return self.elu(self.bn(self.conv(x)))

class MFTEncoderPath(nn.Module):
    def __init__(self, input_shape, config_list):
        super().__init__()
        layers = []
        current_in_ch = input_shape[0]
        for cfg in config_list:
            layers.append(BasicBlock(current_in_ch, cfg['out_ch'], kernel_size=(3, 2), stride=(2, 1)))
            current_in_ch = cfg['out_ch']
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class MFT_CRN_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mft_320 = MFTEncoderPath((1, None, 161), [{'out_ch': 8}])
        self.mft_160 = MFTEncoderPath((1, None, 81),  [{'out_ch': 8}, {'out_ch': 16}])
        self.mft_80  = MFTEncoderPath((1, None, 41),  [{'out_ch': 8}, {'out_ch': 16}, {'out_ch': 32}])
        self.mft_40  = MFTEncoderPath((1, None, 21),  [{'out_ch': 4}, {'out_ch': 8}, {'out_ch': 16}, {'out_ch': 32}])
        self.mft_20  = MFTEncoderPath((1, None, 11),  [{'out_ch': 2}, {'out_ch': 4}, {'out_ch': 8}, {'out_ch': 16}, {'out_ch': 32}])
        self.enc1 = BasicBlock(1, 8, (2, 3), (1, 2))
        self.enc2 = BasicBlock(16, 16, (2, 3), (1, 2))
        self.enc3 = BasicBlock(32, 32, (2, 3), (1, 2))
        self.enc4 = BasicBlock(64, 64, (2, 3), (1, 2))
        self.enc5 = BasicBlock(96, 128, (2, 3), (1, 2))
        self.enc6 = BasicBlock(160, 256, (2, 3), (1, 2))
        self.lstm = nn.LSTM(1024, 1024, 2, batch_first=True)
        self.dec6 = BasicBlock(512, 128, (2, 3), (1, 2), True)
        self.dec5 = BasicBlock(288, 64, (2, 3), (1, 2), True)
        self.dec4 = BasicBlock(160, 32, (2, 3), (1, 2), True)
        self.dec3 = BasicBlock(96, 16, (2, 3), (1, 2), True)
        self.dec2 = BasicBlock(48, 8, (2, 3), (1, 2), True)
        self.dec1 = CausalDeconv2d(24, 1, (2, 3), (1, 2))

    def _align_feature(self, src, tgt):
        df, dt = tgt.shape[3]-src.shape[3], tgt.shape[2]-src.shape[2]
        pad = [0, max(0, df), 0, max(0, dt)]
        if df < 0: src = src[..., :tgt.shape[3]]
        if dt < 0: src = src[..., :tgt.shape[2], :]
        return F.pad(src, pad)

    def forward(self, main, mfts):
        f320, f160, f80, f40, f20 = self.mft_320(mfts[0]), self.mft_160(mfts[1]), self.mft_80(mfts[2]), self.mft_40(mfts[3]), self.mft_20(mfts[4])
        e1 = self.enc1(main)
        cat1 = torch.cat([e1, self._align_feature(f320, e1)], 1)
        e2 = self.enc2(cat1)
        cat2 = torch.cat([e2, self._align_feature(f160, e2)], 1)
        e3 = self.enc3(cat2)
        cat3 = torch.cat([e3, self._align_feature(f80, e3)], 1)
        e4 = self.enc4(cat3)
        cat4 = torch.cat([e4, self._align_feature(f40, e4)], 1)
        e5 = self.enc5(cat4)
        cat5 = torch.cat([e5, self._align_feature(f20, e5)], 1)
        e6 = self.enc6(cat5)
        B, C, T, F = e6.shape
        lstm_out, _ = self.lstm(e6.permute(0, 2, 1, 3).reshape(B, T, C*F))
        d6 = self.dec6(torch.cat([lstm_out.reshape(B, T, C, F).permute(0, 2, 1, 3), e6], 1))
        d5 = self.dec5(torch.cat([self._align_feature(d6, cat5), cat5], 1))
        d4 = self.dec4(torch.cat([self._align_feature(d5, cat4), cat4], 1))
        d3 = self.dec3(torch.cat([self._align_feature(d4, cat3), cat3], 1))
        d2 = self.dec2(torch.cat([self._align_feature(d3, cat2), cat2], 1))
        out = self.dec1(torch.cat([self._align_feature(d2, cat1), cat1], 1))
        return self._align_feature(out, main)

# ============================================================================
# Part 2: 端到端模型 (修改返回值)
# ============================================================================

class MFT_CRN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MFT_CRN_Backbone()
        self.stft_configs = [(640, 320), (320, 160), (160, 80), (80, 40), (40, 20), (20, 10)]
        for i, (win, _) in enumerate(self.stft_configs):
            self.register_buffer(f'window_{win}', torch.hamming_window(win))

    def forward(self, waveform):
        if waveform.dim() == 3: waveform = waveform.squeeze(1)
        mags, main_phase = [], None
        main_length = waveform.shape[-1]
        
        for i, (win, hop) in enumerate(self.stft_configs):
            window = getattr(self, f'window_{win}')
            spec = torch.stft(waveform, win, hop, win, window, center=True, return_complex=True)
            mag = torch.abs(spec)
            mags.append(mag.permute(0, 2, 1).unsqueeze(1))
            if i == 0: main_phase = torch.angle(spec)

        # 网络预测幅度谱 (B, 1, T, F)
        pred_mag_network = self.backbone(mags[0], mags[1:])
        
        # 波形重构
        pred_mag_sq = pred_mag_network.squeeze(1).permute(0, 2, 1) # (B, F, T)
        pred_mag_sq = F.relu(pred_mag_sq) # 保证非负
        
        win_main, hop_main = self.stft_configs[0]
        recon_waveform = torch.istft(
            torch.polar(pred_mag_sq, main_phase), 
            win_main, hop_main, win_main, getattr(self, f'window_{win_main}'), 
            center=True, length=main_length
        )
        
        # 返回: (重建波形, 预测的幅度谱)
        return recon_waveform, pred_mag_network

# ============================================================================
# Part 3: 损失函数 (论文 2.4 节)
# ============================================================================

class MFT_CRN_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # 论文指定: 仅使用 640 窗口的幅度谱作为 Target
        self.win_length = 640
        self.hop_length = 320
        self.n_fft = 640
        self.register_buffer('window', torch.hamming_window(self.win_length))

    def forward(self, pred_mag_network, clean_waveform):
        """
        Args:
            pred_mag_network: (B, 1, T, F) - 模型的直接输出
            clean_waveform: (B, Samples)   - 纯净语音真值
        """
        if clean_waveform.dim() == 3: clean_waveform = clean_waveform.squeeze(1)
        
        # 1. 计算 Clean Speech 的 STFT 幅度谱
        target_spec = torch.stft(
            clean_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True
        ) # (B, F, T)
        
        target_mag = torch.abs(target_spec)
        
        # 2. 调整 Target 维度以匹配网络输出: (B, F, T) -> (B, 1, T, F)
        target_mag = target_mag.permute(0, 2, 1).unsqueeze(1)
        
        # 3. 对齐时间维度 (Handle padding differences if any)
        # 通常 torch.stft 和网络输出是对齐的，但为了鲁棒性，取最小值裁剪
        min_t = min(pred_mag_network.shape[2], target_mag.shape[2])
        pred_crop = pred_mag_network[:, :, :min_t, :]
        target_crop = target_mag[:, :, :min_t, :]
        
        # 4. MSE Loss (Formula 1 in paper)
        # L = || |X~| - |S| ||^2
        loss = F.mse_loss(pred_crop, target_crop)
        
        return loss

# ============================================================================
# Part 4: 运行示例
# ============================================================================
if __name__ == "__main__":
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 初始化
    model = MFT_CRN_Model().to(device)
    # print(f"Model structure:\n{model}")
    criterion = MFT_CRN_Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 2. 模拟数据
    batch_size, samples = 2, 16000 # 1秒
    noisy_audio = torch.randn(batch_size, samples).to(device)
    clean_audio = torch.randn(batch_size, samples).to(device) # GT
    
    print("Training Step Start...")
    
    # 3. 前向传播
    # 注意: 获取第二个返回值 pred_mag 用于计算 Loss
    est_audio, pred_mag = model(noisy_audio)
    
    # 4. 计算 Loss
    # 输入: (网络预测的幅度谱, 纯净音频波形)
    loss = criterion(pred_mag, clean_audio)
    
    print(f"Loss: {loss.item()}")
    
    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Backward pass successful.")
    print(f"Enhanced Audio Shape: {est_audio.shape}")