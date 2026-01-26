import os
import shutil
import yaml
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from super_res_spec import (
    STFTConfig, 
    SuperResEncoder, 
    SuperResDecoder, 
    MultiResConsistencyLoss, 
    GOMPSNRLoss
)

# 设置绘图风格
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. Dataset Class
# ==========================================
class VCTKDataset(Dataset):
    def __init__(self, scp_path, sample_rate=16000, segment_length=1.0, is_train=True):
        """
        读取 wav.scp 文件。
        Format: key /path/to/audio.wav
        """
        self.data_list = []
        if not os.path.exists(scp_path):
            print(f"[Warning] SCP file not found: {scp_path}")
        else:
            with open(scp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) >= 2:
                        self.data_list.append(parts[1]) # 只存路径
        
        self.sr = sample_rate
        self.seg_len_samples = int(sample_rate * segment_length)
        self.is_train = is_train
        print(f"Loaded {len(self.data_list)} files from {scp_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav_path = self.data_list[idx]
        if os.path.exists(wav_path) is False:
            path_head = "/exp/xuanyu.wang/espnet_20250624/egs2/vctk_noisy/enh1"
            wav_path = os.path.join(path_head, wav_path)
        # 读取音频
        try:
            # audio shape: (Time,)
            audio, sr = sf.read(wav_path, dtype='float32')
            
            # 重采样检查 (简单的 assert，实际工程可能需要 resample)
            if sr != self.sr:
                raise ValueError(f"Sample rate mismatch: expected {self.sr}, got {sr}")
            
            # 单声道检查
            if audio.ndim > 1:
                audio = audio[:, 0] # 取第一通道

            current_len = audio.shape[0]

            # 裁剪或填充逻辑
            if current_len >= self.seg_len_samples:
                if self.is_train:
                    # 随机裁剪
                    start = random.randint(0, current_len - self.seg_len_samples)
                else:
                    # 验证集中心裁剪或从头开始
                    start = 0
                audio_seg = audio[start : start + self.seg_len_samples]
            else:
                # 填充
                pad_len = self.seg_len_samples - current_len
                audio_seg = np.pad(audio, (0, pad_len), mode='constant')

            # 转为 Tensor (1, Time)
            return torch.from_numpy(audio_seg).unsqueeze(0)

        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(1, self.seg_len_samples)

# ==========================================
# 2. Utils
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_history(history, save_dir):
    """
    绘制 Loss 曲线：红(Train) 蓝(Valid)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # history 结构: {'train': {'total':[], 'multi':[], ...}, 'valid': {...}}
    metrics = ['total', 'multi_res', 'recon', 'gompsnr']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        train_vals = history['train'].get(metric, [])
        valid_vals = history['valid'].get(metric, [])
        epochs = range(1, len(train_vals) + 1)
        
        plt.plot(epochs, train_vals, color='red', label='Train', linewidth=1.5)
        if len(valid_vals) > 0:
            # 验证集通常比训练集少一个点或者对齐，这里做长度保护
            v_epochs = range(1, len(valid_vals) + 1)
            plt.plot(v_epochs, valid_vals, color='blue', label='Valid', linewidth=1.5, linestyle='--')
            
        plt.title(f'{metric.replace("_", " ").upper()} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)
    plt.close()

# ==========================================
# 3. Trainer
# ==========================================
def train_one_epoch(models, criterions, optimizer, dataloader, loss_cfg, train_cfg, device, epoch):
    encoder, decoder = models['enc'], models['dec']
    crit_multi, crit_gompsnr = criterions['multi'], criterions['gompsnr']
    
    encoder.train()
    decoder.train()
    crit_multi.train() 
    
    loss_meter = {'total': 0.0, 'multi_res': 0.0, 'recon': 0.0, 'gompsnr': 0.0}
    steps = 0
    
    for batch_idx, x in enumerate(dataloader):
        x = x.to(device) # (B, 1, T)
        
        optimizer.zero_grad()
        
        # 1. Forward
        z_complex = encoder(x)
        x_recon = decoder(z_complex)
        
        # 2. Loss Calculation (Raw Unweighted)
        multi_loss_raw, _ = crit_multi(z_complex, x)
        recon_loss_raw = F.mse_loss(x, x_recon)
        gompsnr_loss_raw, _ = crit_gompsnr(x_recon, x)
        
        # 3. Weighted Sum (使用 loss_cfg 获取权重)
        loss_total = (multi_loss_raw * loss_cfg.multi_res_weight + recon_loss_raw * loss_cfg.recon_weight + gompsnr_loss_raw * loss_cfg.gompsnr_weight)
        
        # 4. Backward
        loss_total.backward()
        
        # Gradient Clipping (使用 train_cfg 获取 grad_clip)
        clip_grad_norm_(encoder.parameters(), train_cfg.grad_clip)
        clip_grad_norm_(decoder.parameters(), train_cfg.grad_clip)
        clip_grad_norm_(crit_multi.parameters(), train_cfg.grad_clip)
        
        optimizer.step()
        
        # Record
        loss_meter['total'] += loss_total.item()
        loss_meter['multi_res'] += multi_loss_raw.item()
        loss_meter['recon'] += recon_loss_raw.item()
        loss_meter['gompsnr'] += gompsnr_loss_raw.item()
        steps += 1
        
        # Logging (使用 train_cfg 获取 log_interval)
        if batch_idx % train_cfg.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_total.item():.4f} "
                  f"(M:{multi_loss_raw.item():.4f} R:{recon_loss_raw.item():.4f} G:{gompsnr_loss_raw.item():.4f})")
            
    # Average
    for k in loss_meter:
        loss_meter[k] /= steps
        
    return loss_meter

def validate(models, criterions, dataloader, cfg, device):
    encoder, decoder = models['enc'], models['dec']
    crit_multi, crit_gompsnr = criterions['multi'], criterions['gompsnr']
    
    encoder.eval()
    decoder.eval()
    crit_multi.eval()
    
    loss_meter = {'total': 0.0, 'multi_res': 0.0, 'recon': 0.0, 'gompsnr': 0.0}
    steps = 0
    
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            
            z_complex = encoder(x)
            x_recon = decoder(z_complex)
            
            multi_loss_raw, _ = crit_multi(z_complex, x)
            recon_loss_raw = F.mse_loss(x, x_recon)
            gompsnr_loss_raw, _ = crit_gompsnr(x_recon, x)
            
            loss_total = (multi_loss_raw * cfg.multi_res_weight + recon_loss_raw * cfg.recon_weight + gompsnr_loss_raw * cfg.gompsnr_weight)
            
            loss_meter['total'] += loss_total.item()
            loss_meter['multi_res'] += multi_loss_raw.item()
            loss_meter['recon'] += recon_loss_raw.item()
            loss_meter['gompsnr'] += gompsnr_loss_raw.item()
            steps += 1
            
    for k in loss_meter:
        loss_meter[k] /= steps
        
    return loss_meter

# ==========================================
# 4. Main
# ==========================================
def main():
    # torch.autograd.set_detect_anomaly(True) 
    
    # 1. Load Config
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 实例化 STFTConfig 用于模型初始化
    stft_cfg = STFTConfig(cfg_dict=config_dict)
    
    # 获取各部分配置字典
    train_cfg = config_dict['train']
    loss_cfg = config_dict['loss']
    dataset_cfg = config_dict['dataset']
    audio_cfg = config_dict['audio']
    
    # 2. Setup Directories
    exp_dir = train_cfg['exp_dir']
    image_dir = os.path.join(exp_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    # 备份配置文件
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
    
    # 3. Print Info
    print("="*60)
    print(f"Experiment Dir: {exp_dir}")
    print(f"STFT Config: {stft_cfg}")
    print(f"Train Config: Epochs={train_cfg['epochs']}, BS={train_cfg['batch_size']}, LR={train_cfg['learning_rate']}")
    print("="*60)
    
    # 4. Initialization
    set_seed(train_cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataLoaders
    train_ds = VCTKDataset(dataset_cfg['train_scp'], sample_rate=audio_cfg['sample_rate'],
                           segment_length=audio_cfg['segment_length'], is_train=True)
    valid_ds = VCTKDataset(dataset_cfg['valid_scp'], sample_rate=audio_cfg['sample_rate'],
                           segment_length=audio_cfg['segment_length'], is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], 
                              shuffle=True, num_workers=train_cfg['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=train_cfg['batch_size'], 
                              shuffle=False, num_workers=train_cfg['num_workers'], pin_memory=True)
    
    # Models & Loss
    encoder = SuperResEncoder(stft_cfg).to(device)
    decoder = SuperResDecoder(stft_cfg).to(device)
    crit_multi = MultiResConsistencyLoss(stft_cfg).to(device)
    crit_gompsnr = GOMPSNRLoss(stft_cfg).to(device)
    
    models = {'enc': encoder, 'dec': decoder}
    criterions = {'multi': crit_multi, 'gompsnr': crit_gompsnr}
    
    # Optimizer
    all_params = list(encoder.parameters()) + \
                 list(decoder.parameters()) + \
                 list(crit_multi.parameters())
    optimizer = torch.optim.Adam(all_params, lr=train_cfg['learning_rate'])
    
    # History
    history = {
        'train': {'total':[], 'multi_res':[], 'recon':[], 'gompsnr':[]},
        'valid': {'total':[], 'multi_res':[], 'recon':[], 'gompsnr':[]}
    }
    
    # 5. Saving State Variables
    best_loss = float('inf')
    best_checkpoint_path = None # 记录历史最佳模型文件的路径 
    last_checkpoint_path = None # 记录上一轮模型文件的路径
    
    # 6. Training Loop
    print("\nStart Training...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        start_time = time.time()
        
        from types import SimpleNamespace
        loss_cfg_ns = SimpleNamespace(**loss_cfg)
        train_cfg_ns = SimpleNamespace(**train_cfg)

        train_metrics = train_one_epoch(models, criterions, optimizer, train_loader, loss_cfg_ns, train_cfg_ns, device, epoch)
        valid_metrics = validate(models, criterions, valid_loader, loss_cfg_ns, device)
        
        # Update History & Plot
        for k in history['train']:
            history['train'][k].append(train_metrics[k])
            history['valid'][k].append(valid_metrics[k])
        plot_history(history, image_dir)
        
        end_time = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              f"Epoch {epoch} Done ({end_time - start_time:.1f}s) | "
              f"Train: {train_metrics['total']:.5f} | "
              f"Valid: {valid_metrics['total']:.5f}")
        
        # ---------------- Saving Logic ----------------
        current_val_loss = valid_metrics['total']
        is_best = current_val_loss < best_loss
        
        save_dict = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'loss_multi': crit_multi.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
            'best_loss': best_loss if not is_best else current_val_loss
        }
        
        # 1. 保存当前轮次模型
        current_path = os.path.join(exp_dir, f"{epoch}epoch.pth")
        torch.save(save_dict, current_path)
        
        # 2. 删除上一轮的普通模型 (如果它不是历史最佳)
        if last_checkpoint_path and last_checkpoint_path != best_checkpoint_path:
            if os.path.exists(last_checkpoint_path):
                os.remove(last_checkpoint_path)
        
        # 更新 Last 指针
        last_checkpoint_path = current_path
        
        # 3. 处理最佳模型
        if is_best:
            print(f"  >>> New Best! Loss: {best_loss:.5f} -> {current_val_loss:.5f}")
            if best_checkpoint_path and os.path.exists(best_checkpoint_path) and best_checkpoint_path != current_path:
                os.remove(best_checkpoint_path)
            best_loss = current_val_loss
            best_checkpoint_path = current_path
            
        print("-" * 60)

    # 7. Finalize
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        final_best_path = os.path.join(exp_dir, "best_valid_loss.pth")
        shutil.copy(best_checkpoint_path, final_best_path)
        print(f"Training Finished. Best model copied to: {final_best_path}")
    else:
        print("Training Finished. (No best model found?)")

if __name__ == "__main__":
    main()