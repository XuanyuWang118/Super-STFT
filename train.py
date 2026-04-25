import os
import sys
import re
import shutil
import yaml
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from super_res_spec import (
    STFTConfig, 
    SuperResEncoder, 
    SuperResDecoder, 
    MultiResConsistencyLoss, 
    GOMPSNRLoss, 
    SISNRLoss,
    compute_entropy_loss,
    compute_tv_loss
)

class TeeLogger(object):
    """同时将输出打印到屏幕和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.terminal = sys.stderr
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class VCTKDataset(Dataset):
    def __init__(self, scp_path, sample_rate=16000, segment_length=1.0, is_train=True):
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
                        self.data_list.append(parts[1])
        
        self.sr = sample_rate
        self.seg_len_samples = int(sample_rate * segment_length)
        self.is_train = is_train
        print(f"Loaded {len(self.data_list)} files from {scp_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav_path = self.data_list[idx]
        if os.path.exists(wav_path) is False:
            # 兼容你的路径映射逻辑
            path_head = "/exp/xuanyu.wang/espnet_20250624/egs2/vctk_noisy/enh1"
            wav_path = os.path.join(path_head, wav_path)
        try:
            audio, sr = sf.read(wav_path, dtype='float32')
            if sr != self.sr:
                raise ValueError(f"SR mismatch: {sr} vs {self.sr}")
            if audio.ndim > 1:
                audio = audio[:, 0]

            if audio.shape[0] >= self.seg_len_samples:
                start = random.randint(0, audio.shape[0] - self.seg_len_samples) if self.is_train else 0
                audio_seg = audio[start : start + self.seg_len_samples]
            else:
                audio_seg = np.pad(audio, (0, self.seg_len_samples - audio.shape[0]), mode='constant')

            return torch.from_numpy(audio_seg).unsqueeze(0)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(1, self.seg_len_samples)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'sans-serif'
    metrics = ['total', 'multi_res', 'recon', 'gompsnr', 'sisnr', 'entropy', 'tv']
    plt.figure(figsize=(15, 4 * ((len(metrics)+1)//2)))
    nrows = (len(metrics) + 1) // 2
    for i, metric in enumerate(metrics):
        plt.subplot(nrows, 2, i+1)
        train_vals = history['train'].get(metric, [])
        valid_vals = history['valid'].get(metric, [])
        epochs = range(1, len(train_vals) + 1)
        plt.plot(epochs, train_vals, color='red', label='Train')
        if valid_vals:
            plt.plot(range(1, len(valid_vals) + 1), valid_vals, color='blue', label='Valid', linestyle='--')
        plt.title(f'{metric.upper()} Loss')
        plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)
    plt.close()


def train_one_epoch(models, criterions, optimizer, dataloader, config, device, epoch):
    encoder, decoder = models['enc'], models['dec']
    crit_multi, crit_gompsnr, crit_sisnr = criterions['multi'], criterions['gompsnr'], criterions['sisnr']
    
    encoder.train(); decoder.train(); crit_multi.train() 
    loss_meter = {k: 0.0 for k in ['total', 'multi_res', 'recon', 'gompsnr', 'sisnr', 'entropy', 'tv']}
    
    for batch_idx, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        
        z_complex = encoder(x)
        x_recon = decoder(z_complex)
        # 新增：entropy 和 tv 在 encoder 输出的复数特征上计算
        entropy_loss_raw = compute_entropy_loss(z_complex)
        tv_loss_raw = compute_tv_loss(z_complex)
        
        # 计算原始 Loss
        multi_loss_raw, _ = crit_multi(z_complex, x)
        recon_loss_raw = F.mse_loss(x, x_recon)
        gompsnr_loss_raw, _ = crit_gompsnr(x_recon, x)
        sisnr_loss_raw = crit_sisnr(x_recon, x) 

        # 使用 config 对象进行外部加权
        loss_total = (multi_loss_raw * config.multi_res_weight + 
                      recon_loss_raw * config.recon_weight + 
                      gompsnr_loss_raw * config.gompsnr_weight +
                      sisnr_loss_raw * config.sisnr_weight +
                      entropy_loss_raw * config.entropy_weight +
                      tv_loss_raw * config.tv_weight)
        
        loss_total.backward()
        clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters())+list(crit_multi.parameters()), config.grad_clip)
        optimizer.step()
        
        # 记录
        loss_meter['total'] += loss_total.item()
        loss_meter['multi_res'] += multi_loss_raw.item()
        loss_meter['recon'] += recon_loss_raw.item()
        loss_meter['gompsnr'] += gompsnr_loss_raw.item()
        loss_meter['sisnr'] += sisnr_loss_raw.item()
        loss_meter['entropy'] += entropy_loss_raw.item()
        loss_meter['tv'] += tv_loss_raw.item()
        
        if batch_idx % config.log_interval == 0:
            print(f"Epoch: {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss_total.item():.4f} "
                  f"(M:{multi_loss_raw.item():.2f} R:{recon_loss_raw.item():.2f} G:{gompsnr_loss_raw.item():.2f} S:{sisnr_loss_raw.item():.2f} E:{entropy_loss_raw.item():.2f} T:{tv_loss_raw.item():.2f})")
            
    return {k: v / len(dataloader) for k, v in loss_meter.items()}


def validate(models, criterions, dataloader, config, device):
    encoder, decoder = models['enc'], models['dec']
    crit_multi, crit_gompsnr, crit_sisnr = criterions['multi'], criterions['gompsnr'], criterions['sisnr']
    encoder.eval(); decoder.eval(); crit_multi.eval()
    
    loss_meter = {k: 0.0 for k in ['total', 'multi_res', 'recon', 'gompsnr', 'sisnr', 'entropy', 'tv']}
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            z_complex = encoder(x)
            x_recon = decoder(z_complex)
            entropy_loss_raw = compute_entropy_loss(z_complex)
            tv_loss_raw = compute_tv_loss(z_complex)

            multi_loss_raw, _ = crit_multi(z_complex, x)
            recon_loss_raw = F.mse_loss(x, x_recon)
            gompsnr_loss_raw, _ = crit_gompsnr(x_recon, x)
            sisnr_loss_raw = crit_sisnr(x_recon, x)

            loss_total = (multi_loss_raw * config.multi_res_weight + 
                          recon_loss_raw * config.recon_weight + 
                          gompsnr_loss_raw * config.gompsnr_weight +
                          sisnr_loss_raw * config.sisnr_weight +
                          entropy_loss_raw * config.entropy_weight +
                          tv_loss_raw * config.tv_weight)
            
            loss_meter['total'] += loss_total.item()
            loss_meter['multi_res'] += multi_loss_raw.item()
            loss_meter['recon'] += recon_loss_raw.item()
            loss_meter['gompsnr'] += gompsnr_loss_raw.item()
            loss_meter['sisnr'] += sisnr_loss_raw.item()
            loss_meter['entropy'] += entropy_loss_raw.item()
            loss_meter['tv'] += tv_loss_raw.item()
            
    return {k: v / len(dataloader) for k, v in loss_meter.items()}


def main():
    # 1. 配置加载与重构
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 实例化 STFTConfig，并手动将 train 相关的字典参数注入，实现全属性访问
    config = STFTConfig(cfg_dict=config_dict)
    # 将 train 字典中的 key 变成 config 的属性
    for k, v in config_dict.get('train', {}).items():
        setattr(config, k, v)
    # 将 dataset 字典中的内容也挂载方便使用
    config.dataset = config_dict.get('dataset', {})
    # 将 audio 配置也挂载为属性，方便数据集使用
    config.sample_rate = config_dict.get('audio', {}).get('sample_rate', 16000)
    config.segment_length = config_dict.get('audio', {}).get('segment_length', 1.0)

    # 2. 目录准备与日志重定向
    os.makedirs(config.exp_dir, exist_ok=True)
    image_dir = os.path.join(config.exp_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    # 启用自动日志
    log_path = os.path.join(config.exp_dir, "train.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger  # 重定向标准输出
    sys.stderr = logger  # 重定向标准错误

    # 备份目录，并检查重复配置文件
    src_config_path = os.path.abspath(yaml_path)
    dst_config_path = os.path.abspath(os.path.join(config.exp_dir, "config.yaml"))

    if src_config_path != dst_config_path:
        shutil.copy(yaml_path, dst_config_path)
    # shutil.copy(yaml_path, os.path.join(config.exp_dir, "config.yaml"))
    
    print("="*100)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment Directory: {config.exp_dir}")
    print(f"Full Config: {config}")
    print("="*100)
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. DataLoaders
    train_loader = DataLoader(VCTKDataset(config.dataset['train_scp'],
                                          sample_rate=config.sample_rate,
                                          segment_length=config.segment_length,
                                          is_train=True),
                              batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(VCTKDataset(config.dataset['valid_scp'],
                                          sample_rate=config.sample_rate,
                                          segment_length=config.segment_length,
                                          is_train=False),
                              batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    # 4. Models & Criterions
    encoder = SuperResEncoder(config).to(device)
    decoder = SuperResDecoder(config).to(device)
    crit_multi = MultiResConsistencyLoss(config).to(device)
    crit_gompsnr = GOMPSNRLoss(config).to(device)
    crit_sisnr = SISNRLoss().to(device)
    
    models = {'enc': encoder, 'dec': decoder}
    criterions = {'multi': crit_multi, 'gompsnr': crit_gompsnr, 'sisnr': crit_sisnr}
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + 
                                 list(crit_multi.parameters()), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                            patience=config.lr_patience, min_lr=config.min_lr, verbose=True)
    
    history = {'train': {k: [] for k in ['total', 'multi_res', 'recon', 'gompsnr', 'sisnr', 'entropy', 'tv']},
               'valid': {k: [] for k in ['total', 'multi_res', 'recon', 'gompsnr', 'sisnr', 'entropy', 'tv']}}
    
    best_loss = float('inf')
    best_checkpoint_path = None
    last_checkpoint_path = None

    # 断点续传
    start_epoch = 1
    ckpt_list = [f for f in os.listdir(config.exp_dir) if f.endswith('epoch.pth')]
    if ckpt_list:
        # 排序找到最新的 epoch
        ckpt_list.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
        last_ckpt_name = ckpt_list[-1]
        # 统一使用绝对路径
        resume_path = os.path.abspath(os.path.join(config.exp_dir, last_ckpt_name))
        
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # 恢复权重和状态
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        crit_multi.load_state_dict(checkpoint['loss_multi'])
        if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])

        if 'optimizer' in checkpoint: print("Optimizer state loaded.")
        if 'scheduler' in checkpoint: print("Scheduler state loaded.")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"DEBUG: Resumed with Learning Rate: {current_lr}")
        
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_loss = checkpoint['best_loss']
        
        # 【核心修复】恢复路径指针并标准化
        last_checkpoint_path = resume_path
        saved_best_path = checkpoint.get('best_checkpoint_path')
        
        if saved_best_path:
            best_checkpoint_path = os.path.abspath(saved_best_path)
        else:
            # 如果是首次升级代码恢复，且当前文件就是最好的（通过 loss 判断）
            # 或者猜测第一个文件是最好的（之前的旧逻辑）
            if len(ckpt_list) > 0:
                best_checkpoint_path = os.path.abspath(os.path.join(config.exp_dir, ckpt_list[0]))
        
        print(f"Verified Pointers: Last -> {os.path.basename(last_checkpoint_path)}, Best -> {os.path.basename(best_checkpoint_path)}")
        
    # 5. Training Loop
    print("\nStart Training...")
    for epoch in range(start_epoch, config.epochs + 1):
        start_t = time.time()
        
        train_res = train_one_epoch(models, criterions, optimizer, train_loader, config, device, epoch)
        valid_res = validate(models, criterions, valid_loader, config, device)

        # 更新历史并绘图
        for k in history['train']:
            history['train'][k].append(train_res[k])
            history['valid'][k].append(valid_res[k])
        plot_history(history, image_dir)
        
        scheduler.step(valid_res['total'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | LR: {current_lr:.2e} | Train: {train_res['total']:.5f} | Valid: {valid_res['total']:.5f} | Time: {time.time()-start_t:.1f}s")
        
        current_path = os.path.abspath(os.path.join(config.exp_dir, f"{epoch}epoch.pth"))
        
        # 情况 A 的删除逻辑：如果上一轮不是最优，则删除
        # 使用 os.path.samefile (更安全) 或标准化路径比较
        if last_checkpoint_path and last_checkpoint_path != best_checkpoint_path:
            if os.path.exists(last_checkpoint_path):
                print(f"  >>> Removing old latest: {os.path.basename(last_checkpoint_path)}")
                os.remove(last_checkpoint_path)
        
        # 更新 Last 指针
        last_checkpoint_path = current_path
        
        # 情况 B 的更新逻辑
        is_new_best = False
        if valid_res['total'] < best_loss:
            print(f"  >>> New Best! Valid Loss: {best_loss:.5f} -> {valid_res['total']:.5f}")
            # 如果之前的 best 不是刚才被作为 last 删掉的文件，现在手动删除它
            if best_checkpoint_path and os.path.exists(best_checkpoint_path) and best_checkpoint_path != current_path:
                print(f"  >>> Removing old best: {os.path.basename(best_checkpoint_path)}")
                os.remove(best_checkpoint_path)
            
            best_loss = valid_res['total']
            best_checkpoint_path = current_path
            is_new_best = True

        # 保存当前模型
        save_dict = {
            'epoch': epoch, 'history': history, 'best_loss': best_loss,
            'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 
            'loss_multi': crit_multi.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_checkpoint_path': best_checkpoint_path, # 保存绝对路径
        }
        torch.save(save_dict, current_path)
            
        print("-" * 60)

    # 6. Finalize
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        shutil.copy(best_checkpoint_path, os.path.join(config.exp_dir, "best_valid_loss.pth"))
    print("Training Completed.")

if __name__ == "__main__":
    main()