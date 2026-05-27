import os
import sys
import re
import shutil
import yaml
import random
import time
import matplotlib
matplotlib.use('Agg')
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
    HybridSupervisionLoss,
    MultiScalePhaseAuxLoss,
    GOMPSNRLoss,
    SISNRLoss,
)


# ─── Metric key registry ────────────────────────────────────────────────────
METRIC_KEYS = [
    'total',
    'multi_res', 'loss_mag', 'loss_real', 'loss_imag',   # H-loss breakdown
    'phase', 'ipl', 'iafl', 'gdl',                        # P-loss breakdown
    'gompsnr', 'sisnr', 'recon',
]

# ─── Color palette ───────────────────────────────────────────────────────────
_CLR = {
    # H-loss family (warm)
    'multi_res': '#E67E22',
    'loss_mag':  '#C0392B',
    'loss_real': '#E74C3C',
    'loss_imag': '#F39C12',
    # P-loss family (cool)
    'phase': '#1A237E',
    'ipl':   '#2980B9',
    'iafl':  '#3498DB',
    'gdl':   '#1ABC9C',
    # Scalars
    'total':   '#2C3E50',
    'gompsnr': '#8E44AD',
    'sisnr':   '#27AE60',
    'recon':   '#16A085',
}


# ─── Logging ─────────────────────────────────────────────────────────────────
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout          # keep reference to original stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ─── Dataset ─────────────────────────────────────────────────────────────────
class VCTKDataset(Dataset):
    def __init__(self, scp_path, sample_rate=16000, segment_length=1.0, is_train=True):
        self.data_list = []
        if not os.path.exists(scp_path):
            print(f"[Warning] SCP file not found: {scp_path}")
        else:
            with open(scp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
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
        if not os.path.exists(wav_path):
            path_head = "/exp/xuanyu.wang/espnet_20250624/egs2/vctk_noisy/enh1"
            wav_path = os.path.join(path_head, wav_path)
        try:
            audio, sr = sf.read(wav_path, dtype='float32')
            if sr != self.sr:
                raise ValueError(f"SR mismatch: {sr} vs {self.sr}")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if audio.shape[0] >= self.seg_len_samples:
                start = (random.randint(0, audio.shape[0] - self.seg_len_samples)
                         if self.is_train else 0)
                audio = audio[start: start + self.seg_len_samples]
            else:
                audio = np.pad(audio, (0, self.seg_len_samples - audio.shape[0]))
            return torch.from_numpy(audio).unsqueeze(0)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(1, self.seg_len_samples)


# ─── Utilities ───────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_meter():
    return {k: 0.0 for k in METRIC_KEYS}


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_history(history, save_dir, config):
    """
    2×3 layout:
      Row 0: Total (2 lines) | H-loss 1+3 (8 lines) | P-loss 1+3 (8 lines)
      Row 1: GOMP (2 lines)  | SI-SNR (2 lines)      | Recon (2 lines)

    Unweighted values for all component panels; outer weights shown in labels.
    Solid = train, dashed = val (same colour per metric).
    """
    os.makedirs(save_dir, exist_ok=True)

    tr, va = history['train'], history['valid']
    n_ep = len(tr.get('total', []))
    if n_ep == 0:
        return
    epochs = list(range(1, n_ep + 1))

    # Fetch weights for labels
    mw     = getattr(config, 'multi_res_weight', 300)
    pw     = getattr(config, 'phase_weight',     0.0)
    gw     = getattr(config, 'gompsnr_weight',   100)
    rw     = getattr(config, 'recon_weight',     100)
    sw     = getattr(config, 'sisnr_weight',     0.0)
    mag_w  = getattr(config, 'mag_weight',       0.33)
    real_w = getattr(config, 'real_weight',      0.33)
    imag_w = getattr(config, 'imag_weight',      0.33)
    ip_w   = getattr(config, 'w_ip', 0.2)
    gd_w   = getattr(config, 'w_gd', 1.0)
    if_w   = getattr(config, 'w_if', 1.0)

    def draw(ax, key, label, lw_main=1.8):
        """Plot train (solid) and val (dashed, same colour, no separate legend)."""
        color  = _CLR[key]
        t_vals = tr.get(key, [])[:n_ep]
        v_vals = va.get(key, [])[:n_ep]
        ep_t   = epochs[:len(t_vals)]
        ep_v   = epochs[:len(v_vals)]
        if t_vals:
            ax.plot(ep_t, t_vals, color=color, lw=lw_main,
                    solid_capstyle='round', label=label)
        if v_vals:
            ax.plot(ep_v, v_vals, color=color, lw=max(1.0, lw_main * 0.65),
                    ls='--', alpha=0.65, label=f'{label} ·val')

    def style(ax, title, ncol=1):
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss (unweighted)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.85,
                  ncol=ncol, handlelength=1.8, columnspacing=1.0)
        ax.text(0.01, 0.01, 'solid = train  ╌╌ = val',
                transform=ax.transAxes, fontsize=6, color='#7f8c8d', va='bottom')

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle('Training History', fontsize=13, fontweight='bold', y=1.005)

    # ── [0,0] Total ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    draw(ax, 'total', 'Total', lw_main=2.4)
    style(ax, 'Total Loss (weighted sum)', ncol=1)

    # ── [0,1] H-loss (total + mag + real + imag) ─────────────────────────────
    ax = axes[0, 1]
    draw(ax, 'multi_res', f'H-total  ×{mw:.0f}',         lw_main=2.2)
    draw(ax, 'loss_mag',  f'mag   (w={mag_w:.2f})',        lw_main=1.6)
    draw(ax, 'loss_real', f'real  (w={real_w:.2f})',       lw_main=1.6)
    draw(ax, 'loss_imag', f'imag  (w={imag_w:.2f})',       lw_main=1.6)
    style(ax, f'H-loss Components  [outer ×{mw:.0f}]', ncol=2)

    # ── [0,2] P-loss (total + ipl + iafl + gdl) ─────────────────────────────
    ax = axes[0, 2]
    if pw > 0:
        draw(ax, 'phase', f'P-total  ×{pw:.0f}',          lw_main=2.2)
        draw(ax, 'ipl',   f'IP    (w={ip_w:.1f})',           lw_main=1.6)
        draw(ax, 'iafl',  f'IF    (w={if_w:.1f})',           lw_main=1.6)
        draw(ax, 'gdl',   f'GD    (w={gd_w:.1f})',           lw_main=1.6)
        style(ax, f'P-loss Components  [outer ×{pw:.0f}]', ncol=2)
    else:
        ax.text(0.5, 0.5, 'Phase loss disabled\n(phase_weight = 0)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#95a5a6', style='italic')
        style(ax, 'P-loss Components  [disabled]', ncol=1)

    # ── [1,0] GOMP ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    draw(ax, 'gompsnr', f'GOMP  ×{gw:.0f}', lw_main=1.8)
    style(ax, f'GOMPSNR Loss  [weight ×{gw:.0f}]', ncol=1)

    # ── [1,1] SI-SNR ─────────────────────────────────────────────────────────
    ax = axes[1, 1]
    draw(ax, 'sisnr', f'SI-SNR  ×{sw:.1f}', lw_main=1.8)
    style(ax, f'SI-SNR Loss  [weight ×{sw:.1f}]', ncol=1)

    # ── [1,2] Recon ──────────────────────────────────────────────────────────
    ax = axes[1, 2]
    draw(ax, 'recon', f'Recon  ×{rw:.0f}', lw_main=1.8)
    style(ax, f'Reconstruction MSE  [weight ×{rw:.0f}]', ncol=1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Training / Validation ───────────────────────────────────────────────────
def _run_one_batch(x, models, criterions, config, device, is_train):
    encoder, decoder = models['enc'], models['dec']
    crit_multi   = criterions['multi']
    crit_gompsnr = criterions['gompsnr']
    crit_sisnr   = criterions['sisnr']
    crit_phase   = criterions.get('phase')

    z = encoder(x)                          # (B, F, T_enc, 2)
    x_recon = decoder(z)                    # (B, 1, T)

    multi_loss, multi_det = crit_multi(z, x)
    recon_loss             = F.mse_loss(x, x_recon)
    gomp_loss, _           = crit_gompsnr(x_recon, x)
    sisnr_loss             = crit_sisnr(x_recon, x)

    phase_loss = z.new_zeros(())
    phase_det  = {}
    if crit_phase is not None and config.phase_weight > 0:
        phase_loss, phase_det = crit_phase(z, x)

    total = (
        multi_loss  * config.multi_res_weight +
        recon_loss  * config.recon_weight     +
        gomp_loss   * config.gompsnr_weight   +
        sisnr_loss  * config.sisnr_weight     +
        phase_loss  * config.phase_weight
    )

    # Summarize per-scale phase detail into a single avg IP/GD/IF
    p_ip = sum(v['ip']  for v in phase_det.values()) / max(len(phase_det), 1)
    p_gd = sum(v['gd']  for v in phase_det.values()) / max(len(phase_det), 1)
    p_if = sum(v['if_'] for v in phase_det.values()) / max(len(phase_det), 1)

    meter = {
        'total':     total.item(),
        'multi_res': multi_loss.item(),
        'loss_mag':  multi_det.get('loss_mag',  0.0),
        'loss_real': multi_det.get('loss_real', 0.0),
        'loss_imag': multi_det.get('loss_imag', 0.0),
        'phase':     phase_loss.item(),
        'ipl':       p_ip,
        'iafl':      p_if,
        'gdl':       p_gd,
        'gompsnr':   gomp_loss.item(),
        'sisnr':     sisnr_loss.item(),
        'recon':     recon_loss.item(),
    }
    return total, meter


def train_one_epoch(models, criterions, optimizer, dataloader, config, device, epoch):
    encoder, decoder = models['enc'], models['dec']
    criterions['multi'].train()
    if criterions.get('phase') is not None:
        criterions['phase'].train()
    encoder.train(); decoder.train()

    accum   = _new_meter()
    n_batch = len(dataloader)

    for batch_idx, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        total, meter = _run_one_batch(x, models, criterions, config, device, is_train=True)

        total.backward()
        _phase = criterions.get('phase')
        clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()) +
            list(criterions['multi'].parameters()) +
            (list(_phase.parameters()) if _phase is not None else []),
            config.grad_clip,
        )
        optimizer.step()

        for k in METRIC_KEYS:
            accum[k] += meter[k]

        if batch_idx % config.log_interval == 0:
            print(
                f"Ep{epoch:3d} [{batch_idx:4d}/{n_batch}] "
                f"total={meter['total']:.4f} | "
                f"H={meter['multi_res']:.3f}"
                f"(mag={meter['loss_mag']:.3f} re={meter['loss_real']:.3f} im={meter['loss_imag']:.3f}) | "
                f"P={meter['phase']:.3f}"
                f"(ipl={meter['ipl']:.3f} iafl={meter['iafl']:.3f} gdl={meter['gdl']:.3f}) | "
                f"G={meter['gompsnr']:.3f} S={meter['sisnr']:.3f} R={meter['recon']:.5f}"
            )

    return {k: v / n_batch for k, v in accum.items()}


def validate(models, criterions, dataloader, config, device):
    encoder, decoder = models['enc'], models['dec']
    criterions['multi'].eval()
    if criterions.get('phase') is not None:
        criterions['phase'].eval()
    encoder.eval(); decoder.eval()

    accum   = _new_meter()
    n_batch = len(dataloader)

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            _, meter = _run_one_batch(x, models, criterions, config, device, is_train=False)
            for k in METRIC_KEYS:
                accum[k] += meter[k]

    return {k: v / n_batch for k, v in accum.items()}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = STFTConfig(cfg_dict=config_dict)
    for k, v in config_dict.get('train', {}).items():
        setattr(config, k, v)
    config.dataset        = config_dict.get('dataset', {})
    config.sample_rate    = config_dict.get('audio', {}).get('sample_rate', 16000)
    config.segment_length = config_dict.get('audio', {}).get('segment_length', 1.0)

    # Phase loss params (nested under loss: → phase_weight:)
    phase_cfg = config_dict.get('loss', {}).get('phase_weight', {})
    config.phase_weight = phase_cfg.get('weight', 0.0)
    config.w_ip         = phase_cfg.get('w_ip', 0.2)
    config.w_gd         = phase_cfg.get('w_gd', 1.0)
    config.w_if         = phase_cfg.get('w_if', 1.0)

    # Directories & logging
    os.makedirs(config.exp_dir, exist_ok=True)
    image_dir = os.path.join(config.exp_dir, "image")
    os.makedirs(image_dir, exist_ok=True)

    sys.stdout = TeeLogger(os.path.join(config.exp_dir, "train.log"))
    sys.stderr = sys.stdout

    src = os.path.abspath(yaml_path)
    dst = os.path.abspath(os.path.join(config.exp_dir, "config.yaml"))
    if src != dst:
        shutil.copy(yaml_path, dst)

    print("=" * 100)
    print(f"Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment: {config.exp_dir}")
    print(f"Phase loss: weight={config.phase_weight}  w_ip={config.w_ip}  "
          f"w_gd={config.w_gd}  w_if={config.w_if}")
    print(f"Config    : {config}")
    print("=" * 100)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_loader = DataLoader(
        VCTKDataset(config.dataset['train_scp'], config.sample_rate, config.segment_length, True),
        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        VCTKDataset(config.dataset['valid_scp'], config.sample_rate, config.segment_length, False),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        pin_memory=True,
    )

    # Models & losses
    encoder    = SuperResEncoder(config).to(device)
    decoder    = SuperResDecoder(config=config).to(device)
    crit_multi = (HybridSupervisionLoss(config) if config.mask_type == 'none'
                  else MultiResConsistencyLoss(config)).to(device)
    crit_gomp  = GOMPSNRLoss(config).to(device)
    crit_sisnr = SISNRLoss().to(device)
    crit_phase = (
        MultiScalePhaseAuxLoss(
            config,
            w_ip=getattr(config, 'w_ip', 0.2),
            w_gd=getattr(config, 'w_gd', 1.0),
            w_if=getattr(config, 'w_if', 1.0),
        ).to(device)
        if config.phase_weight > 0 else None
    )
    print(f"Using: {'HybridSupervisionLoss' if config.mask_type == 'none' else 'MultiResConsistencyLoss'} | "
          f"PhaseConsistencyLoss={'ON (w=' + str(config.phase_weight) + ')' if crit_phase else 'OFF'}")

    models     = {'enc': encoder, 'dec': decoder}
    criterions = {'multi': crit_multi, 'gompsnr': crit_gomp,
                  'sisnr': crit_sisnr, 'phase': crit_phase}

    phase_params = list(crit_phase.parameters()) if crit_phase is not None else []
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(crit_multi.parameters()) + phase_params,
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_factor,
        patience=config.lr_patience, min_lr=config.min_lr,
    )

    history = {split: {k: [] for k in METRIC_KEYS} for split in ('train', 'valid')}
    best_loss            = float('inf')
    best_checkpoint_path = None
    last_checkpoint_path = None
    start_epoch          = 1

    # Resume from checkpoint
    ckpt_list = sorted(
        [f for f in os.listdir(config.exp_dir) if f.endswith('epoch.pth')],
        key=lambda f: int(re.findall(r'\d+', f)[0]),
    )
    if ckpt_list:
        resume_path = os.path.abspath(os.path.join(config.exp_dir, ckpt_list[-1]))
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        crit_multi.load_state_dict(ckpt['loss_multi'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt['best_loss']
        # Merge history keys gracefully (new keys start empty if checkpoint predates them)
        saved_hist = ckpt.get('history', {})
        for split in ('train', 'valid'):
            for k in METRIC_KEYS:
                history[split][k] = saved_hist.get(split, {}).get(k, [])
        last_checkpoint_path = resume_path
        saved_best = ckpt.get('best_checkpoint_path')
        best_checkpoint_path = (
            os.path.abspath(saved_best) if saved_best
            else os.path.abspath(os.path.join(config.exp_dir, ckpt_list[0]))
        )
        print(f"  Resumed at epoch {start_epoch - 1} | "
              f"LR={optimizer.param_groups[0]['lr']:.2e} | best_val={best_loss:.5f}")
        print(f"  Last={os.path.basename(last_checkpoint_path)} | "
              f"Best={os.path.basename(best_checkpoint_path)}")

    # Training loop
    print("\nStart Training...")
    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()

        train_res = train_one_epoch(models, criterions, optimizer, train_loader, config, device, epoch)
        valid_res = validate(models, criterions, valid_loader, config, device)

        for k in METRIC_KEYS:
            history['train'][k].append(train_res[k])
            history['valid'][k].append(valid_res[k])

        plot_history(history, image_dir, config)
        scheduler.step(valid_res['total'])
        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d} | LR={lr_now:.2e} | "
            f"Train={train_res['total']:.5f}  Valid={valid_res['total']:.5f} | "
            f"H(tr/va)={train_res['multi_res']:.4f}/{valid_res['multi_res']:.4f}  "
            f"P(tr/va)={train_res['phase']:.4f}/{valid_res['phase']:.4f}  "
            f"G={valid_res['gompsnr']:.4f}  S={valid_res['sisnr']:.4f}  "
            f"R={valid_res['recon']:.5f}  | {time.time() - t0:.1f}s"
        )

        current_path = os.path.abspath(os.path.join(config.exp_dir, f"{epoch}epoch.pth"))

        # Remove previous non-best checkpoint
        if last_checkpoint_path and last_checkpoint_path != best_checkpoint_path:
            if os.path.exists(last_checkpoint_path):
                os.remove(last_checkpoint_path)
        last_checkpoint_path = current_path

        # Update best
        if valid_res['total'] < best_loss:
            print(f"  >>> New best: {best_loss:.5f} → {valid_res['total']:.5f}")
            if (best_checkpoint_path and os.path.exists(best_checkpoint_path)
                    and best_checkpoint_path != current_path):
                os.remove(best_checkpoint_path)
            best_loss            = valid_res['total']
            best_checkpoint_path = current_path

        torch.save({
            'epoch': epoch, 'history': history, 'best_loss': best_loss,
            'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
            'loss_multi': crit_multi.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_checkpoint_path': best_checkpoint_path,
        }, current_path)
        print("-" * 80)

    # Finalise
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        shutil.copy(best_checkpoint_path, os.path.join(config.exp_dir, "best_valid_loss.pth"))
    print("Training complete.")


if __name__ == "__main__":
    main()
