#!/usr/bin/env python3
"""
Fine-tune SuperRes encoder+decoder on RealMAN 32-mic dataset.

Objective: improve inter-channel phase coherence for beamforming.

Training procedure per sample:
  1. Random 2-N channels of clean speech (ma_speech, same room)
  2. Same channel indices from a noise recording (ma_noise, same room)
  3. Scale noise to random SNR, add to speech -> noisy mix
  4. Encode S=enc(speech), N=enc(noise), X=enc(mix)
  5. IRM(S,N) -> MVDR_Souden on X -> enhanced Y  (F, T_f)
  6. HybridSupervisionLoss on (S, speech), (N, noise), (X, mix)
  7. L1 loss: dec(Y) vs dec(S[ref_ch])
  8. Backprop enc + dec

Usage:
  python finetune_realman.py --config config.yaml
"""

import argparse
import json
import logging
import os
import re
import random
import shutil
import sys
import time
import traceback
import yaml
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espnet2.enh.layers.beamformer_th as bf
from super_res_spec import STFTConfig, SuperResEncoder, SuperResDecoder, HybridSupervisionLoss


# -----------------------------------------------------------------------
# Data indexing helpers
# -----------------------------------------------------------------------

def _scandir_group_channels(directory: str):
    """One os.scandir() call; returns dict: utt_base -> [(ch_num, path), ...]
    Avoids per-channel Path.exists() calls on slow network filesystems."""
    groups = defaultdict(list)
    with os.scandir(directory) as it:
        for entry in it:
            name = entry.name
            if not name.endswith(".flac") or "_CH" not in name:
                continue
            ch_pos = name.rfind("_CH")
            utt_base = name[:ch_pos]
            try:
                ch_num = int(name[ch_pos + 3:-5])
            except ValueError:
                continue
            groups[utt_base].append((ch_num, entry.path))
    return groups


def _index_speech(speech_root: Path):
    """Returns list of (room, [path_ch0, path_ch1, ...]) per utterance."""
    items = []
    for room_dir in sorted(speech_root.iterdir()):
        if not room_dir.is_dir() or room_dir.name.endswith(".rar"):
            continue
        room = room_dir.name
        for motion in ("static", "moving"):
            motion_dir = room_dir / motion
            if not motion_dir.exists():
                continue
            for spk_dir in sorted(motion_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                for utt_base, ch_list in _scandir_group_channels(str(spk_dir)).items():
                    ch_list.sort()
                    ch_files = [p for _, p in ch_list]
                    if len(ch_files) >= 2:
                        items.append((room, ch_files))
    return items


def _index_noise(noise_root: Path):
    """Returns dict: room -> list of [path_ch0, ...] per recording."""
    idx = defaultdict(list)
    for room_dir in sorted(noise_root.iterdir()):
        if not room_dir.is_dir() or room_dir.name.endswith(".rar"):
            continue
        room = room_dir.name
        for rec_base, ch_list in _scandir_group_channels(str(room_dir)).items():
            ch_list.sort()
            ch_files = [p for _, p in ch_list]
            if len(ch_files) >= 2:
                idx[room].append(ch_files)
    return idx


def _build_or_load_index(speech_root: Path, noise_root: Path, cache_path: str):
    """Load from JSON cache if available; otherwise build and save.
    Subsequent runs skip all filesystem scanning entirely."""
    if os.path.exists(cache_path):
        logger.info(f"Loading index cache: {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        items = [(d["room"], d["ch_files"]) for d in data["speech"]]
        noise_idx = defaultdict(list)
        for room, recs in data["noise"].items():
            noise_idx[room] = recs
        logger.info(f"  speech={len(items)}  noise_rooms={len(noise_idx)}")
        return items, noise_idx

    logger.info("Building index (first run — will be cached)...")
    t0 = time.time()
    items = _index_speech(speech_root)
    noise_idx = _index_noise(noise_root)
    logger.info(f"  Indexed {len(items)} utterances in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "speech": [{"room": r, "ch_files": chs} for r, chs in items],
            "noise":  {room: recs for room, recs in noise_idx.items()},
        }, f)
    logger.info(f"  Cache saved: {cache_path}")
    return items, noise_idx


def _load_channels(ch_files, ch_indices, seg_len: int, target_sr: int,
                   random_start: bool = True):
    """Load specific channel indices, reading only the needed segment.

    - sf.info() called ONCE to get file length (not per channel)
    - sf.read(start, stop) reads only seg_len frames (critical for long noise files)
    - All channels share the same start offset (temporal alignment preserved)
    Returns (C, seg_len) float32 ndarray.
    """
    info = sf.info(str(ch_files[ch_indices[0]]))
    file_sr = info.samplerate
    n_frames = info.frames

    frames_needed = int(np.ceil(seg_len * file_sr / target_sr)) if file_sr != target_sr else seg_len

    if n_frames >= frames_needed:
        start = random.randint(0, n_frames - frames_needed) if random_start else 0
        stop  = start + frames_needed
    else:
        start, stop = 0, n_frames

    waves = []
    for ci in ch_indices:
        audio, _ = sf.read(str(ch_files[ci]), start=start, stop=stop, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if file_sr != target_sr:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(audio_t, file_sr, target_sr).squeeze(0).numpy()
        if len(audio) >= seg_len:
            audio = audio[:seg_len]
        else:
            audio = np.pad(audio, (0, seg_len - len(audio)))
        waves.append(audio)
    return np.stack(waves, axis=0)  # (C, seg_len)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class RealMANDataset(Dataset):
    def __init__(self, items, noise_index, sr=16000, seg_len_s=2.0,
                 min_ch=2, max_ch=8, snr_min=-5.0, snr_max=20.0):
        self.items = items
        self.noise_index = noise_index
        self._all_noise = [v for vals in noise_index.values() for v in vals]
        self.sr = sr
        self.seg_len = int(sr * seg_len_s)
        self.min_ch = min_ch
        self.max_ch = max_ch
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        room, speech_chs = self.items[idx]

        # Pick noise first so we can constrain channel selection to indices
        # valid for BOTH speech and noise (preserves spatial correspondence)
        noise_recs = self.noise_index.get(room) or self._all_noise
        noise_chs = random.choice(noise_recs)

        n_common = min(len(speech_chs), len(noise_chs))
        n_sel = random.randint(min(self.min_ch, n_common), min(self.max_ch, n_common))
        ch_indices = sorted(random.sample(range(n_common), n_sel))

        try:
            speech_np = _load_channels(speech_chs, ch_indices, self.seg_len, self.sr)
        except Exception:
            return None

        try:
            noise_np = _load_channels(noise_chs, ch_indices, self.seg_len, self.sr)
        except Exception:
            return None

        # Scale noise to target SNR
        snr_db = random.uniform(self.snr_min, self.snr_max)
        sp_rms = np.sqrt(np.mean(speech_np ** 2) + 1e-10)
        no_rms = np.sqrt(np.mean(noise_np ** 2) + 1e-10)
        scale = sp_rms / no_rms * 10 ** (-snr_db / 20.0)
        noise_scaled = noise_np * scale
        mix_np = speech_np + noise_scaled

        return (
            torch.from_numpy(speech_np).float(),       # (C, T)
            torch.from_numpy(noise_scaled).float(),    # (C, T)
            torch.from_numpy(mix_np).float(),          # (C, T)
        )


def _collate(batch):
    return [b for b in batch if b is not None]


# -----------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------

def encode_mc(wav_mc: torch.Tensor, enc: SuperResEncoder) -> torch.Tensor:
    """(C, T) -> (C, F, T_f, 2)"""
    return enc(wav_mc.unsqueeze(1))


def irm_beamform(
    S: torch.Tensor,   # (C, F, T_f) complex
    N: torch.Tensor,   # (C, F, T_f) complex
    X: torch.Tensor,   # (C, F, T_f) complex
    ref_ch: int = 0,
    diag_eps: float = 1e-7,
) -> torch.Tensor:
    """IRM -> MVDR_Souden; returns (F, T_f) complex enhanced features."""
    ps = S.abs().pow(2)
    pn = N.abs().pow(2)
    denom = ps + pn + 1e-10
    ms = (ps / denom).permute(1, 0, 2)   # (F, C, T_f)
    mn = (pn / denom).permute(1, 0, 2)

    data = X.permute(1, 0, 2).unsqueeze(0).cdouble()
    stats = bf.prepare_beamformer_stats(
        data, [ms.unsqueeze(0).double()], mn.unsqueeze(0).double(),
        beamformer_type="mvdr_souden", bdelay=3, btaps=5, eps=1e-6,
    )
    C = S.shape[0]
    u = torch.zeros(1, C, dtype=torch.double, device=S.device)
    u[0, ref_ch] = 1.0
    w   = bf.get_mvdr_vector(stats["psd_speech"], stats["psd_n"], u,
                             diagonal_loading=True, diag_eps=diag_eps)
    out = bf.apply_beamforming_vector(w, data)
    return out.squeeze(0).to(torch.complex64)   # (F, T_f)


# -----------------------------------------------------------------------
# One training step
# -----------------------------------------------------------------------

def process_sample(enc, dec, hybrid_loss, sample, device, bf_weight: float):
    """Returns (loss, stft_loss, bf_loss) as scalar tensors."""
    speech_mc, noise_mc, mix_mc = [x.to(device) for x in sample]
    C, T = speech_mc.shape

    # Encode: (C, F, T_f, 2)
    S_ri = encode_mc(speech_mc, enc)
    N_ri = encode_mc(noise_mc,  enc)
    X_ri = encode_mc(mix_mc,    enc)

    # HybridSupervisionLoss on each signal independently.
    # Outer multi_res_weight (300×) from pretraining is intentionally not applied
    # here; fine-tuning uses the raw normalized internal loss.
    loss_S, _ = hybrid_loss(S_ri, speech_mc.unsqueeze(1))
    loss_N, _ = hybrid_loss(N_ri, noise_mc.unsqueeze(1))
    loss_X, _ = hybrid_loss(X_ri, mix_mc.unsqueeze(1))
    stft_loss = loss_S + loss_N + loss_X

    # Beamforming loss
    bf_loss = torch.tensor(0.0, device=device)
    if bf_weight > 0.0:
        try:
            S_c = torch.view_as_complex(S_ri.contiguous())
            N_c = torch.view_as_complex(N_ri.contiguous())
            X_c = torch.view_as_complex(X_ri.contiguous())
            with torch.no_grad():
                Y = irm_beamform(S_c, N_c, X_c, ref_ch=0)
            Y_ri     = torch.view_as_real(Y).unsqueeze(0)
            y_wav    = dec(Y_ri, target_len=T).squeeze()
            s_ri_ref = S_ri[0:1]
            s_wav    = dec(s_ri_ref, target_len=T).squeeze()
            bf_loss  = F.l1_loss(y_wav, s_wav)
        except Exception:
            logger.warning(f"    bf step failed: {traceback.format_exc()}")

    loss = stft_loss + bf_weight * bf_loss
    return loss, stft_loss, bf_loss


# -----------------------------------------------------------------------
# Epoch loop
# -----------------------------------------------------------------------

def run_epoch(enc, dec, hybrid_loss, optimizer, samples, device,
              is_train, grad_clip, bf_weight, log_interval=200,
              total_samples=None, max_steps=None):
    enc.train(is_train); dec.train(is_train)

    totals = dict(total=0.0, stft=0.0, bf=0.0)
    n = 0
    t0 = time.time()
    cap = min(max_steps, total_samples) if (max_steps and total_samples) else (max_steps or total_samples)

    logger.info("  waiting for first batch...")
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, sample in enumerate(samples):
            if i == 0:
                logger.info("  first batch received")
            t_step = time.time()
            loss, sl, bl = process_sample(
                enc, dec, hybrid_loss, sample, device, bf_weight
            )
            if i == 0:
                logger.info(f"  first sample done in {time.time()-t_step:.2f}s")
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), grad_clip)
                optimizer.step()

            totals["total"] += loss.item()
            totals["stft"]  += sl.item()
            totals["bf"]    += bl.item() if isinstance(bl, torch.Tensor) else float(bl)
            n += 1

            if log_interval > 0 and n % log_interval == 0:
                elapsed = time.time() - t0
                avg = elapsed / n
                if cap:
                    eta_s = int((cap - n) * avg)
                    eta_str = f"  ETA={eta_s//3600:02d}h{eta_s%3600//60:02d}m{eta_s%60:02d}s"
                    progress = f"[{n}/{cap}]"
                else:
                    eta_str = ""
                    progress = f"[{n}]"
                logger.info(
                    f"  {progress} "
                    f"total={totals['total']/n:.4f}  "
                    f"stft={totals['stft']/n:.4f}  "
                    f"bf={totals['bf']/n:.4f}  "
                    f"avg={avg:.2f}s/sample{eta_str}"
                )

            if max_steps and n >= max_steps:
                break

    denom = max(n, 1)
    return {key: val / denom for key, val in totals.items()}


# -----------------------------------------------------------------------
# Loss curve plotting
# -----------------------------------------------------------------------

def plot_loss_curves(history: dict, save_path: Path, best_epoch: int = None):
    epochs  = list(range(1, len(history["train"]) + 1))
    metrics = [
        ("total", "Total Loss"),
        ("stft",  "STFT Alignment Loss"),
        ("bf",    "Beamforming L1 Loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Phase Fine-tune Loss Curves", fontsize=13)
    for ax, (key, title) in zip(axes, metrics):
        tr_vals = [h[key] for h in history["train"]]
        vl_vals = [h[key] for h in history["val"]]
        ax.plot(epochs, tr_vals, color="tab:red",  label="Train", linewidth=1.5)
        ax.plot(epochs, vl_vals, color="tab:blue", label="Val",   linewidth=1.5, linestyle="--")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2,
                       label=f"Best (ep{best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------

def init_distributed():
    """Init NCCL process group when launched via torchrun; no-op otherwise.
    Returns (is_dist, rank, world_size).
    """
    if "RANK" not in os.environ:
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return True, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes to both stdout and train.log simultaneously."""
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class _StderrToLogger:
    """Redirect sys.stderr so CUDA/worker errors also land in train.log."""
    def __init__(self, lg: logging.Logger):
        self._lg = lg
        self._buf = ""

    def write(self, msg: str):
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._lg.error(line)

    def flush(self):
        if self._buf.strip():
            self._lg.error(self._buf)
            self._buf = ""

    def fileno(self):
        return sys.__stderr__.fileno()


# Module-level logger — populated in main() before any training code runs.
logger: logging.Logger = logging.getLogger("finetune")


# -----------------------------------------------------------------------
# Config & args
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune SuperRes enc+dec on RealMAN for beamforming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml; reads [phase_finetune] section")
    return p.parse_args()


def load_ft_config(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    ft = raw.get("phase_finetune", {})
    defaults = dict(
        data_root="/gpfs/sjtu/audiocc/data/import/RealMAN",
        pretrain_ckpt="exp/v6-4/best_valid_loss.pth",
        pretrain_config="exp/v6-4/config.yaml",
        exp_dir="exp/phase_finetune",
        epochs=20, learning_rate=1e-4, lr_factor=0.5,
        lr_patience=5, min_lr=1e-7, grad_clip=5.0,
        log_interval=200, seed=42, num_workers=8,
        sr=16000, seg_len=2.0, min_ch=2, max_ch=8,
        snr_min=-5.0, snr_max=20.0, bf_weight=1.0,
        max_steps_per_epoch=None,
        val_rooms=["Cafeteria2", "Park", "Car-Electric"],
    )
    defaults.update(ft)
    return defaults


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    global logger

    args = parse_args()
    ft   = load_ft_config(args.config)
    set_seed(ft["seed"])

    exp_dir = Path(ft["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging: all output (info + errors) goes to both stdout and train.log
    logger = setup_logger(exp_dir / "train.log")
    sys.stderr = _StderrToLogger(logger)

    logger.info(f"Config: {args.config}  [phase_finetune]")
    logger.info(f"Settings: {ft}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = STFTConfig(yaml_path=ft["pretrain_config"])
    enc = SuperResEncoder(cfg).to(device)
    dec = SuperResDecoder(cfg).to(device)
    ckpt = torch.load(ft["pretrain_ckpt"], map_location="cpu")
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    logger.info(f"Loaded checkpoint: {ft['pretrain_ckpt']}  (epoch {ckpt.get('epoch', '?')})")

    hybrid_loss = HybridSupervisionLoss(cfg).to(device)
    for param in hybrid_loss.parameters():
        param.requires_grad_(False)

    data_root  = Path(ft["data_root"])
    train_root = data_root / "train"
    cache_path = str(exp_dir / "realman_index_cache.json")
    all_items, noise_idx = _build_or_load_index(
        train_root / "ma_speech", train_root / "ma_noise", cache_path
    )
    logger.info(f"Total utterances indexed: {len(all_items)}")

    val_rooms   = set(ft["val_rooms"])
    train_items = [(r, c) for r, c in all_items if r not in val_rooms]
    val_items   = [(r, c) for r, c in all_items if r in val_rooms]
    logger.info(f"Train: {len(train_items)}  Val: {len(val_items)}  (val rooms: {sorted(val_rooms)})")

    train_noise = {r: v for r, v in noise_idx.items() if r not in val_rooms}
    val_noise   = {r: v for r, v in noise_idx.items() if r in val_rooms}
    if not val_noise:
        val_noise = noise_idx

    train_ds = RealMANDataset(train_items, train_noise,
                              sr=ft["sr"], seg_len_s=ft["seg_len"],
                              min_ch=ft["min_ch"], max_ch=ft["max_ch"],
                              snr_min=ft["snr_min"], snr_max=ft["snr_max"])
    val_ds   = RealMANDataset(val_items, val_noise,
                              sr=ft["sr"], seg_len_s=ft["seg_len"],
                              min_ch=ft["min_ch"], max_ch=ft["max_ch"],
                              snr_min=ft["snr_min"], snr_max=ft["snr_max"])

    nw = ft["num_workers"]
    loader_kwargs = dict(
        batch_size=1, collate_fn=_collate, pin_memory=False,
        num_workers=nw,
        **(dict(multiprocessing_context="spawn", persistent_workers=True,
                prefetch_factor=2) if nw > 0 else {}),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    logger.info(f"DataLoader: num_workers={nw}"
                + (" (spawn+persistent)" if nw > 0 else " (in-process)"))

    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=ft["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=ft["lr_factor"],
        patience=ft["lr_patience"], min_lr=ft["min_lr"],
    )

    def iter_samples(loader):
        for batch in loader:
            for item in batch:
                yield item

    # ------------------------------------------------------------------ #
    # Checkpoint resume: find latest {N}epoch.pth and restore all state  #
    # ------------------------------------------------------------------ #
    best_val  = float("inf")
    history   = {"train": [], "val": []}
    start_epoch = 1
    best_ckpt_path = None   # absolute path of current best checkpoint
    last_ckpt_path = None   # absolute path of last-saved checkpoint

    ckpt_files = sorted(
        [f for f in exp_dir.iterdir() if re.fullmatch(r"\d+epoch\.pth", f.name)],
        key=lambda f: int(re.findall(r"\d+", f.name)[0]),
    )
    if ckpt_files:
        resume_path = ckpt_files[-1].resolve()
        logger.info(f"Resuming from checkpoint: {resume_path.name}")
        ckpt = torch.load(str(resume_path), map_location="cpu")
        enc.load_state_dict(ckpt["encoder"])
        dec.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val     = ckpt["best_val"]
        history      = ckpt["history"]
        last_ckpt_path = str(resume_path)
        saved_best = ckpt.get("best_ckpt_path")
        best_ckpt_path = str(Path(saved_best).resolve()) if saved_best else str(resume_path)
        logger.info(f"  last={resume_path.name}  best={Path(best_ckpt_path).name}  "
                    f"start_epoch={start_epoch}  best_val={best_val:.4f}")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, ft["epochs"] + 1):
        t0 = time.time()
        logger.info(f"\n=== Epoch {epoch}/{ft['epochs']}  (lr={optimizer.param_groups[0]['lr']:.2e}) ===")

        logger.info("  [Train]")
        try:
            tr = run_epoch(enc, dec, hybrid_loss, optimizer,
                           iter_samples(train_loader), device,
                           is_train=True, grad_clip=ft["grad_clip"],
                           bf_weight=ft["bf_weight"], log_interval=ft["log_interval"],
                           total_samples=len(train_ds),
                           max_steps=ft.get("max_steps_per_epoch"))
        except Exception:
            logger.error(f"[ERROR] Train epoch {epoch} failed:\n{traceback.format_exc()}")
            break

        logger.info("  [Val]")
        try:
            vl = run_epoch(enc, dec, hybrid_loss, None,
                           iter_samples(val_loader), device,
                           is_train=False, grad_clip=ft["grad_clip"],
                           bf_weight=ft["bf_weight"], log_interval=0,
                           total_samples=len(val_ds),
                           max_steps=ft.get("max_steps_per_epoch"))
        except Exception:
            logger.error(f"[ERROR] Val epoch {epoch} failed:\n{traceback.format_exc()}")
            break

        elapsed = time.time() - t0
        logger.info(
            f"  Epoch {epoch} | "
            f"train total={tr['total']:.4f} stft={tr['stft']:.4f} bf={tr['bf']:.4f} | "
            f"val total={vl['total']:.4f} stft={vl['stft']:.4f} bf={vl['bf']:.4f} | "
            f"{elapsed:.0f}s"
        )

        scheduler.step(vl["total"])
        history["train"].append(tr)
        history["val"].append(vl)

        best_ep = int(np.argmin([h["total"] for h in history["val"]])) + 1
        plot_loss_curves(history, exp_dir / "loss_curves.png", best_epoch=best_ep)

        # ---------- determine if this epoch is new best ----------
        is_new_best = vl["total"] < best_val
        if is_new_best:
            best_val = vl["total"]
            logger.info(f"  ** New best val loss: {best_val:.4f}")

        # ---------- save current checkpoint as {epoch}epoch.pth ----------
        current_ckpt_path = str((exp_dir / f"{epoch}epoch.pth").resolve())
        ckpt_out = {
            "epoch":          epoch,
            "encoder":        enc.state_dict(),
            "decoder":        dec.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val":       best_val,
            "history":        history,
            "best_ckpt_path": current_ckpt_path if is_new_best else best_ckpt_path,
        }
        torch.save(ckpt_out, current_ckpt_path)

        # ---------- cleanup: delete last ckpt if it is not best ----------
        if last_ckpt_path and last_ckpt_path != best_ckpt_path:
            if os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
                logger.info(f"  removed old latest: {Path(last_ckpt_path).name}")

        # ---------- update best pointer (delete old best if superseded) ----------
        if is_new_best:
            if best_ckpt_path and best_ckpt_path != current_ckpt_path:
                if os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                    logger.info(f"  removed old best: {Path(best_ckpt_path).name}")
            best_ckpt_path = current_ckpt_path

        last_ckpt_path = current_ckpt_path

        # ---------- loss CSV ----------
        csv_path = exp_dir / "loss_history.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a") as fcsv:
            if write_header:
                fcsv.write("epoch,tr_total,tr_stft,tr_bf,vl_total,vl_stft,vl_bf\n")
            fcsv.write(
                f"{epoch},{tr['total']:.6f},{tr['stft']:.6f},{tr['bf']:.6f},"
                f"{vl['total']:.6f},{vl['stft']:.6f},{vl['bf']:.6f}\n"
            )

    # ---------- finalize: copy best to best_valid_loss.pth ----------
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        dst = str(exp_dir / "best_valid_loss.pth")
        shutil.copy(best_ckpt_path, dst)
        logger.info(f"Copied {Path(best_ckpt_path).name} -> best_valid_loss.pth")

    logger.info(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        msg = traceback.format_exc()
        logger.error(f"[FATAL] {msg}")
        sys.exit(1)
