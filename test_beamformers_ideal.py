#!/usr/bin/env python3
"""
Batch evaluation of non-neural beamformers on CHiME4 simu 6ch track data.

Uses ideal IRM masks (oracle) and a pretrained SuperRes encoder/decoder
in place of the standard STFT/iSTFT.

Noise is derived as: noise = mix - clean  (exact for simu data)
Channel layout: CH1 CH3 CH4 CH5 CH6 → index 0 1 2 3 4; ref_channel=3 → CH5

Usage
-----
    python test_beamformers_ideal.py \
        --data_root /home/wangyou.zhang/espnet_my/egs2/chime4/enh1 \
        --splits dt05 et05 \
        --ref_channel 3 \
        --config_yaml exp/v6-4/config.yaml \
        --checkpoint  exp/v6-4/best_valid_loss.pth \
        --output_dir  /tmp/enh_out

Note: the following user-supplied aliases are resolved automatically:
  mvar  → mvdr   |   wmpd  → wmpdr   |   mwt → mwf
"""

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import espnet2.enh.layers.beamformer_th as bf
from super_res_spec import STFTConfig, SuperResEncoder, SuperResDecoder

# ---- beamformer name aliases (typo-tolerance) ----
BF_ALIASES = {
    "mvar": "mvdr",
    "wmpd": "wmpdr",
    "mwt":  "mwf",
}

DEFAULT_BFS = [
    "mvdr_souden", "mvdr",
    "mpdr_souden", "mpdr",
    "wmpdr_souden", "wmpdr",
    "mwf",
    "wpd_souden", "wpd",
]

MULTI_SPK_ONLY = {"lcmv", "lcmp", "wlcmp", "mvdr_tfs", "mvdr_tfs_souden"}


# ------------------------------------------------------------------ #
# Model loading and encode / decode wrappers
# ------------------------------------------------------------------ #

def load_model(config_yaml: str, checkpoint: str, device: torch.device):
    cfg = STFTConfig(yaml_path=config_yaml)
    enc = SuperResEncoder(cfg).to(device)
    dec = SuperResDecoder(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval()
    dec.eval()
    return enc, dec, cfg


def encode_mc(wav_mc: torch.Tensor, enc: SuperResEncoder) -> torch.Tensor:
    """(C, T) → (C, F, T_f) complex64"""
    x = wav_mc.unsqueeze(1)          # (C, 1, T)
    with torch.no_grad():
        spec = enc(x)                # (C, F, T_f, 2)
    return torch.view_as_complex(spec.contiguous())  # (C, F, T_f)


def decode_ch(spec: torch.Tensor, T: int, dec: SuperResDecoder) -> torch.Tensor:
    """(F, T_f) complex → (T,)"""
    spec_ri = torch.view_as_real(spec).unsqueeze(0)  # (1, F, T_f, 2)
    with torch.no_grad():
        wav = dec(spec_ri, target_len=T)             # (1, 1, T)
    return wav.squeeze()


# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #

def parse_scp(scp_path: str, data_root: str) -> dict:
    """Read Kaldi-style scp → {utt_id: absolute_path}"""
    entries = {}
    with open(scp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, rel_path = parts
            entries[utt_id] = str(Path(data_root) / rel_path)
    return entries


def load_utterance(mix_path: str, clean_path: str, device: torch.device):
    """Load 5ch mix and clean flac, compute noise = mix - clean.

    Returns (mix_mc, speech_mc, noise_mc) each (C, T) on device, plus T and sr.
    """
    mix,   sr = torchaudio.load(mix_path)
    clean, _  = torchaudio.load(clean_path)
    T = min(mix.shape[1], clean.shape[1])
    mix   = mix[:, :T].to(device)
    clean = clean[:, :T].to(device)
    noise = mix - clean
    return mix, clean, noise, T, sr


# ------------------------------------------------------------------ #
# Ideal masks
# ------------------------------------------------------------------ #

def ideal_ratio_mask(S_speech: torch.Tensor, S_noise: torch.Tensor):
    """(C, F, T_f) complex → mask_speech, mask_noise each (F, C, T_f) float"""
    ps = S_speech.abs().pow(2)
    pn = S_noise.abs().pow(2)
    denom = ps + pn + 1e-10
    ms = (ps / denom).permute(1, 0, 2)
    mn = (pn / denom).permute(1, 0, 2)
    return ms, mn


def ideal_binary_mask(S_speech: torch.Tensor, S_noise: torch.Tensor):
    """(C, F, T_f) complex → mask_speech, mask_noise each (F, C, T_f) float"""
    ms = (S_speech.abs() >= S_noise.abs()).float().permute(1, 0, 2)
    return ms, 1.0 - ms


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    est = estimate - estimate.mean()
    ref = target   - target.mean()
    alpha = (est * ref).sum() / (ref.pow(2).sum() + eps)
    proj  = alpha * ref
    noise = est - proj
    return (10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + eps))).item()


# ------------------------------------------------------------------ #
# Beamformer application
# ------------------------------------------------------------------ #

def apply_beamformer(
    mix_stft:    torch.Tensor,   # (C, F, T_f) complex
    mask_speech: torch.Tensor,   # (F, C, T_f) float
    mask_noise:  torch.Tensor,   # (F, C, T_f) float
    btype: str,
    ref_ch:    int = 0,
    rtf_iter:  int = 3,
    diag_eps:  float = 1e-7,
    btaps:     int = 5,
    bdelay:    int = 3,
) -> torch.Tensor:
    """Apply beamformer; returns enhanced (F, T_f) complex64."""
    data = mix_stft.permute(1, 0, 2).unsqueeze(0).cdouble()   # (1, F, C, T)
    ms   = mask_speech.unsqueeze(0).double()                   # (1, F, C, T)
    mn   = mask_noise.unsqueeze(0).double()

    stats = bf.prepare_beamformer_stats(
        data, [ms], mn,
        beamformer_type=btype, bdelay=bdelay, btaps=btaps, eps=1e-6,
    )
    psd_s    = stats["psd_speech"]
    psd_n    = stats["psd_n"]
    psd_dist = stats.get("psd_distortion")

    C = mix_stft.shape[0]
    if (btype.endswith("_souden")
            or btype.startswith("gev")
            or btype in ("mwf", "wmwf", "sdw_mwf", "r1mwf")):
        u = torch.zeros(1, C, dtype=torch.double, device=data.device)
        u[0, ref_ch] = 1.0
    else:
        u = ref_ch

    if btype in ("mvdr_souden", "mpdr_souden", "wmpdr_souden"):
        w = bf.get_mvdr_vector(psd_s, psd_n, u, diagonal_loading=True, diag_eps=diag_eps)
        out = bf.apply_beamforming_vector(w, data)

    elif btype in ("mvdr", "mpdr", "wmpdr"):
        w = bf.get_mvdr_vector_with_rtf(
            psd_n, psd_s, psd_dist,
            iterations=rtf_iter, reference_vector=u,
            diagonal_loading=True, diag_eps=diag_eps,
        )
        out = bf.apply_beamforming_vector(w, data)

    elif btype in ("mwf", "wmwf"):
        w = bf.get_mwf_vector(psd_s, psd_n, u, diagonal_loading=True, diag_eps=diag_eps)
        out = bf.apply_beamforming_vector(w, data)

    elif btype == "wpd_souden":
        w = bf.get_WPD_filter_v2(psd_s, psd_n, u, diagonal_loading=True, diag_eps=diag_eps)
        out = bf.perform_WPD_filtering(w, data, bdelay, btaps)

    elif btype == "wpd":
        w = bf.get_WPD_filter_with_rtf(
            psd_n, psd_s, psd_dist,
            iterations=rtf_iter, reference_vector=u,
            diagonal_loading=True, diag_eps=diag_eps,
        )
        out = bf.perform_WPD_filtering(w, data, bdelay, btaps)

    else:
        raise ValueError(f"Unsupported beamformer type: '{btype}'")

    return out.squeeze(0).to(torch.complex64)   # (F, T_f)


# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #

def parse_args():
    _here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Batch beamformer evaluation on CHiME4 simu 6ch data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", default="/home/wangyou.zhang/espnet_my/egs2/chime4/enh1",
                   help="Root of the CHiME4 enh1 recipe")
    p.add_argument("--splits", nargs="+", default=["dt05", "et05"],
                   help="Which splits to evaluate (dt05, et05, tr05)")
    p.add_argument("--ref_channel", type=int, default=3,
                   help="Reference mic index in the 5-ch array (3 = CH5)")
    p.add_argument("--mask_type", choices=["irm", "ibm"], default="irm")
    p.add_argument("--config_yaml", default=str(_here / "exp/v6-4/config.yaml"))
    p.add_argument("--checkpoint",  default=str(_here / "exp/v6-4/best_valid_loss.pth"))
    p.add_argument("--beamformers", nargs="+", default=DEFAULT_BFS,
                   metavar="BF", help="Beamformer types to evaluate")
    p.add_argument("--btaps",  type=int, default=5)
    p.add_argument("--bdelay", type=int, default=3)
    p.add_argument("--rtf_iterations", type=int, default=3)
    p.add_argument("--save_wav", action="store_true",
                   help="Save per-utterance enhanced WAVs under results dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_utts", type=int, default=None,
                   help="Cap utterances per split (for quick debugging)")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Statistics helpers
# ------------------------------------------------------------------ #

def compute_stats(vals: np.ndarray, input_mean: float) -> dict:
    """Compute rich statistics for a 1-D array of SI-SDR values."""
    ok = vals[~np.isnan(vals)]
    n  = len(ok)
    if n == 0:
        return {}
    mean   = ok.mean()
    std    = ok.std(ddof=1) if n > 1 else 0.0
    sem    = std / np.sqrt(n)
    ci95   = scipy_stats.t.ppf(0.975, df=max(n - 1, 1)) * sem
    return {
        "n":       n,
        "n_fail":  int(np.isnan(vals).sum()),
        "mean":    mean,
        "std":     std,
        "ci95":    ci95,
        "median":  float(np.median(ok)),
        "p25":     float(np.percentile(ok, 25)),
        "p75":     float(np.percentile(ok, 75)),
        "p10":     float(np.percentile(ok, 10)),
        "p90":     float(np.percentile(ok, 90)),
        "min":     float(ok.min()),
        "max":     float(ok.max()),
        "delta":   mean - input_mean,
    }


def fmt_stats_row(label: str, s: dict, w: int = 22) -> str:
    if not s:
        return f"{label:<{w}}  ALL FAILED"
    ci = f"±{s['ci95']:.2f}"
    return (
        f"{label:<{w}}"
        f"  {s['n']:>5}"
        f"  {s['mean']:>7.2f}"
        f"  {s['std']:>6.2f}"
        f"  {ci:>8}"
        f"  {s['median']:>7.2f}"
        f"  {s['p25']:>6.2f}"
        f"  {s['p75']:>6.2f}"
        f"  {s['p10']:>6.2f}"
        f"  {s['p90']:>6.2f}"
        f"  {s['min']:>7.2f}"
        f"  {s['max']:>7.2f}"
        f"  {s['delta']:>+7.2f}"
        + (f"  [{s['n_fail']} fail]" if s["n_fail"] else "")
    )


STATS_HDR = (
    f"{'Label':<22}"
    f"  {'N':>5}"
    f"  {'Mean':>7}"
    f"  {'Std':>6}"
    f"  {'95% CI':>8}"
    f"  {'Median':>7}"
    f"  {'P25':>6}"
    f"  {'P75':>6}"
    f"  {'P10':>6}"
    f"  {'P90':>6}"
    f"  {'Min':>7}"
    f"  {'Max':>7}"
    f"  {'ΔMean':>7}"
)


def write_stats_table(
    lines: list[str],
    title: str,
    bfs: list[str],
    scores: dict,        # btype → np.ndarray of all values
    input_vals: np.ndarray,
    grouper=None,        # optional fn(utt_id) → group label
    group_name: str = "",
) -> list[str]:
    """Append a formatted stats block to `lines` and return it."""
    input_mean = float(np.nanmean(input_vals)) if len(input_vals) else float("nan")
    sep = "=" * len(STATS_HDR)

    lines += [sep, title, sep, STATS_HDR, "-" * len(STATS_HDR)]

    # input row
    in_s = compute_stats(input_vals, input_mean)
    lines.append(fmt_stats_row("Input (ref ch)", in_s))
    lines.append("-" * len(STATS_HDR))

    for btype in bfs:
        vals = scores[btype]
        s = compute_stats(vals, input_mean)
        lines.append(fmt_stats_row(btype, s))

    lines.append(sep)
    return lines


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    device = torch.device(args.device)

    # resolve aliases and drop multi-speaker-only types
    bfs = []
    for b in args.beamformers:
        resolved = BF_ALIASES.get(b, b)
        if resolved in MULTI_SPK_ONLY:
            print(f"[SKIP] {b} → {resolved} (multi-speaker only)")
            continue
        if b != resolved:
            print(f"[ALIAS] {b} → {resolved}")
        bfs.append(resolved)

    # create timestamped results directory
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path(__file__).resolve().parent / "results" / f"eval_{ts}"
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {res_dir}")

    # load pretrained encoder / decoder
    print(f"\nLoading model ...")
    print(f"  config:     {args.config_yaml}")
    print(f"  checkpoint: {args.checkpoint}")
    enc, dec, cfg = load_model(args.config_yaml, args.checkpoint, device)
    print(f"  base_win_len={cfg.base_win_len}  base_hop_len={cfg.base_hop_len}  "
          f"freq_bins={cfg.base_win_len // 2 + 1}")

    # ---- per-utterance records (for CSV) ----
    # columns: utt_id, split, noise_cond, input_sisdr, <btype>...
    csv_path = res_dir / "per_utt.csv"
    csv_cols = ["utt_id", "split", "noise_cond", "input_sisdr"] + bfs
    csv_rows = []

    # ---- accumulators keyed by (split, btype) and (cond, btype) ----
    split_scores: dict = defaultdict(lambda: defaultdict(list))  # split → btype → [vals]
    cond_scores:  dict = defaultdict(lambda: defaultdict(list))  # cond  → btype → [vals]
    all_scores:   dict = defaultdict(list)                       # btype → [vals]
    split_input:  dict = defaultdict(list)
    cond_input:   dict = defaultdict(list)
    all_input:    list = []

    # ---- iterate over splits ----
    for split in args.splits:
        scp_dir  = Path(args.data_root) / "dump" / "raw" / f"{split}_simu_isolated_6ch_track"
        wav_scp  = scp_dir / "wav.scp"
        spk1_scp = scp_dir / "spk1.scp"

        if not wav_scp.exists() or not spk1_scp.exists():
            print(f"\n[WARN] scp not found in {scp_dir}, skipping.")
            continue

        wav_dict  = parse_scp(str(wav_scp),  args.data_root)
        spk1_dict = parse_scp(str(spk1_scp), args.data_root)
        utt_ids   = sorted(set(wav_dict) & set(spk1_dict))
        if args.max_utts:
            utt_ids = utt_ids[: args.max_utts]

        print(f"\n{'=' * 60}")
        print(f"Split: {split}  ({len(utt_ids)} utterances)")
        print(f"{'=' * 60}")

        t_split = time.perf_counter()
        for i, utt_id in enumerate(utt_ids):
            # noise condition: e.g. "F01_050C0103_BUS_SIMU" → "BUS"
            parts = utt_id.upper().split("_")
            noise_cond = parts[-2] if len(parts) >= 2 else "UNK"

            mix_mc, speech_mc, noise_mc, T, sr = load_utterance(
                wav_dict[utt_id], spk1_dict[utt_id], device
            )

            mix_stft    = encode_mc(mix_mc,    enc)
            speech_stft = encode_mc(speech_mc, enc)
            noise_stft  = encode_mc(noise_mc,  enc)

            if args.mask_type == "irm":
                mask_s, mask_n = ideal_ratio_mask(speech_stft, noise_stft)
            else:
                mask_s, mask_n = ideal_binary_mask(speech_stft, noise_stft)

            ref = args.ref_channel
            speech_ref = speech_mc[ref]
            in_si = si_sdr(mix_mc[ref], speech_ref)

            all_input.append(in_si)
            split_input[split].append(in_si)
            cond_input[noise_cond].append(in_si)

            row = {
                "utt_id": utt_id, "split": split,
                "noise_cond": noise_cond, "input_sisdr": round(in_si, 4),
            }

            for btype in bfs:
                try:
                    enh_stft = apply_beamformer(
                        mix_stft, mask_s, mask_n,
                        btype=btype, ref_ch=ref,
                        rtf_iter=args.rtf_iterations,
                        btaps=args.btaps, bdelay=args.bdelay,
                    )
                    enh_wav = decode_ch(enh_stft, T, dec)
                    val = si_sdr(enh_wav, speech_ref)

                    if args.save_wav:
                        out_p = res_dir / "wav" / split / btype / f"{utt_id}.wav"
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        torchaudio.save(str(out_p), enh_wav.unsqueeze(0).cpu().float(), sr)

                except Exception as exc:
                    val = float("nan")
                    print(f"  [{utt_id}] {btype} FAILED: {exc}")

                all_scores[btype].append(val)
                split_scores[split][btype].append(val)
                cond_scores[noise_cond][btype].append(val)
                row[btype] = round(val, 4) if not np.isnan(val) else "nan"

            csv_rows.append(row)

            if (i + 1) == 1 or (i + 1) % 100 == 0 or (i + 1) == len(utt_ids):
                elapsed = time.perf_counter() - t_split
                avg = elapsed / (i + 1)
                eta = avg * (len(utt_ids) - i - 1)
                print(f"  {i + 1:4d}/{len(utt_ids)}"
                      f"  elapsed={elapsed:.0f}s"
                      f"  avg={avg:.1f}s/utt"
                      f"  ETA={eta:.0f}s (~{eta/3600:.1f}h)")

    # ------------------------------------------------------------------ #
    # Save per-utterance CSV
    # ------------------------------------------------------------------ #
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nPer-utterance scores → {csv_path}")

    # ------------------------------------------------------------------ #
    # Build stats tables
    # ------------------------------------------------------------------ #
    all_input_arr = np.array(all_input, dtype=float)

    report_lines = [
        f"Beamformer Evaluation Report",
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint: {args.checkpoint}",
        f"Splits    : {args.splits}",
        f"Mask type : {args.mask_type.upper()}",
        f"Ref ch    : {args.ref_channel}  (CH5)",
        f"Total utts: {len(all_input)}",
        "",
    ]

    # 1. Overall
    overall_scores = {b: np.array(all_scores[b], dtype=float) for b in bfs}
    write_stats_table(report_lines,
                      "OVERALL",
                      bfs, overall_scores, all_input_arr)

    # 2. Per split
    for split in args.splits:
        if split not in split_scores:
            continue
        sc = {b: np.array(split_scores[split][b], dtype=float) for b in bfs}
        iv = np.array(split_input[split], dtype=float)
        write_stats_table(report_lines, f"SPLIT: {split}", bfs, sc, iv)

    # 3. Per noise condition
    for cond in sorted(cond_scores.keys()):
        sc = {b: np.array(cond_scores[cond][b], dtype=float) for b in bfs}
        iv = np.array(cond_input[cond], dtype=float)
        write_stats_table(report_lines, f"NOISE CONDITION: {cond}", bfs, sc, iv)

    # ------------------------------------------------------------------ #
    # Print + save report
    # ------------------------------------------------------------------ #
    report_txt = "\n".join(report_lines)
    print("\n" + report_txt)

    report_path = res_dir / "summary.txt"
    report_path.write_text(report_txt)
    print(f"\nSummary report → {report_path}")


if __name__ == "__main__":
    main()
