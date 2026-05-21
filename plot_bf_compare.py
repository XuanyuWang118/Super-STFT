#!/usr/bin/env python3
"""
Plot overall SI-SDR comparison between SuperRes encoder and standard STFT baseline.

Usage
-----
    python plot_comparison.py \
        --superres exp/bf_eval/eval_20260513_001549/per_utt.csv \
        --stft     exp/bf_eval/eval_stft_20260513_102505/per_utt.csv \
        --output   exp/bf_eval/comparison.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

BFS = [
    "mvdr_souden", "mvdr",
    "mpdr_souden", "mpdr",
    "wmpdr_souden", "wmpdr",
    "mwf",
    "wpd_souden", "wpd",
]

BF_LABELS = {
    "mvdr_souden":  "MVDR\n(Souden)",
    "mvdr":         "MVDR\n(RTF)",
    "mpdr_souden":  "MPDR\n(Souden)",
    "mpdr":         "MPDR\n(RTF)",
    "wmpdr_souden": "wMPDR\n(Souden)",
    "wmpdr":        "wMPDR\n(RTF)",
    "mwf":          "MWF",
    "wpd_souden":   "WPD\n(Souden)",
    "wpd":          "WPD\n(RTF)",
}

COLOR_STFT     = "#5b9bd5"   # blue
COLOR_SUPERRES = "#ed7d31"   # orange
COLOR_INPUT    = "#a5a5a5"   # grey


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def mean_ci95(vals: np.ndarray):
    vals = vals[~np.isnan(vals)]
    n    = len(vals)
    m    = vals.mean()
    se   = vals.std(ddof=1) / np.sqrt(n)
    ci   = scipy_stats.t.ppf(0.975, df=n - 1) * se
    return m, ci


def parse_args():
    _here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--superres", default=str(_here / "exp/bf_eval/eval_20260513_001549/per_utt.csv"))
    p.add_argument("--stft",     default=str(_here / "exp/bf_eval/eval_stft_20260513_102505/per_utt.csv"))
    p.add_argument("--output",   default=str(_here / "exp/bf_eval/comparison.png"))
    return p.parse_args()


def main():
    args = parse_args()

    df_sr   = load_csv(args.superres)
    df_stft = load_csv(args.stft)

    bfs    = [b for b in BFS if b in df_sr.columns]
    labels = [BF_LABELS[b] for b in bfs]
    x      = np.arange(len(bfs))
    width  = 0.32

    # ---- collect means and CIs ----
    means_sr,   ci_sr   = [], []
    means_stft, ci_stft = [], []

    for b in bfs:
        m, c = mean_ci95(df_sr[b].values)
        means_sr.append(m);   ci_sr.append(c)
        m, c = mean_ci95(df_stft[b].values)
        means_stft.append(m); ci_stft.append(c)

    input_mean = mean_ci95(df_sr["input_sisdr"].values)[0]

    # ---- figure layout: 2 subplots ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={"height_ratios": [3, 1.4]})
    fig.subplots_adjust(hspace=0.35)

    # ── Subplot 1: absolute SI-SDR ──────────────────────────────────────
    bars_stft = ax1.bar(x - width / 2, means_stft, width,
                        color=COLOR_STFT, label="Baseline STFT\n(n_fft=1024, hop=16)",
                        yerr=ci_stft, capsize=4, error_kw={"elinewidth": 1.2})
    bars_sr   = ax1.bar(x + width / 2, means_sr, width,
                        color=COLOR_SUPERRES, label="SuperRes Encoder\n(n_fft=1024, hop=16)",
                        yerr=ci_sr, capsize=4, error_kw={"elinewidth": 1.2})

    ax1.axhline(input_mean, color=COLOR_INPUT, linestyle="--", linewidth=1.5,
                label=f"Input (ref ch)  {input_mean:.2f} dB")

    # value labels on top of bars
    for bar in bars_stft:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color=COLOR_STFT)
    for bar in bars_sr:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color=COLOR_SUPERRES)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("SI-SDR (dB)", fontsize=11)
    ax1.set_title("Overall SI-SDR: SuperRes Encoder vs Standard STFT\n"
                  "(CHiME4 simu 6ch, oracle IRM, CH5 ref, tr+dt+et = 10098 utts)",
                  fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xlim(-0.6, len(bfs) - 0.4)

    # ── Subplot 2: ΔSI-SDR (SuperRes − STFT) ───────────────────────────
    deltas = np.array(means_sr) - np.array(means_stft)

    # per-utterance paired difference and CI
    delta_ci = []
    for b in bfs:
        diff = df_sr[b].values - df_stft[b].values
        _, c = mean_ci95(diff)
        delta_ci.append(c)

    bar_colors = [COLOR_SUPERRES if d >= 0 else COLOR_STFT for d in deltas]
    ax2.bar(x, deltas, width * 1.8, color=bar_colors, alpha=0.85,
            yerr=delta_ci, capsize=4, error_kw={"elinewidth": 1.2})

    for i, (d, c) in enumerate(zip(deltas, delta_ci)):
        sign = "+" if d >= 0 else ""
        ax2.text(x[i], d + (0.05 if d >= 0 else -0.15),
                 f"{sign}{d:.2f}", ha="center", va="bottom" if d >= 0 else "top",
                 fontsize=8)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("ΔSI-SDR (dB)\nSuperRes − STFT", fontsize=10)
    ax2.set_title("Per-Beamformer Improvement of SuperRes over Standard STFT", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xlim(-0.6, len(bfs) - 0.4)

    patch_pos = mpatches.Patch(color=COLOR_SUPERRES, alpha=0.85, label="SuperRes better")
    patch_neg = mpatches.Patch(color=COLOR_STFT,     alpha=0.85, label="STFT better")
    ax2.legend(handles=[patch_pos, patch_neg], fontsize=9, loc="upper right")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # ── Console summary ─────────────────────────────────────────────────
    print(f"\n{'Beamformer':<16}  {'STFT':>7}  {'SuperRes':>9}  {'Δ':>7}")
    print("-" * 45)
    for b, ms, mr, d in zip(bfs, means_stft, means_sr, deltas):
        sign = "+" if d >= 0 else ""
        print(f"{b:<16}  {ms:>7.2f}  {mr:>9.2f}  {sign}{d:>6.2f}")
    print(f"\n{'Input mean':<16}  {input_mean:>7.2f}")


if __name__ == "__main__":
    main()
