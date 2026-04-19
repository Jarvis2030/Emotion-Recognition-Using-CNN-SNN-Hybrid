"""
dreamer_crosssubject_diagnosis.py
==================================
Diagnostic visualizations for DREAMER LOSO cross-subject accuracy drop.
Analyzes 6 potential causes:
  1. Per-subject label imbalance (class distribution)
  2. Per-subject trial count (data scarcity)
  3. Per-subject EEG signal quality (SNR proxy)
  4. Cross-subject feature distribution shift (PCA + MMD proxy)
  5. Per-subject segment length distribution (drop_last impact)
  6. Correlation: subject data stats vs test accuracy (if results CSV exists)

Usage:
  python dreamer_crosssubject_diagnosis.py \
      --csv ./data/EEG_clean_table.csv \
      [--results ./output/dreamer_loso_mmd_results.csv]
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as scipy_entropy

# ── Try importing SNN_data helpers; if not available, fall back to raw CSV ──
try:
    from SNN_data import mat_dataset_load, label_balancing
    HAS_SNN_DATA = True
except ImportError:
    HAS_SNN_DATA = False

# ── Color palette ────────────────────────────────────────────────────────────
CLASS_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
CLASS_NAMES  = ['HAHV', 'HALV', 'LAHV', 'LALV']
SUBJ_CMAP    = 'tab20'
WINDOW_SIZE  = 384
FS           = 128

# ─────────────────────────────────────────────────────────────────────────────
# 0. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_df(csv_path: str) -> pd.DataFrame:
    if HAS_SNN_DATA:
        df = mat_dataset_load(csv_path)
        df['session_idx'] = -1
        df['dataset'] = 'dreamer'
    else:
        df = pd.read_csv(csv_path)
    df = df[df['dataset'].astype(str).str.lower() == 'dreamer'].copy()
    df['subject'] = df['subject'].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-subject label distribution
# ─────────────────────────────────────────────────────────────────────────────

def per_subject_label_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Return (subject × label) segment-count matrix."""
    grp = df.groupby(['subject', 'label']).size().unstack(fill_value=0)
    grp.columns = [int(c) for c in grp.columns]
    return grp


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-subject trial count
# ─────────────────────────────────────────────────────────────────────────────

def per_subject_trial_count(df: pd.DataFrame) -> pd.Series:
    return df.drop_duplicates(['subject', 'video']).groupby('subject').size()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-subject SNR proxy (signal RMS / noise RMS)
#    noise = high-freq residual after low-pass (diff proxy)
# ─────────────────────────────────────────────────────────────────────────────

def snr_proxy(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    rms_signal = np.sqrt(np.mean(signal ** 2))
    noise = np.diff(signal)           # first-order diff ≈ high-freq noise
    rms_noise = np.sqrt(np.mean(noise ** 2)) + 1e-12
    return 20 * np.log10(rms_signal / rms_noise + 1e-12)


def per_subject_snr(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        sig = np.asarray(row['EEG_clean'], dtype=np.float32)
        records.append({'subject': int(row['subject']),
                        'channel': row.get('channel', 0),
                        'snr': snr_proxy(sig)})
    snr_df = pd.DataFrame(records)
    return snr_df.groupby('subject')['snr'].mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature distribution shift (PCA on per-subject mean band power)
#    Band power: delta, theta, alpha, beta, gamma
# ─────────────────────────────────────────────────────────────────────────────

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45)}


def bandpower_welch(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    from numpy.fft import rfft, rfftfreq
    n = len(sig)
    freqs = rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(rfft(sig)) ** 2) / n
    bp = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        bp.append(psd[mask].mean() if mask.any() else 0.0)
    return np.array(bp, dtype=np.float32)


def per_subject_band_features(df: pd.DataFrame,
                               max_trials_per_subj: int = 18
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (X: n_segments × n_features, subjects: n_segments)."""
    X_list, subj_list = [], []
    for subj, g in df.groupby('subject'):
        vids = g['video'].unique()[:max_trials_per_subj]
        sub_df = g[g['video'].isin(vids)]
        for _, row in sub_df.iterrows():
            sig = np.asarray(row['EEG_clean'], dtype=np.float32)
            bp = bandpower_welch(sig)
            X_list.append(bp)
            subj_list.append(int(subj))
    return np.array(X_list, dtype=np.float32), np.array(subj_list, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-subject effective segment count after drop_last windowing
# ─────────────────────────────────────────────────────────────────────────────

def count_segments(signal_len: int, window: int = WINDOW_SIZE, stride: int = WINDOW_SIZE) -> int:
    count = 0
    for start in range(0, signal_len, stride):
        if start + window <= signal_len:
            count += 1
    return count


def per_subject_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        sig = np.asarray(row['EEG_clean'], dtype=np.float32)
        T = len(sig)
        n_segs = count_segments(T)
        leftover = T % WINDOW_SIZE
        records.append({
            'subject': int(row['subject']),
            'sig_len': T,
            'n_segs': n_segs,
            'leftover_samples': leftover,
            'leftover_pct': 100.0 * leftover / max(T, 1),
        })
    seg_df = pd.DataFrame(records)
    return seg_df.groupby('subject').agg(
        mean_sig_len=('sig_len', 'mean'),
        total_segs=('n_segs', 'sum'),
        mean_leftover_pct=('leftover_pct', 'mean'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Label entropy per subject (imbalance summary)
# ─────────────────────────────────────────────────────────────────────────────

def label_entropy(dist_row: pd.Series) -> float:
    counts = dist_row.values.astype(float)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(csv_path: str, results_csv: str = None,
                    out_path: str = './output/dreamer_crosssubject_diagnosis.png'):
    print('Loading data ...')
    df = load_df(csv_path)
    subjects = sorted(df['subject'].unique().tolist())
    n_subj = len(subjects)
    print(f'  {n_subj} subjects, {len(df)} rows')

    # ── Compute all stats ────────────────────────────────────────────────────
    print('Computing label distributions ...')
    label_dist = per_subject_label_dist(df)

    print('Computing trial counts ...')
    trial_cnt = per_subject_trial_count(df)

    print('Computing SNR proxy ...')
    snr_series = per_subject_snr(df)

    print('Computing segment stats ...')
    seg_stats = per_subject_segment_stats(df)

    print('Computing band-power PCA ...')
    X_bp, subj_arr = per_subject_band_features(df)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_bp)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    # Per-subject PCA centroid
    pca_centroids = {}
    for s in subjects:
        mask = subj_arr == s
        pca_centroids[s] = X_pca[mask].mean(axis=0) if mask.any() else np.array([0.0, 0.0])
    centroid_arr = np.array([pca_centroids[s] for s in subjects])

    # Intra-subject spread (std of PCA coords) → domain shift proxy
    pca_spread = {}
    for s in subjects:
        mask = subj_arr == s
        pca_spread[s] = X_pca[mask].std() if mask.sum() > 1 else 0.0

    # Label entropy
    entropies = {s: label_entropy(label_dist.loc[s]) for s in subjects if s in label_dist.index}

    # Load accuracy results if available
    result_df = None
    if results_csv and Path(results_csv).exists():
        result_df = pd.read_csv(results_csv)
        result_df['target_subject'] = result_df['target_subject'].astype(int)
        result_df = result_df.set_index('target_subject')
        print(f'  Loaded results for {len(result_df)} subjects.')

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor('#0f0f1a')

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.07, right=0.96, top=0.93, bottom=0.05)

    ax_title  = fig.add_subplot(gs[0, :])   # row0 full-width → stacked bar
    ax_trial  = fig.add_subplot(gs[1, 0])
    ax_snr    = fig.add_subplot(gs[1, 1])
    ax_seg    = fig.add_subplot(gs[1, 2])
    ax_pca    = fig.add_subplot(gs[2, 0])
    ax_spread = fig.add_subplot(gs[2, 1])
    ax_ent    = fig.add_subplot(gs[2, 2])
    ax_corr   = fig.add_subplot(gs[3, :])   # row3 full-width → correlation

    dark_bg  = '#0f0f1a'
    panel_bg = '#1a1a2e'
    text_col = '#e0e0f0'
    grid_col = '#2e2e4a'

    def style_ax(ax, title):
        ax.set_facecolor(panel_bg)
        ax.spines[:].set_color(grid_col)
        ax.tick_params(colors=text_col, labelsize=8)
        ax.xaxis.label.set_color(text_col)
        ax.yaxis.label.set_color(text_col)
        ax.set_title(title, color=text_col, fontsize=10, fontweight='bold', pad=6)
        ax.grid(axis='y', color=grid_col, linewidth=0.5, alpha=0.7)

    fig.suptitle('DREAMER LOSO — Cross-Subject Accuracy Drop Diagnosis',
                 color='white', fontsize=16, fontweight='bold', y=0.97)

    x_idx  = np.arange(n_subj)
    x_labs = [str(s) for s in subjects]

    # ─── Panel 1: Stacked label distribution ────────────────────────────────
    ax_title.set_facecolor(panel_bg)
    ax_title.spines[:].set_color(grid_col)
    ax_title.tick_params(colors=text_col, labelsize=8)
    ax_title.set_title('① Per-Subject Class Distribution  (segment count)',
                        color=text_col, fontsize=10, fontweight='bold', pad=6)
    ax_title.grid(axis='y', color=grid_col, linewidth=0.5, alpha=0.7)

    bottoms = np.zeros(n_subj)
    for ci, cls in enumerate(sorted(label_dist.columns)):
        vals = np.array([label_dist.loc[s, cls] if s in label_dist.index else 0 for s in subjects],
                        dtype=float)
        ax_title.bar(x_idx, vals, bottom=bottoms,
                     color=CLASS_COLORS[ci % len(CLASS_COLORS)],
                     label=CLASS_NAMES[ci % len(CLASS_NAMES)],
                     edgecolor='none', width=0.7)
        bottoms += vals

    ax_title.set_xticks(x_idx)
    ax_title.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
    ax_title.set_xlabel('Subject ID', color=text_col)
    ax_title.set_ylabel('Segment Count', color=text_col)
    legend_handles = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
    ax_title.legend(handles=legend_handles, loc='upper right',
                    facecolor='#1a1a2e', edgecolor=grid_col,
                    labelcolor=text_col, fontsize=8)

    # Overlay label entropy as scatter
    ax2 = ax_title.twinx()
    ax2.set_facecolor('none')
    ax2.tick_params(colors='#ffdd88', labelsize=8)
    ax2.spines[:].set_color(grid_col)
    ax2.spines['right'].set_color('#ffdd88')
    ent_vals = [entropies.get(s, np.nan) for s in subjects]
    ax2.plot(x_idx, ent_vals, 'o--', color='#ffdd88', markersize=5,
             linewidth=1.2, label='Label Entropy (bits)', zorder=5)
    ax2.set_ylabel('Label Entropy (bits)', color='#ffdd88', fontsize=8)
    ax2.set_ylim(0, 2.5)
    ax2.legend(loc='upper left', facecolor='#1a1a2e', edgecolor=grid_col,
               labelcolor='#ffdd88', fontsize=8)

    # ─── Panel 2: Trial count ────────────────────────────────────────────────
    style_ax(ax_trial, '② Trial Count per Subject  (# unique videos)')
    tc_vals = [trial_cnt.get(s, 0) for s in subjects]
    bars = ax_trial.bar(x_idx, tc_vals, color='#4C72B0', edgecolor='none', width=0.7)
    # highlight subjects with fewer trials
    median_tc = np.median(tc_vals)
    for i, v in enumerate(tc_vals):
        if v < median_tc:
            bars[i].set_color('#C44E52')
    ax_trial.axhline(median_tc, color='#ffdd88', linewidth=1, linestyle='--', label=f'Median={median_tc:.0f}')
    ax_trial.set_xticks(x_idx); ax_trial.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
    ax_trial.set_xlabel('Subject ID', color=text_col)
    ax_trial.set_ylabel('# Trials', color=text_col)
    ax_trial.legend(facecolor='#1a1a2e', edgecolor=grid_col, labelcolor=text_col, fontsize=7)
    low_handles = [Patch(facecolor='#4C72B0', label='Above median'),
                   Patch(facecolor='#C44E52', label='Below median')]
    ax_trial.legend(handles=low_handles, facecolor='#1a1a2e',
                    edgecolor=grid_col, labelcolor=text_col, fontsize=7)

    # ─── Panel 3: SNR proxy ──────────────────────────────────────────────────
    style_ax(ax_snr, '③ Mean SNR Proxy per Subject  (dB, higher = cleaner signal)')
    snr_vals = [snr_series.get(s, np.nan) for s in subjects]
    snr_arr  = np.array(snr_vals, dtype=float)
    norm_snr = (snr_arr - np.nanmin(snr_arr)) / (np.nanmax(snr_arr) - np.nanmin(snr_arr) + 1e-8)
    bar_colors = plt.cm.RdYlGn(norm_snr)
    ax_snr.bar(x_idx, snr_arr, color=bar_colors, edgecolor='none', width=0.7)
    ax_snr.axhline(np.nanmedian(snr_arr), color='#ffdd88', linewidth=1,
                   linestyle='--', label=f'Median={np.nanmedian(snr_arr):.1f} dB')
    ax_snr.set_xticks(x_idx); ax_snr.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
    ax_snr.set_xlabel('Subject ID', color=text_col)
    ax_snr.set_ylabel('SNR (dB)', color=text_col)
    ax_snr.legend(facecolor='#1a1a2e', edgecolor=grid_col, labelcolor=text_col, fontsize=7)

    # ─── Panel 4: Segment stats (drop_last waste) ────────────────────────────
    style_ax(ax_seg, '④ drop_last Waste: Mean Leftover % per Subject')
    left_vals = [seg_stats.loc[s, 'mean_leftover_pct'] if s in seg_stats.index else 0.0
                 for s in subjects]
    seg_colors = ['#C44E52' if v > 25 else '#55A868' for v in left_vals]
    ax_seg.bar(x_idx, left_vals, color=seg_colors, edgecolor='none', width=0.7)
    ax_seg.axhline(25, color='#ffdd88', linewidth=1, linestyle='--', label='25% threshold')
    ax_seg.set_xticks(x_idx); ax_seg.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
    ax_seg.set_xlabel('Subject ID', color=text_col)
    ax_seg.set_ylabel('Mean Leftover (%)', color=text_col)
    ax_seg.set_ylim(0, 100)
    handles_seg = [Patch(facecolor='#C44E52', label='>25% wasted'),
                   Patch(facecolor='#55A868', label='≤25% wasted'),
                   Line2D([0], [0], color='#ffdd88', linestyle='--', label='25% line')]
    ax_seg.legend(handles=handles_seg, facecolor='#1a1a2e',
                  edgecolor=grid_col, labelcolor=text_col, fontsize=7)

    # ─── Panel 5: PCA scatter ───────────────────────────────────────────────
    style_ax(ax_pca, '⑤ Band-Power PCA  (cross-subject feature shift)')
    ax_pca.grid(axis='both', color=grid_col, linewidth=0.5, alpha=0.7)
    cmap = plt.cm.get_cmap(SUBJ_CMAP, n_subj)
    for si, s in enumerate(subjects):
        mask = subj_arr == s
        ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=6, alpha=0.35, color=cmap(si), linewidths=0)
        cx, cy = pca_centroids[s]
        ax_pca.scatter(cx, cy, s=45, color=cmap(si),
                       edgecolors='white', linewidths=0.5, zorder=5)
        ax_pca.annotate(str(s), (cx, cy), fontsize=6, color='white',
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', color=text_col)
    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', color=text_col)

    # ─── Panel 6: Intra-subject PCA spread ──────────────────────────────────
    style_ax(ax_spread, '⑥ Intra-Subject Feature Spread  (domain shift proxy, lower = stable)')
    spread_vals = [pca_spread.get(s, 0.0) for s in subjects]
    spread_arr  = np.array(spread_vals, dtype=float)
    norm_sp = (spread_arr - spread_arr.min()) / (spread_arr.max() - spread_arr.min() + 1e-8)
    sp_colors = plt.cm.coolwarm(norm_sp)
    ax_spread.bar(x_idx, spread_vals, color=sp_colors, edgecolor='none', width=0.7)
    ax_spread.axhline(np.median(spread_vals), color='#ffdd88', linewidth=1,
                      linestyle='--', label=f'Median={np.median(spread_vals):.2f}')
    ax_spread.set_xticks(x_idx); ax_spread.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
    ax_spread.set_xlabel('Subject ID', color=text_col)
    ax_spread.set_ylabel('PCA Spread (std)', color=text_col)
    ax_spread.legend(facecolor='#1a1a2e', edgecolor=grid_col, labelcolor=text_col, fontsize=7)

    # ─── Panel 7: Label entropy bar ─────────────────────────────────────────
    style_ax(ax_ent, '⑦ Label Entropy per Subject  (2 bits = perfectly balanced)')
    ent_arr = np.array([entropies.get(s, 0.0) for s in subjects], dtype=float)
    norm_ent = ent_arr / 2.0
    ent_colors = plt.cm.RdYlGn(norm_ent)
    ax_ent.bar(x_idx, ent_arr, color=ent_colors, edgecolor='none', width=0.7)
    ax_ent.axhline(2.0, color='#4C72B0', linewidth=1, linestyle='--', label='Max entropy (2 bits)')
    ax_ent.set_ylim(0, 2.2)
    ax_ent.set_xticks(x_idx); ax_ent.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
    ax_ent.set_xlabel('Subject ID', color=text_col)
    ax_ent.set_ylabel('Entropy (bits)', color=text_col)
    ax_ent.legend(facecolor='#1a1a2e', edgecolor=grid_col, labelcolor=text_col, fontsize=7)

    # ─── Panel 8: Correlation with test accuracy (if results available) ──────
    ax_corr.set_facecolor(panel_bg)
    ax_corr.spines[:].set_color(grid_col)
    ax_corr.tick_params(colors=text_col, labelsize=8)
    ax_corr.xaxis.label.set_color(text_col)
    ax_corr.yaxis.label.set_color(text_col)

    if result_df is not None:
        ax_corr.set_title('⑧ Per-Subject Test Trial Accuracy  vs Diagnosis Factors',
                          color=text_col, fontsize=10, fontweight='bold', pad=6)
        ax_corr.grid(color=grid_col, linewidth=0.5, alpha=0.7)

        acc_vals = [result_df.loc[s, 'final_test_trial_acc']
                    if s in result_df.index else np.nan for s in subjects]
        acc_arr = np.array(acc_vals, dtype=float)

        factors = {
            'Trial Count':     np.array(tc_vals, dtype=float),
            'SNR (dB)':        snr_arr,
            'Label Entropy':   ent_arr,
            'Feature Spread':  spread_arr,
            'Leftover %':      np.array(left_vals, dtype=float),
        }
        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        bar_width = 0.55
        fac_x = np.arange(n_factors)
        corr_vals = []
        for fname in factor_names:
            fvals = factors[fname]
            mask = ~np.isnan(fvals) & ~np.isnan(acc_arr)
            if mask.sum() > 2:
                c = np.corrcoef(fvals[mask], acc_arr[mask])[0, 1]
            else:
                c = 0.0
            corr_vals.append(c)

        bar_cols = ['#55A868' if c > 0 else '#C44E52' for c in corr_vals]
        ax_corr.bar(fac_x, corr_vals, color=bar_cols, edgecolor='none', width=bar_width)
        ax_corr.axhline(0, color=text_col, linewidth=0.8)
        ax_corr.set_xticks(fac_x)
        ax_corr.set_xticklabels(factor_names, fontsize=9)
        ax_corr.set_ylabel("Pearson r  (with test trial acc)", color=text_col)
        ax_corr.set_ylim(-1.1, 1.1)

        # Annotate correlation values
        for xi, rv in zip(fac_x, corr_vals):
            ax_corr.text(xi, rv + (0.04 if rv >= 0 else -0.08),
                         f'{rv:.2f}', ha='center', fontsize=9,
                         color=text_col, fontweight='bold')

        # Also overlay per-subject accuracy on a twin axis
        ax_acc = ax_corr.inset_axes([0.0, -0.42, 1.0, 0.35])
        ax_acc.set_facecolor(panel_bg)
        ax_acc.spines[:].set_color(grid_col)
        ax_acc.tick_params(colors=text_col, labelsize=8)
        norm_acc = (acc_arr - np.nanmin(acc_arr)) / (np.nanmax(acc_arr) - np.nanmin(acc_arr) + 1e-8)
        acc_colors = plt.cm.RdYlGn(norm_acc)
        ax_acc.bar(x_idx, acc_arr, color=acc_colors, edgecolor='none', width=0.7)
        ax_acc.axhline(0.25, color='#aaaaaa', linewidth=0.8, linestyle=':',
                       label='Chance (25%)')
        ax_acc.axhline(np.nanmean(acc_arr), color='#ffdd88', linewidth=1,
                       linestyle='--', label=f'Mean={np.nanmean(acc_arr):.2f}')
        ax_acc.set_xticks(x_idx); ax_acc.set_xticklabels(x_labs, rotation=0, fontsize=7.5)
        ax_acc.set_ylabel('Test Trial Acc', color=text_col, fontsize=8)
        ax_acc.set_ylim(0, 1.0)
        ax_acc.set_title('Test Accuracy per Subject', color=text_col, fontsize=9, pad=4)
        ax_acc.grid(axis='y', color=grid_col, linewidth=0.5, alpha=0.7)
        ax_acc.legend(facecolor='#1a1a2e', edgecolor=grid_col,
                      labelcolor=text_col, fontsize=7, loc='upper right')
        for xi in x_idx:
            if not np.isnan(acc_arr[xi]):
                ax_acc.text(xi, acc_arr[xi] + 0.02, f'{acc_arr[xi]:.2f}',
                            ha='center', fontsize=6, color='white', rotation=70,
                            va='bottom')
    else:
        ax_corr.set_title(
            '⑧ Correlation Analysis  (run with --results to enable)',
            color='#888888', fontsize=10, fontweight='bold', pad=6)
        ax_corr.text(0.5, 0.5,
                     'No results CSV found.\nRun training first, then pass\n'
                     '--results ./output/dreamer_loso_mmd_results.csv',
                     transform=ax_corr.transAxes, ha='center', va='center',
                     color='#888888', fontsize=12)
        ax_corr.axis('off')

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor=dark_bg)
    plt.close(fig)
    print(f'\nDiagnostic plot saved → {out_path}')
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DREAMER cross-subject diagnostic plots')
    parser.add_argument('--csv',     type=str, default='./data/EEG_clean_table.csv',
                        help='Path to EEG_clean_table.csv')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to dreamer_loso_mmd_results.csv (optional)')
    parser.add_argument('--out',     type=str, default='./output/dreamer_crosssubject_diagnosis.png',
                        help='Output PNG path')
    args = parser.parse_args()
    run_diagnostics(args.csv, results_csv=args.results, out_path=args.out)
