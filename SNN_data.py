import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import TensorDataset

from scipy.signal import butter, filtfilt, stft
from scipy.interpolate import interp1d


from sklearn.model_selection import train_test_split

fs = 128 # Hz, change if needed


import torch
from torch.utils.data import Dataset
import ast

def parse_eeg_str(s):
    if isinstance(s, (list, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    # 去除空白，處理成 python literal
    try:
        arr = np.array(ast.literal_eval(s), dtype=np.float32)
    except Exception:
        # 如果不是標準 list 格式，改用逗點分割
        arr = np.array([float(x) for x in s.replace('[','').replace(']','').split(',')], 
                       dtype=np.float32)
    return arr

# =============================================================================
# Data Augmentation
# =============================================================================
class EEGAugmentDataset(Dataset):
    def __init__(self, base_dataset,
                 noise_std=0.01,
                 amp_scale_range=(0.9, 1.1),
                 max_shift_ratio=0.05):
        """
        base_dataset: 原本的 TensorDataset(x, y, tid)
        noise_std:    噪音強度比例 (乘以每個樣本的 std)
        amp_scale_range: 幅度縮放範圍 (min, max)
        max_shift_ratio: 最多平移多少比例的時間長度 (例如 0.05 = 5%)

        注意: 只用在 training dataloader 上，valid / test 不要用這個包。
        """
        self.base = base_dataset
        self.noise_std = noise_std
        self.amp_scale_range = amp_scale_range
        self.max_shift_ratio = max_shift_ratio

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, tid = self.base[idx]  # x: (C_in, T, ?) 依你現在的設定是 (9, C, T) 或 (9, T, C)

        # 只在訓練時做 augmentation，這個 class 本來就只會用在 train_loader
        x_aug = x.clone()

        # 假設 x_aug shape = (C_in, T, C_eeg)
        # 若你之後有 permute，這裡仍然是 (channels, time, electrodes)

        # 1) amplitude scaling
        if self.amp_scale_range is not None:
            low, high = self.amp_scale_range
            scale = torch.empty(1).uniform_(low, high)
            x_aug = x_aug * scale

        # 2) 加小高斯噪音 (jitter)
        if self.noise_std is not None and self.noise_std > 0:
            # 以每個 sample 的 std 當基準
            std = x_aug.std()
            noise = torch.randn_like(x_aug) * (self.noise_std * std)
            x_aug = x_aug + noise

        return x_aug, y, tid


# =============================================================================
# Data loading — MODIFY THIS FOR YOUR DATASET
# =============================================================================
def load_data(
    df,
    train_ratio=0.6,
    valid_ratio=0.2,
    test_ratio=0.2,
    s_time = 10, # trim time start
    e_time = 10, # trim time end
    random_state=42,
    num_channels=14,
    window_size=384,     # e.g. 3 sec if fs=128
    stride=384,          # non-overlap; set 192 for 50% overlap
    drop_last=True,      # whether to drop the last incomplete window
):
    """
    Input df:
        one row = one channel of one full EEG trial

    Expected columns:
        - subject
        - video
        - channel
        - EEG_array   : full 1D EEG signal of that channel, shape (T_full,)
        - label

    Pipeline:
        1. segment each full trial signal into windows
        2. stack 14 channels into 2D EEG_array of shape (C, T_window)
        3. split by (subject, video), so all segments from one trial stay together

    Returns:
        train_dataset, valid_dataset, test_dataset, in_channels
    """

    # -----------------------------
    # 0) basic checks
    # -----------------------------
    required_cols = {"subject", "video", "channel", "EEG_clean", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    df = df.copy()
    df = df.sort_values(["subject", "video", "channel"]).reset_index(drop=True)




    # -----------------------------
    # 1) segment full EEG trial first
    #    output rows: subject, video, channel, segment, EEG_segment, label
    # -----------------------------
    segmented_rows = []

    trial_group_cols = ["subject", "video", "channel", "session_idx"]
    grouped_trial_channel = df.groupby(trial_group_cols)

    for (sub, vid, ch, ses), g in grouped_trial_channel:
        if len(g) != 1:
            raise ValueError(
                f"(subject={sub}, video={vid}, channel={ch}, session={ses}) has {len(g)} rows. "
                "Expected exactly 1 row per full-trial channel."
            )

        full_signal = np.asarray(g.iloc[0]["EEG_clean"], dtype=np.float32)
        label = int(g.iloc[0]["label"])

        # ============= Trim off the first 10 and the last 10 second =============
        start = fs* s_time
        end = fs* e_time
        full_signal = full_signal[start:-end]

        if full_signal.ndim != 1:
            raise ValueError(
                f"(subject={sub}, video={vid}, channel={ch}), session={ses} full EEG must be 1D, "
                f"got shape {full_signal.shape}"
            )

        T_full = len(full_signal)

        # if the total length is smaller then 1 window
        if T_full < window_size:
            # remove incomplete window, length alignment
            if drop_last:
                continue
            
            # add the padding to fill up the incomplete window
            else:
                padded = np.zeros(window_size, dtype=np.float32)
                padded[:T_full] = full_signal
                segmented_rows.append({
                    "subject": sub,
                    "video": vid,
                    "channel": ch,
                    "segment": 0,
                    "EEG_segment": padded,
                    "label": label,
                })
                continue

        seg_idx = 0
        for start in range(0, T_full, stride):
            end = start + window_size
            # Normal segment process
            if end <= T_full:
                seg = full_signal[start:end]

            # edge case
            else:
                # remove any incomplete window, length alignment
                if drop_last:
                    break
                seg = np.zeros(window_size, dtype=np.float32)
                valid_len = T_full - start
                if valid_len <= 0:
                    break
                seg[:valid_len] = full_signal[start:T_full]

            segmented_rows.append({
                "subject": sub,
                "video": vid,
                "channel": ch,
                "segment": seg_idx,
                "EEG_segment": seg,
                "label": label,
            })
            seg_idx += 1

    df_segch = pd.DataFrame(segmented_rows)

    if len(df_segch) == 0:
        raise ValueError("No segmented data generated. Check window_size/stride/drop_last.")

    # -----------------------------
    # 2) stack 14 channels -> 2D (C, T)
    #    one row becomes one (subject, video, segment)
    # -----------------------------
    df_segch = df_segch.sort_values(
        ["subject", "video", "segment", "channel",  "session_idx"]
    ).reset_index(drop=True)

    group_cols = ["subject", "video", "segment",  "session_idx"]
    grouped = df_segch.groupby(group_cols)

    rows = []
    for (sub, vid, seg), g in grouped:
        if len(g) != num_channels:
            raise ValueError(
                f"(subject={sub}, video={vid}, segment={seg}) "
                f"has {len(g)} channels, expected {num_channels}"
            )

        signals = []
        for _, row in g.iterrows():
            sig = np.asarray(row["EEG_segment"], dtype=np.float32)
            if sig.ndim != 1:
                raise ValueError(
                    f"(subject={sub}, video={vid}, segment={seg}) "
                    f"segment must be 1D, got shape {sig.shape}"
                )
            signals.append(sig)

        lengths = {len(s) for s in signals}
        if len(lengths) != 1:
            raise ValueError(
                f"(subject={sub}, video={vid}, segment={seg}) "
                f"channel segment lengths mismatch: {lengths}"
            )

        eeg_2d = np.stack(signals, axis=0)   # (C, T_window)
        label = int(g["label"].iloc[0])

        rows.append({
            "subject": sub,
            "video": vid,
            "segment": seg,
            "EEG_array": eeg_2d,
            "label": label,
        })

    df = pd.DataFrame(rows)

    # -----------------------------
    # 3) build group id = one full trial
    # -----------------------------
    df["group_id"] = df["subject"].astype(str) + "__" + df["video"].astype(str)
    unique_groups = df["group_id"].unique()

    # -----------------------------
    # 4) split by group, not by row
    # -----------------------------
    train_groups, temp_groups = train_test_split(
        unique_groups,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        shuffle=True,
    )

    valid_portion_of_temp = valid_ratio / (valid_ratio + test_ratio)

    valid_groups, test_groups = train_test_split(
        temp_groups,
        test_size=(1.0 - valid_portion_of_temp),
        random_state=random_state,
        shuffle=True,
    )

    df["trial_id"] = df["subject"] * 1000 + df["video"]   #adding trial id for trial-based validation

    train_df = df[df["group_id"].isin(train_groups)].copy()
    valid_df = df[df["group_id"].isin(valid_groups)].copy()
    test_df  = df[df["group_id"].isin(test_groups)].copy()


    # -----------------------------
    # 5) Convert table -> tensors
    # -----------------------------
    def df_to_dataset(split_df):
        if len(split_df) == 0:
            raise ValueError("One split is empty. Adjust split ratios or dataset size.")

        x_list = []
        y_list = []
        tid_list = []

        for _, row in split_df.iterrows():
            x = np.asarray(row["EEG_array"], dtype=np.float32)   # (C, T_window)

            if x.ndim != 2:
                raise ValueError(
                    f"Each EEG_array must have shape (C, T), got shape {x.shape}"
                )
            
            # ===== EEG_band_analysis =====
            featured_x = EEG_band_analysis(fs=fs, seg=x, out_T=window_size)


            x_list.append(featured_x)
            y_list.append(int(row["label"]))
            tid_list.append(int(row["trial_id"]))

        x = np.stack(x_list, axis=0)   # (N, C, T)
        y = np.asarray(y_list, dtype=np.int64)
        tids = np.asarray(tid_list, dtype=np.int64)

        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long()
        tid_tensor = torch.from_numpy(tids).long()

        return TensorDataset(x_tensor, y_tensor, tid_tensor)

    train_dataset = df_to_dataset(train_df)
    valid_dataset = df_to_dataset(valid_df)
    test_dataset  = df_to_dataset(test_df)

     # data augmentation
    print("Run data augmentation")
    train_dataset = EEGAugmentDataset(
        train_dataset,
        noise_std=0.01,             # 先從很小開始
        amp_scale_range=(0.9, 1.1),
        max_shift_ratio=0.05        # 最多平移 5% 的時間長度
    )

    # -----------------------------
    # 6) infer input channels
    # -----------------------------
    sample_x = np.asarray(df.iloc[0]["EEG_array"], dtype=np.float32)
    in_channels = sample_x.shape[0]

    # -----------------------------
    # 7) logging
    # -----------------------------
    print("=== Group-wise split summary ===")
    print(f"Total segment-samples: {len(df)}")
    print(f"Total trials(groups): {len(unique_groups)}")
    print(f"Train segments: {len(train_df)} | groups: {len(train_groups)}")
    print(f"Valid segments: {len(valid_df)} | groups: {len(valid_groups)}")
    print(f"Test  segments: {len(test_df)} | groups: {len(test_groups)}")
    print(f"in_channels: {in_channels}")
    print(f"window_size: {window_size}, stride: {stride}")
    print("No trial is split across train/valid/test.")

    return train_dataset, valid_dataset, test_dataset



def mat_dataset_load(path = '/Users/linyuchun/Desktop/Project/SNN/data/EEG_clean_table.csv', eeg_prefix = 'EEG_clean'):
    seg_eeg = pd.read_csv(path)
    eeg_cols = [c for c in seg_eeg.columns if c.startswith(eeg_prefix)]

    seg_eeg[eeg_prefix] = (
        seg_eeg[eeg_cols]
        .apply(lambda row: row.dropna().astype(float).values, axis=1)
    )

    seg_eeg = seg_eeg.drop(columns=eeg_cols)
    return seg_eeg

def label_balancing(seg_eeg):

    # 已有: seg_eeg，欄位至少有: subject, video, label

    # 1) 建立 trial_id
    seg_eeg["trial_id"] = seg_eeg["subject"] * 1000 + seg_eeg["video"]

    # 2) 先在 trial level 建一個 df_trials，並算出每個 trial 的 label
    #    這裡我用每個 trial 的第一個 segment 的 label（假設同一 trial 內 label 全部一樣）
    df_trials = (
        seg_eeg.groupby("trial_id")
            .agg({
                "subject": "first",
                "video": "first",
                "label": "first",
            })
            .reset_index()
    )

    print("df_trials (trial-level):")
    print(df_trials.head())
    print("len(df_trials) =", len(df_trials))

    # 3) 在 trial level 做 4 類平衡
    labels_trial = df_trials["label"].values  # 長度 = len(df_trials)

    idx0 = np.where(labels_trial == 0)[0]
    idx1 = np.where(labels_trial == 1)[0]
    idx2 = np.where(labels_trial == 2)[0]
    idx3 = np.where(labels_trial == 3)[0]

    n0, n1, n2, n3 = map(len, [idx0, idx1, idx2, idx3])
    print(f"Trial-level counts before balance: [0:{n0}, 1:{n1}, 2:{n2}, 3:{n3}]")

    n_per_class = min(n0, n1, n2, n3)
    print(f"Using {n_per_class} trials per class (total {4 * n_per_class}).")

    # 完全不 random：直接取每一類的前 n_per_class 個 trial index（trial-level index）
    idx0_sel = idx0[:n_per_class]
    idx1_sel = idx1[:n_per_class]
    idx2_sel = idx2[:n_per_class]
    idx3_sel = idx3[:n_per_class]

    keep_idx = np.concatenate([idx0_sel, idx1_sel, idx2_sel, idx3_sel], axis=0)

    print("keep_idx min/max:", keep_idx.min(), keep_idx.max())

    # 這裡用的是 trial-level df_trials，所以 iloc 是安全的
    df_trials_bal = df_trials.iloc[keep_idx].reset_index(drop=True)

    print("Trial-level counts after balance:")
    print(df_trials_bal["label"].value_counts().sort_index())

    # 4) 用保留下來的 trial_id 去 filter 原本 seg_eeg（一次 drop 掉整個 subject-video 的所有 channel/segment）
    keep_trial_ids = df_trials_bal["trial_id"].values
    mask_keep = seg_eeg["trial_id"].isin(keep_trial_ids)
    df_bal = seg_eeg[mask_keep].reset_index(drop=True)

    print("df_bal (segment/channel-level) shape:", df_bal.shape)

    # 檢查 channel/segment-level 的 label 分布（用 merge 把 trial label 帶回來）
    df_bal_with_label = df_bal.merge(
        df_trials_bal[["trial_id", "label"]],
        on="trial_id",
        how="left",
        suffixes=("", "_trial")
    )

    print("Channel-level label counts after trial balancing:")
    print(df_bal_with_label["label_trial"].value_counts().sort_index())

    df_bal_with_label = df_bal_with_label.drop(columns=["label_trial"])
    seg_eeg = df_bal_with_label.drop(columns=["trial_id"])

    return seg_eeg

def EEG_band_analysis(fs, seg, freq_bend = [(1,4), (4,8), (8,13), (13,30)], out_T = 1):
    """
    Input: 
    fs:
        sampling rate.
    seg:
        one row = 14 channels of segmented EEG
        shape = [ch, seq_window]

    freq_band:
        Desired band to be extracted
    
    out_T:
        Desired sample points per segment to be output 

    Pipeline:
        1. segment each full trial signal into windows
        2. stack 14 channels into 2D EEG_array of shape (C, T_window)
        3. s

    Returns:

    """
    C, T = seg.shape
    band_list = []

    for (low, high) in freq_bend:
        band_sig = np.zeros_like(seg)
        psd_all_ch = np.zeros_like(seg)

        # for each channel
        for ch in range(C):
            x = seg[ch,:]
            
            # STFT: Zxx shape = (n_freq, n_time)
            # 例如目標 1 秒視窗 -> nperseg ≈ fs * 1.0，但不能超過 T_raw
            nperseg = max(16, min(out_T, T))  # 至少 16 samples，最多 T_raw

            noverlap = int(nperseg * 0.5)
            noverlap = min(noverlap, nperseg - 1)  
            f, t, Zxx = stft(x, fs=fs, nperseg= nperseg, noverlap=noverlap)

            # PSD
            Pxx = np.abs(Zxx) ** 2  # (n_freq, n_time)

            mask = (f >= low) & (f <= high)
            if np.any(mask):
                power_t = Pxx[mask, :].mean(axis=0).astype(np.float32)
            else:
                power_t = np.zeros(Pxx.shape[1], dtype=np.float32)
            # power_t 現在是長度 n_time 的序列，要插值到 out_T
            n_time = power_t.shape[0]
            if n_time == 1:
                # 只有一個時間點，就直接複製
                seq = np.full((out_T,), power_t[0], dtype=np.float32)
            else:
                # 線性插值到 out_T 個時間點
                x_src = np.linspace(0, 1, n_time)
                x_tgt = np.linspace(0, 1, out_T)
                f_interp = interp1d(x_src, power_t, kind='linear')
                seq = f_interp(x_tgt).astype(np.float32)  # (out_T,)

            psd_all_ch[ch,:] = seq

            # decompose the original EEG
            b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
            band_sig[ch,:] = filtfilt(b, a, x)
        
        # append back to data
        # 4 band data, 4 PSD corresponding to each band, 1 original eeg
        band_list.append(band_sig)   # [C,T]
        band_list.append(psd_all_ch)

    band_list.append(seg)

    # Stack back to the 3D shape
    featured_x = np.stack(band_list, axis=0)
    return featured_x  # [9, C, T]



# ===== test ======
# # restructure seed to distinguish each session
# seed = pd.read_csv('/Users/linyuchun/Desktop/Project/SNN/data/EEG_all_sessions_combined.csv')
# seed = seed.sort_values(["subject", "video", "channel"]).reset_index(drop=True)

# # 對每個 (subject, video, channel) 計算它出現的次數順序 0,1,2 -> session index
# seed["session_idx"] = seed.groupby(["subject", "video", "channel"]).cumcount()
# seed["EEG_clean"] = seed["EEG_clean"].apply(parse_eeg_str)

# dreamer = mat_dataset_load()

# # Add a columns to distingish the db
# dreamer["session_idx"] = -1
# dreamer["dataset"] = "dreamer"
# seed["dataset"] = "seed"

# df = pd.concat([dreamer, seed], axis=0, ignore_index=True)
# df = label_balancing(df)
# train_dataset, valid_dataset, test_dataset = load_data(df)