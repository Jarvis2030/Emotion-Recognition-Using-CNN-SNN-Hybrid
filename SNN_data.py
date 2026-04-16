import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


import snntorch.functional

from ref_cnn_snn_model import CnnSnn

import snntorch.functional

import lava.lib.dl.slayer as slayer

from sklearn.model_selection import train_test_split


# =============================================================================
# Data loading — MODIFY THIS FOR YOUR DATASET
# =============================================================================
def load_data(
    df,
    train_ratio=0.6,
    valid_ratio=0.2,
    test_ratio=0.2,
    random_state=42,
    num_channels=14,
    window_size=128,     # e.g. 3 sec if fs=128
    stride=128,          # non-overlap; set 192 for 50% overlap
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

    trial_group_cols = ["subject", "video", "channel"]
    grouped_trial_channel = df.groupby(trial_group_cols)

    for (sub, vid, ch), g in grouped_trial_channel:
        if len(g) != 1:
            raise ValueError(
                f"(subject={sub}, video={vid}, channel={ch}) has {len(g)} rows. "
                "Expected exactly 1 row per full-trial channel."
            )

        full_signal = np.asarray(g.iloc[0]["EEG_clean"], dtype=np.float32)
        label = int(g.iloc[0]["label"])

        if full_signal.ndim != 1:
            raise ValueError(
                f"(subject={sub}, video={vid}, channel={ch}) full EEG must be 1D, "
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
        ["subject", "video", "segment", "channel"]
    ).reset_index(drop=True)

    group_cols = ["subject", "video", "segment"]
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

        for _, row in split_df.iterrows():
            x = np.asarray(row["EEG_array"], dtype=np.float32)   # (C, T_window)

            if x.ndim != 2:
                raise ValueError(
                    f"Each EEG_array must have shape (C, T), got shape {x.shape}"
                )

            x_list.append(x)
            y_list.append(int(row["label"]))

        x = np.stack(x_list, axis=0)   # (N, C, T)
        y = np.asarray(y_list, dtype=np.int64)

        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long()
        return TensorDataset(x_tensor, y_tensor)

    train_dataset = df_to_dataset(train_df)
    valid_dataset = df_to_dataset(valid_df)
    test_dataset  = df_to_dataset(test_df)

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


# ===== test ======
df = mat_dataset_load()
print(df)
# df = label_balancing(df)
# train_dataset, valid_dataset, test_dataset = load_data(df)