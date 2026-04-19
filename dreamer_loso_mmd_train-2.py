import copy
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from SNN_data import mat_dataset_load, label_balancing, EEG_band_analysis


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EEGUnlabeledDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _, tid = self.base[idx]
        return x, tid


class EEG2DCNNLSTMTemporalDA(nn.Module):
    def __init__(self, fs, input_time, in_channels, out_channels, n_classes, eeg_channels, lstm_hidden=64, lstm_layers=1, dropout=0.5):
        super().__init__()
        self.fs = fs
        self.decision_window = input_time
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.window_samples = fs * input_time
        self.kernel_time = 32
        self.stride_time = 16

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(1, self.kernel_time),
                stride=(1, self.stride_time),
                padding=(0, self.kernel_time // 2),
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.Dropout(dropout),
        )

        self.spat_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(eeg_channels, 1),
                stride=(1, self.stride_time),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=out_channels // 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.spat_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion_dim = lstm_hidden + out_channels // 2
        self.fc = nn.Sequential(
            nn.Linear(self.fusion_dim, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channels, n_classes),
        )

    def extract_features(self, x):
        x_t = self.cnn(x)
        x_t_seq = x_t.squeeze(-1).permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x_t_seq)
        h_last = h_n[-1]

        x_s = self.spat_branch(x)
        x_s = self.spat_pool(x_s)
        x_s = x_s.view(x_s.size(0), -1)

        feat = torch.cat([h_last, x_s], dim=1)
        return feat

    def classify(self, feat):
        return self.fc(feat)

    def forward(self, x, return_feat=False):
        feat = self.extract_features(x)
        logits = self.classify(feat)
        if return_feat:
            return logits, feat
        return logits


def gaussian_kernel(x, y, sigmas=(1, 2, 4, 8, 16)):
    beta = 1.0 / (2.0 * torch.tensor(sigmas, device=x.device, dtype=x.dtype).view(-1, 1, 1))
    dist = torch.cdist(x, y, p=2).pow(2).unsqueeze(0)
    return torch.exp(-beta * dist).sum(dim=0)


def mmd_loss(source, target, sigmas=(1, 2, 4, 8, 16)):
    k_ss = gaussian_kernel(source, source, sigmas)
    k_tt = gaussian_kernel(target, target, sigmas)
    k_st = gaussian_kernel(source, target, sigmas)
    return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()


def build_feature_dataset(split_df, fs=128, window_size=384):
    if len(split_df) == 0:
        raise ValueError('One split is empty.')

    x_list, y_list, tid_list = [], [], []
    for _, row in split_df.iterrows():
        x = np.asarray(row['EEG_array'], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Each EEG_array must have shape (C, T), got {x.shape}")
        featured_x = EEG_band_analysis(fs=fs, seg=x, out_T=window_size)
        x_list.append(featured_x)
        y_list.append(int(row['label']))
        tid_list.append(str(row['trial_id']))

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    _, tid_ints = np.unique(tid_list, return_inverse=True)
    tids = tid_ints.astype(np.int64)

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).long()
    tid_tensor = torch.from_numpy(tids)
    return TensorDataset(x_tensor, y_tensor, tid_tensor)


def dreamer_loso_feature_splits(df, target_subject, val_ratio=0.15, random_state=42, num_channels=14, window_size=384, stride=384, drop_last=True):
    required_cols = {'dataset', 'subject', 'video', 'channel', 'session_idx', 'EEG_clean', 'label'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df.copy()
    df = df[df['dataset'].astype(str).str.lower() == 'dreamer'].copy()
    df = df.sort_values(['dataset', 'subject', 'session_idx', 'video', 'channel']).reset_index(drop=True)

    segmented_rows = []
    trial_group_cols = ['dataset', 'subject', 'session_idx', 'video', 'channel']
    grouped_trial_channel = df.groupby(trial_group_cols)

    for (dset, sub, ses, vid, ch), g in grouped_trial_channel:
        if len(g) != 1:
            raise ValueError(f'(dataset={dset}, subject={sub}, session={ses}, video={vid}, channel={ch}) has {len(g)} rows.')

        full_signal = np.asarray(g.iloc[0]['EEG_clean'], dtype=np.float32)
        label = int(g.iloc[0]['label'])
        if full_signal.ndim != 1:
            raise ValueError(f'Full EEG must be 1D, got {full_signal.shape}')

        T_full = len(full_signal)
        if T_full < window_size:
            if drop_last:
                continue
            padded = np.zeros(window_size, dtype=np.float32)
            padded[:T_full] = full_signal
            segmented_rows.append({
                'dataset': dset, 'subject': sub, 'session_idx': ses, 'video': vid,
                'channel': ch, 'segment': 0, 'EEG_segment': padded, 'label': label,
            })
            continue

        seg_idx = 0
        for start in range(0, T_full, stride):
            end = start + window_size
            if end <= T_full:
                seg = full_signal[start:end]
            else:
                if drop_last:
                    break
                seg = np.zeros(window_size, dtype=np.float32)
                valid_len = T_full - start
                if valid_len <= 0:
                    break
                seg[:valid_len] = full_signal[start:T_full]

            segmented_rows.append({
                'dataset': dset, 'subject': sub, 'session_idx': ses, 'video': vid,
                'channel': ch, 'segment': seg_idx, 'EEG_segment': seg, 'label': label,
            })
            seg_idx += 1

    df_segch = pd.DataFrame(segmented_rows)
    if len(df_segch) == 0:
        raise ValueError('No segmented data generated.')

    df_segch = df_segch.sort_values(['dataset', 'subject', 'session_idx', 'video', 'segment', 'channel']).reset_index(drop=True)
    group_cols = ['dataset', 'subject', 'session_idx', 'video', 'segment']
    grouped = df_segch.groupby(group_cols)

    rows = []
    for (dset, sub, ses, vid, seg), g in grouped:
        if len(g) != num_channels:
            raise ValueError(f'(dataset={dset}, subject={sub}, session={ses}, video={vid}, segment={seg}) has {len(g)} channels, expected {num_channels}')
        signals = []
        for _, row in g.iterrows():
            sig = np.asarray(row['EEG_segment'], dtype=np.float32)
            signals.append(sig)
        lengths = {len(s) for s in signals}
        if len(lengths) != 1:
            raise ValueError(f'Channel lengths mismatch: {lengths}')

        eeg_2d = np.stack(signals, axis=0)
        label = int(g['label'].iloc[0])
        trial_id = f"{dset}__{sub}__{ses}__{vid}"
        rows.append({
            'dataset': dset,
            'subject': sub,
            'session_idx': ses,
            'video': vid,
            'segment': seg,
            'EEG_array': eeg_2d,
            'label': label,
            'trial_id': trial_id,
        })

    seg_df = pd.DataFrame(rows)

    source_df = seg_df[seg_df['subject'] != target_subject].copy()
    target_df = seg_df[seg_df['subject'] == target_subject].copy()
    if len(target_df) == 0:
        raise ValueError(f'No rows found for target subject {target_subject}')

    source_trials = source_df[['trial_id', 'label']].drop_duplicates().reset_index(drop=True)
    train_trials, val_trials = train_test_split(
        source_trials,
        test_size=val_ratio,
        random_state=random_state,
        stratify=source_trials['label'],
        shuffle=True,
    )

    train_df = source_df[source_df['trial_id'].isin(train_trials['trial_id'])].copy()
    valid_df = source_df[source_df['trial_id'].isin(val_trials['trial_id'])].copy()
    test_df = target_df.copy()

    train_dataset = build_feature_dataset(train_df, fs=128, window_size=window_size)
    valid_dataset = build_feature_dataset(valid_df, fs=128, window_size=window_size)
    test_dataset = build_feature_dataset(test_df, fs=128, window_size=window_size)
    target_unlabeled_dataset = EEGUnlabeledDataset(test_dataset)

    return train_dataset, valid_dataset, test_dataset, target_unlabeled_dataset


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, n_classes=4):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    trial_preds = defaultdict(list)
    trial_labels = {}

    for data, labels, trial_ids in dataloader:
        data = data.to(device, dtype=torch.float32).permute(0, 1, 3, 2)
        labels = labels.to(device, dtype=torch.long)
        logits = model(data)
        loss = criterion(logits, labels)

        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        preds_np = pred.cpu().numpy()
        labels_np = labels.cpu().numpy()
        tids_np = trial_ids.cpu().numpy()
        for p, y, tid in zip(preds_np, labels_np, tids_np):
            trial_preds[int(tid)].append(int(p))
            trial_labels[int(tid)] = int(y)

    seg_loss = total_loss / max(total, 1)
    seg_acc = correct / max(total, 1)

    y_true, y_pred = [], []
    for tid in sorted(trial_preds.keys()):
        counts = np.bincount(np.array(trial_preds[tid], dtype=np.int64), minlength=n_classes)
        y_pred.append(int(counts.argmax()))
        y_true.append(int(trial_labels[tid]))
    trial_acc = (np.array(y_true) == np.array(y_pred)).mean() if y_true else 0.0
    return seg_loss, seg_acc, trial_acc


def train_one_epoch_mmd(model, src_loader, tgt_loader, criterion, optimizer, device, lambda_mmd=0.3):
    model.train()
    total_loss, total_cls, total_mmd = 0.0, 0.0, 0.0
    correct, total = 0, 0

    tgt_iter = iter(tgt_loader)
    for src_data, src_labels, _ in src_loader:
        try:
            tgt_data, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            tgt_data, _ = next(tgt_iter)

        src_data = src_data.to(device, dtype=torch.float32).permute(0, 1, 3, 2)
        src_labels = src_labels.to(device, dtype=torch.long)
        tgt_data = tgt_data.to(device, dtype=torch.float32).permute(0, 1, 3, 2)

        optimizer.zero_grad()
        src_logits, src_feat = model(src_data, return_feat=True)
        _, tgt_feat = model(tgt_data, return_feat=True)

        cls = criterion(src_logits, src_labels)
        mmd = mmd_loss(src_feat, tgt_feat)
        loss = cls#  + lambda_mmd * mmd
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * src_data.size(0)
        total_cls += cls.item() * src_data.size(0)
        total_mmd += mmd.item() * src_data.size(0)
        pred = src_logits.argmax(dim=1)
        correct += (pred == src_labels).sum().item()
        total += src_labels.size(0)

    return {
        'loss': total_loss / max(total, 1),
        'cls_loss': total_cls / max(total, 1),
        'mmd_loss': total_mmd / max(total, 1),
        'acc': correct / max(total, 1),
    }


def run_loso_fold(df, target_subject, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_train_ds, src_val_ds, tgt_test_ds, tgt_unlab_ds = dreamer_loso_feature_splits(
        df,
        target_subject=target_subject,
        val_ratio=config['val_ratio'],
        random_state=config['seed'],
        num_channels=config['eeg_channels'],
        window_size=config['window_size'],
        stride=config['stride'],
        drop_last=config['drop_last'],
    )

    src_train_loader = DataLoader(src_train_ds, batch_size=config['batch'], shuffle=True, drop_last=True)
    src_val_loader = DataLoader(src_val_ds, batch_size=config['batch'], shuffle=False, drop_last=False)
    tgt_unlab_loader = DataLoader(tgt_unlab_ds, batch_size=config['batch'], shuffle=True, drop_last=True)
    tgt_test_loader = DataLoader(tgt_test_ds, batch_size=config['batch'], shuffle=False, drop_last=False)

    model = EEG2DCNNLSTMTemporalDA(
        fs=config['fs'],
        input_time=config['decision_window'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_classes=config['n_classes'],
        eeg_channels=config['eeg_channels'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_state = None
    best_trial = -1.0
    patience = 0
    history = []

    for epoch in range(config['num_epochs']):
        train_stats = train_one_epoch_mmd(model, src_train_loader, tgt_unlab_loader, criterion, optimizer, device, lambda_mmd=config['lambda_mmd'])
        val_loss, val_acc, val_trial = evaluate(model, src_val_loader, criterion, device, config['n_classes'])
        test_loss, test_acc, test_trial = evaluate(model, tgt_test_loader, criterion, device, config['n_classes'])

        history.append({
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_cls_loss': train_stats['cls_loss'],
            'train_mmd_loss': train_stats['mmd_loss'],
            'train_acc': train_stats['acc'],
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_trial_acc': val_trial,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_trial_acc': test_trial,
        })

        print(
            f"[Subject {target_subject:02d}] Epoch {epoch:03d} | "
            f"Train Loss {train_stats['loss']:.4f} (CE {train_stats['cls_loss']:.4f}, MMD {train_stats['mmd_loss']:.4f}) "
            f"Acc {train_stats['acc']:.4f} | "
            f"Val Acc {val_acc:.4f} Trial {val_trial:.4f} | "
            f"Test Acc {test_acc:.4f} Trial {test_trial:.4f}"
        )

        if val_trial > best_trial:
            best_trial = val_trial
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config['early_stop_patience']:
                break

    model.load_state_dict(best_state)
    test_loss, test_acc, test_trial = evaluate(model, tgt_test_loader, criterion, device, config['n_classes'])
    return {
        'target_subject': target_subject,
        'best_val_trial_acc': best_trial,
        'final_test_loss': test_loss,
        'final_test_acc': test_acc,
        'final_test_trial_acc': test_trial,
        'history': history,
    }


def load_dreamer_dataframe(csv_path='./data/EEG_clean_table.csv', balance=True):
    dreamer = mat_dataset_load(csv_path)
    dreamer['session_idx'] = -1
    dreamer['dataset'] = 'dreamer'
    if balance:
        dreamer = label_balancing(dreamer)
    return dreamer


def main():
    CONFIG = {
        'fs': 128,
        'decision_window': 3,
        'window_size': 384,
        'stride': 384,
        'drop_last': True,
        'in_channels': 9,
        'eeg_channels': 14,
        'out_channels': 30,
        'lstm_hidden': 64,
        'lstm_layers': 1,
        'dropout': 0.25,
        'n_classes': 4,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'lambda_mmd': 0.3,
        'num_epochs': 60,
        'early_stop_patience': 10,
        'batch': 32,
        'val_ratio': 0.15,
        'seed': 42,
        'subjects': None,
    }

    set_seed(CONFIG['seed'])
    df = load_dreamer_dataframe(balance=True)
    subjects = sorted(df['subject'].dropna().astype(int).unique().tolist())
    if CONFIG['subjects'] is not None:
        subjects = [s for s in subjects if s in CONFIG['subjects']]

    print('Dreamer subjects:', subjects)
    print('Rows:', len(df))
    print('Label counts:')
    print(df['label'].value_counts().sort_index())

    all_results = []
    for subj in subjects:
        fold_result = run_loso_fold(df, subj, CONFIG)
        all_results.append(fold_result)
        print(f"[Subject {subj:02d}] Best Val Trial {fold_result['best_val_trial_acc']:.4f} | Final Test Trial {fold_result['final_test_trial_acc']:.4f}")

    rows = [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
    out_df = pd.DataFrame(rows)
    out_dir = Path('./output')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'dreamer_loso_mmd_results.csv'
    out_df.to_csv(out_csv, index=False)

    print('\n=== LOSO Summary ===')
    print(out_df[['target_subject', 'best_val_trial_acc', 'final_test_acc', 'final_test_trial_acc']])
    print('\nMean test seg acc:', out_df['final_test_acc'].mean())
    print('Mean test trial acc:', out_df['final_test_trial_acc'].mean())
    print(f'Results saved to: {out_csv}')


if __name__ == '__main__':
    main()
