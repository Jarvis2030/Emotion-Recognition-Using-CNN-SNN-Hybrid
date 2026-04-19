import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import snntorch.functional

import lava.lib.dl.slayer as slayer

from sklearn.model_selection import train_test_split

fs = 128 #Hz

class EEG_CNN_LTSM(nn.Module):
    def __init__(self, n_classes=4, input_time=384, input_channels=14,  lstm_hidden=64, lstm_layers=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(25, 1)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(6, 6, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.3, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # nn.Conv2d(6, 12, kernel_size=(25, 1)),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.Dropout(0.4),

            # nn.Conv2d(12, 12, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1)),
            # nn.BatchNorm2d(12),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.MaxPool2d(kernel_size=(2, 1)),
        )


        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_time, input_channels)
            feat = self.features(dummy)
            # flat_dim = feat.view(1, -1).shape[1]
            C_feat, H_feat, W_feat = feat.shape[1:]
             
        # 我們把 W_feat 平均掉，只保留時間方向 H_feat 當作序列長度
        self.C_feat = C_feat
        self.H_feat = H_feat
        self.W_feat = W_feat
             
        # LSTM input_size = C_feat（每個 time step 的 feature 維度）
        self.lstm = nn.LSTM(
            input_size=C_feat,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(8, n_classes)
        )


class EEG2DCNN(nn.Module):
    def __init__(self, n_classes=4, input_time=384, in_channels = 1, eeg_channels=14):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=(eeg_channels, 1)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(10, 10, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.3, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(10, 12, kernel_size=(25, 1)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(12, 12, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.3, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_time, eeg_channels)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 10),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(10, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class CNNBench(nn.Module):
    """Hybrid CNN-SNN for binary classification of multi-channel time-series data.

    Args:
        fs: Sampling frequency (Hz).
        decision_window: Duration of each input window (seconds).
        in_channels: Number of input channels (e.g., EEG electrodes).
        out_channels: Number of convolutional filters / hidden units.
        n_classes: Number of final predicted classes.
        dropout: Dropout probability applied after batch norm. Default: 0.5.
    """
    def __init__(self, fs, decision_window, in_channels, out_channels, n_classes, dropout=0.5):
        super().__init__() # 14 channel, segmetation window = 3s (128Hz)

        self.fs = fs
        self.decision_window = decision_window
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes

        # Derived params
        self.window_samples = fs * decision_window
        self.kernel_time = fs // 4   # 32
        self.stride_time = fs // 8   # 16
        self.time_steps = (self.window_samples - self.kernel_time) // self.stride_time + 1


        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels // 2,
                kernel_size=(1, self.kernel_time),    # (channel=1, time=K)
                stride=(1, self.stride_time),
                padding=(0, self.kernel_time // 2+self.stride_time//2),
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

         # spatial branch: 只沿 channel 捲，time kernel=1
        self.spat_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels // 2,
                kernel_size=(in_channels, self.kernel_time//2),    # 一次看所有 channel
                stride=(1, self.stride_time),
                padding=(0, self.kernel_time // 2),
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.Dropout(dropout),
        )

        # fuse + pool
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.AdaptiveAvgPool2d((1, 1)),
        )


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, self.n_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)               # (B, 1, C, T)

        z_t = self.cnn(x)        # (B, F/2, C, T')
        z_s = self.spat_branch(x)        # (B, F/2, 1, T)
        # 對齊 channel 維度（spatial branch 只有 1，高度 broadcast）
        if z_s.size(2) == 1 and z_t.size(2) > 1:
            z_s = z_s.expand(-1, -1, z_t.size(2), -1)  # (B, F/2, C, T)

        z = torch.cat([z_t, z_s], dim=1) # (B, F, C, T')
        z = self.fuse(z)                 # (B, F, 1, 1)
        z = self.fc(z)       
        # out = self.cnn(x)              # (B, C, 1)
        # out = out.squeeze(-1)          # (B, C)
        # out = self.fc(out)             # (B, num_classes)

        # out = out.unsqueeze(2) 
        return z
    
class SNNEMotionNet(nn.Module):
    """Hybrid CNN-SNN for binary classification of multi-channel time-series data.

    Args:
        fs: Sampling frequency (Hz).
        decision_window: Duration of each input window (seconds).
        in_channels: Number of input channels (e.g., EEG electrodes).
        out_channels: Number of convolutional filters / hidden units.
        n_classes: Number of final predicted classes.
        dropout: Dropout probability applied after batch norm. Default: 0.5.
    """
    def __init__(self, fs, decision_window, in_channels, out_channels, n_classes, dropout=0.5):
        super().__init__() # 14 channel, segmetation window = 3s (128Hz)

        self.fs = fs
        self.decision_window = decision_window
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes

        # Derived params
        self.window_samples = fs * decision_window
        self.kernel_time = fs // 4   # 64
        self.stride_time = fs // 8   # 32
        self.time_steps = (self.window_samples - self.kernel_time) // self.stride_time + 1

        # Neurons param
        encoder_params = {
                "threshold": 0.3,          # U_thr = 1 V from paper
                "current_decay": 0.9,      # Current time step 留下 90%
                "voltage_decay": 0.9,      # Voltage time step 留下 90%
                "tau_grad": 1.0,           # surrogate gradient time const
                "scale_grad": 1.0,         # gradient value, default 1
                "scale": 1 << 6,           # default initial setting
                "norm": None,
                "dropout": None,
                "shared_param": True,      # 先全 channel 共用一組參數
                "persistent_state": False, # 每個 batch 從 0 開始積分，符合論文 batch 訓練
                "requires_grad": True,    # encoder 參數先不學
                "graded_spike": False,     # 輸出 0/1 spike，不要連續值
            }
        dense_params = {
                'threshold'     : 0.1,
                'current_decay' : 1,    # default setting from tutorial, no decay 
                'voltage_decay' : 0.1,  # default setting, keep only 0.1 
                'requires_grad' : True, # learn from back-prop    
            }


        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channels, 
                out_channels = out_channels,
                kernel_size = self.kernel_time,
                stride=self.stride_time, # moving step
            ),
            nn.BatchNorm1d(out_channels),

            nn.Conv1d(out_channels, out_channels,
              kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout), # prevent overfitting
        )

        self.endcoder = nn.ModuleList([
            slayer.block.cuba.Input(encoder_params),  # delta encoding of the input
        ])
        self.snn = nn.ModuleList([
            slayer.block.cuba.Dense(dense_params, out_channels, out_channels),
            slayer.block.cuba.Dense(dense_params, out_channels, self.n_classes),
        ])

        # data visulization param init
        self.spike_mon_enc = None   # 用來暫存 spike
        self.spike_mon_snn1 = None  
        self.spike_mon_snn2 = None   

    def forward(self, x):
        # print("Original Shape: ", x.shape)
        z = self.cnn(x) #  run CNN
        # print("After CNN:",z.shape)
        for block in self.endcoder: # pass into SNN blocks
            z = block(z)
        
        self.spike_mon_enc = z.detach().cpu()  # 存下來，shape [B, 50, 1, T]
        # print("After Encoder:",z.shape)
        # print("After modify", z.shape)

        # --------- For output Checking ----------
        # z = self.snn[0](z)
        # self.spike_mon_snn1 = z.detach().cpu()
        # z = self.snn[1](z)
        # self.spike_mon_snn2 = z.detach().cpu()
        # ---------------end of checking ------
        
        for block in self.snn: # pass into SNN blocks
            z = block(z)
        # print("Final output shape: ", z.shape)
        # z = z.squeeze(0).squeeze(2)   # [B, C, T] 
        return z