"""
CNN-SNN: A hybrid Convolutional Neural Network - Spiking Neural Network model.

Architecture:
    1D Conv -> BatchNorm -> Leaky Integrate-and-Fire (spike encoding) ->
    FC -> LIF -> FC -> LIF (output)

The convolutional layer extracts temporal features from multi-channel input
signals, which are then encoded into spike trains and processed through
two fully connected spiking layers for binary classification.

Reference:
    R. Gall et al., "CNN-SNN" (see associated publication for full citation).
"""

import torch
import torch.nn as nn
import snntorch


class CnnSnn(nn.Module):
    """Hybrid CNN-SNN for binary classification of multi-channel time-series data.

    Args:
        fs: Sampling frequency (Hz).
        decision_window: Duration of each input window (seconds).
        in_channels: Number of input channels (e.g., EEG electrodes).
        out_channels: Number of convolutional filters / hidden units.
        dropout: Dropout probability applied after batch norm. Default: 0.5.
    """

    def __init__(self, fs, decision_window, in_channels, out_channels, dropout=0.5):
        super().__init__()

        self.fs = fs
        self.decision_window = decision_window
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Derived dimensions
        self.window_samples = fs * decision_window
        self.conv_kernel = fs // 4
        self.stride = self.conv_kernel // 2
        self.time_steps = (self.window_samples - self.conv_kernel) // self.stride + 1

        # Learnable LIF parameters (per-neuron beta and threshold)
        beta_encoder = torch.rand(out_channels)
        beta_hidden = torch.rand(out_channels)
        beta_output = torch.rand(2)

        # Layers
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.conv_kernel,
            stride=self.stride,
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.spike_encoder = snntorch.Leaky(beta=beta_encoder, learn_beta=True)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.lif1 = snntorch.Leaky(beta=beta_hidden, learn_beta=True)
        self.fc2 = nn.Linear(out_channels, 2)
        self.lif2 = snntorch.Leaky(beta=beta_output, learn_beta=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, window_samples).

        Returns:
            mem_final: Final membrane potential of the output layer (batch, 2).
            spike_rec: Spike record over time (time_steps, batch, 2).
            mem_rec: Membrane potential record over time (time_steps, batch, 2).
        """
        # Initialize hidden states
        enc_mem = self.spike_encoder.init_leaky()
        hid_mem = self.lif1.init_leaky()
        out_mem = self.lif2.init_leaky()

        spike_rec = []
        mem_rec = []

        # CNN feature extraction: (batch, C_in, L) -> (batch, C_out, T)
        out = self.conv1d(x)
        out = self.bn(out)
        out = self.dropout(out)

        # Reshape for temporal SNN processing: (T, batch, C_out)
        out = out.permute(2, 0, 1)

        # SNN temporal processing
        for time_step in out:
            spk_enc, enc_mem = self.spike_encoder(time_step, enc_mem)
            cur1 = self.fc1(spk_enc)
            spk_hid, hid_mem = self.lif1(cur1, hid_mem)
            cur2 = self.fc2(spk_hid)
            spk_out, out_mem = self.lif2(cur2, out_mem)
            spike_rec.append(spk_out)
            mem_rec.append(out_mem)

        return out_mem, torch.stack(spike_rec), torch.stack(mem_rec)

    def __repr__(self):
        return (
            f"CnnSnn(fs={self.fs}, window={self.decision_window}s, "
            f"in_ch={self.in_channels}, out_ch={self.out_channels}, "
            f"kernel={self.conv_kernel}, stride={self.stride}, "
            f"time_steps={self.time_steps})"
        )