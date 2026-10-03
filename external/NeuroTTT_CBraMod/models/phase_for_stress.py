# phase.py
import math
import random
import torch

TARGET_CH_NAMES = ['EEG Fz', 'EEG F3', 'EEG F4', 'EEG Fp1', 'EEG Fp2', 'EEG Cz', 'EEG Pz']

class PhasePretext(torch.nn.Module):
    def __init__(self, input_dim, num_steps=4, prop=1.0,
                 channel_names=None, selected_channels=None):
        """
        channel_names: Names of the current input channels (length = input_dim).
                       If omitted, default to selected_channels, matching preprocessing.
        selected_channels: Full preprocessing channel order, used to resolve indices.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps

        # Resolve channel indices in the exact preprocessing order.
        if channel_names is None:
            if selected_channels is None:
                selected_channels = ['EEG Fp1','EEG Fp2','EEG F3','EEG F4','EEG F7','EEG F8',
                                     'EEG T3','EEG T4','EEG C3','EEG C4','EEG T5','EEG T6',
                                     'EEG P3','EEG P4','EEG O1','EEG O2','EEG Fz','EEG Cz',
                                     'EEG Pz','EEG A2-A1']
            channel_names = selected_channels

        name_to_idx = {ch:i for i, ch in enumerate(channel_names)}
        self.target_idxs = torch.tensor(
            [name_to_idx[ch] for ch in TARGET_CH_NAMES if ch in name_to_idx],
            dtype=torch.long
        )
        assert len(self.target_idxs) == 7, f"Found {len(self.target_idxs)} target channel indices; check channel-name consistency."

        self.classifier = torch.nn.Linear(in_features=input_dim, out_features=self.num_steps)

    def phase_shift(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        device = x.device
        target_idxs = self.target_idxs.to(device)

        # Eight discrete phases: {0, pi/8, pi/4, 3pi/8, pi/2, 5pi/8, 3pi/4, 7pi/8}.
        # possible_shifts = torch.linspace(0, 7*math.pi/8, steps=self.num_steps, device=device)
        possible_shifts = torch.tensor([0.0, math.pi/4, math.pi/2, 3*math.pi/4], device=device)
        phase_shift_label = random.randrange(len(possible_shifts))
        phase_shift = possible_shifts[phase_shift_label]

        # Rotate only these seven channels in the frequency domain using the same phase.
        freq_x = torch.fft.fft(x, dim=2)                  # Complex tensor of shape [B, C, T].
        phase_shift_factor = torch.exp(phase_shift * 1j)  # Complex scalar.
        freq_x[:, target_idxs, :] = freq_x[:, target_idxs, :] * phase_shift_factor

        time_x = torch.fft.ifft(freq_x, dim=2).real
        return time_x, phase_shift_label

    def forward(self, shifted_x, shift_label):
        z = self.classifier(shifted_x)
        label = torch.full((shifted_x.shape[0],), fill_value=shift_label,
                           dtype=torch.long, device=shifted_x.device)
        loss = torch.nn.functional.cross_entropy(z, label)
        preds = torch.argmax(torch.nn.functional.softmax(z, dim=-1), dim=-1)
        return loss, preds, label
