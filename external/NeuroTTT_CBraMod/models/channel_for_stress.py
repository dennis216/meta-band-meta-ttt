# CBraMod/models/channel.py
import torch
import random

# Channel order (zero-based indices).
DEFAULT_SELECTED_CHANNELS = [
    'EEG Fp1','EEG Fp2','EEG F3','EEG F4','EEG F7','EEG F8',
    'EEG T3','EEG T4','EEG C3','EEG C4','EEG T5','EEG T6',
    'EEG P3','EEG P4','EEG O1','EEG O2','EEG Fz','EEG Cz',
    'EEG Pz','EEG A2-A1'
]

AP_PAIRS_BY_NAME = [
    ("EEG Fp1", "EEG O1"),   # Frontopolar <-> occipital
    ("EEG Fp2", "EEG O2"),
    ("EEG F3",  "EEG P3"),   # Left frontal <-> parietal
    ("EEG F4",  "EEG P4"),   # Right frontal <-> parietal
    ("EEG T3",  "EEG T5"),   # Left anterior <-> posterior temporal
    ("EEG T4",  "EEG T6"),   # Right anterior <-> posterior temporal
    ("EEG Fz",  "EEG Pz"),   # Midline frontal <-> parietal
    # C3/C4/Cz have no direct posterior counterparts; A2-A1 is a reference and is not swapped.
]

class ChannelPretext(torch.nn.Module):
    """
    Randomly swap an anterior/posterior channel pair; the classification head predicts which pair was swapped.
    x shape: (B, C, T)
    """
    def __init__(self, input_dim, channel_names=None, selected_channels=None):
        super().__init__()
        if channel_names is None:
            channel_names = selected_channels or DEFAULT_SELECTED_CHANNELS

        # Channel names -> indices
        name2idx = {ch: i for i, ch in enumerate(channel_names)}

        # Retain pairs only when both channels exist in the current array.
        self.ap_pairs = [(name2idx[a], name2idx[b])
                         for (a, b) in AP_PAIRS_BY_NAME
                         if a in name2idx and b in name2idx]

        assert len(self.ap_pairs) >= 1, "No valid anterior/posterior channel pair found; check channel names."
        self.num_pairs = len(self.ap_pairs)

        # Classifier: one class per channel pair.
        self.classifier = torch.nn.Linear(in_features=input_dim, out_features=self.num_pairs)

    @torch.no_grad()
    def swap_one_ap_pair(self, x):
        """
        Randomly select a (front, back) pair and swap its waveforms.
        Input x: (B, C, T).
        Returns: x_swapped, pair_label (0..num_pairs-1).
        """
        B, C, T = x.shape
        pair_label = random.randrange(self.num_pairs)
        i_front, i_back = self.ap_pairs[pair_label]

        x_swapped = x.clone()
        # Swap these two channels over the entire time interval.
        tmp = x_swapped[:, i_front, :].clone()
        x_swapped[:, i_front, :] = x_swapped[:, i_back, :]
        x_swapped[:, i_back, :] = tmp
        return x_swapped, pair_label

        # To swap all anterior/posterior pairs at once, replace the logic above with these two lines:
        # x_swapped = x.clone()
        # for i_front, i_back in self.ap_pairs:
        #     x_swapped[:, [i_front, i_back], :] = x_swapped[:, [i_back, i_front], :]
        # return x_swapped, -1  # Alternatively, classify whether all pairs were swapped.

    def forward(self, swapped_x, pair_label):
        """
        Predict which channel pair was swapped.
        swapped_x: Tensor produced by swap_one_ap_pair.
        pair_label: Channel-pair index (0..num_pairs-1).
        """
        logits = self.classifier(swapped_x)  # (B, num_pairs)
        labels = torch.full((swapped_x.shape[0],), pair_label,
                            dtype=torch.long, device=swapped_x.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels
