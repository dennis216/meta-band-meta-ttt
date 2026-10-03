# # CBraMod/models/reverse.py
# import random
# import torch
# import torch.nn as nn

# class ReversePretext(nn.Module):
#     """
#     Self-supervised temporal reversal task:
#     - Randomly decide whether to reverse a subset of channels in time (last dimension).
#     - If selected, reverse a random subset containing more than 50% of channels.
#     - Predict reversal: 0 = unchanged, 1 = partially or fully reversed.
#     - Match the feature size of other pretext tasks (default: channel-averaged 3*200).
#     """
#     def __init__(self, input_dim: int):
#         super().__init__()
#         self.classifier = nn.Linear(in_features=input_dim, out_features=2)

#     def flip_all_or_not(self, x: torch.Tensor):
#         """
#         Arguments:
#             x: Tensor of shape (B, C, T), where T = seg * pts.
#         returns:
#             x_out: x with a subset of channels potentially reversed.
#             label: int if B == 1; otherwise a Python list[int] of length B.
#         """
#         assert x.dim() == 3, f"Expected x shape (B, C, T), got {tuple(x.shape)}"
#         B, C, T = x.shape
#         x_out = x.clone()
#         labels = []

#         # Compute the minimum channel count strictly greater than 50%.
#         min_k = C // 2 + 1  # e.g., C=64 -> 33, C=63 -> 32

#         for i in range(B):
#             if random.random() < 0.5:
#                 # Sample the channel count k from [min_k, C].
#                 k = random.randint(min_k, C)
#                 # Sample k channel indices without replacement on the same device as x.
#                 idx = torch.randperm(C, device=x_out.device)[:k]
#                 # Reverse only these channels along the time dimension.
#                 x_out[i, idx] = torch.flip(x_out[i, idx], dims=[-1])  # See torch.flip documentation.
#                 labels.append(1)
#             else:
#                 labels.append(0)

#         return x_out, (labels[0] if B == 1 else labels)

# CBraMod/models/reverse.py
import random
import torch
import torch.nn as nn

class ReversePretext(nn.Module):
    """
    Self-supervised temporal reversal task:
    - Randomly decide whether to reverse all channels of each sample in time (last dimension).
    - Predict reversal: 0 = unchanged, 1 = reversed.
    - Match the feature size of other pretext tasks (default: channel-averaged 3*200).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(in_features=input_dim, out_features=2)

    def flip_all_or_not(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor of shape (B, C, T), where T = seg * pts.
        returns:
            x_out: reversed x
            label: int if B == 1; otherwise a Python list[int] of length B.
        """
        assert x.dim() == 3, f"Expected x shape (B, C, T), got {tuple(x.shape)}"
        B, C, T = x.shape
        x_out = x.clone()
        labels = []
        for i in range(B):
            if random.random() < 0.5:
                # Reverse the time dimension.
                x_out[i] = torch.flip(x_out[i], dims=[-1])
                labels.append(1)
            else:
                labels.append(0)
        return x_out, (labels[0] if B == 1 else labels)

