# -*- coding: utf-8 -*-
"""
Attention-Based Multiple Instance Learning (ABMIL) Implementation

This script contains an incomplete implementation of ABMIL for computational pathology.
Complete the implementation to make the model functional.
"""

import torch
import torch.nn as nn


class Attention_TanhSigmoidGating(nn.Module):
    def __init__(self, D=64, L=64, dropout=0.25):
        r"""
        Global attention pooling layer with tanh non-linearity and sigmoid gating
        (Ilse et al. 2018).
        """
        super(Attention_TanhSigmoidGating, self).__init__()
        if D <= 0 or L <= 0:
            raise ValueError("D and L must be positive integers")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.tanhV = nn.Sequential(
            nn.Linear(D, L),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.sigmU = nn.Sequential(
            nn.Linear(D, L),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.w = nn.Linear(L, 1)

    def forward(self, H):
        if not isinstance(H, torch.Tensor):
            raise TypeError("H must be a torch.Tensor")
        if H.dim() != 2:
            raise ValueError("H must have shape [M, D]")
        if H.size(0) == 0:
            raise ValueError("H must contain at least one instance")

        A = self.tanhV(H) * self.sigmU(H)
        scores = self.w(A)
        scores = scores - scores.max(dim=0, keepdim=True).values
        A_norm = torch.softmax(scores, dim=0)
        return A_norm


class ABMIL(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=64, dropout=0.25, n_classes=2):
        r"""
        Attention-Based Multiple Instance Learning (Ilse et al. 2018).
        """
        super(ABMIL, self).__init__()
        if input_dim <= 0 or hidden_dim <= 0 or n_classes <= 0:
            raise ValueError("input_dim, hidden_dim, and n_classes must be positive integers")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.inst_level_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_attn = Attention_TanhSigmoidGating(D=hidden_dim, L=hidden_dim, dropout=dropout)
        self.bag_level_classifier = nn.Linear(hidden_dim, n_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

    def forward(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")
        if X.dim() != 2:
            raise ValueError("X must have shape [M, D]")
        if X.size(0) == 0:
            raise ValueError("Bag must contain at least one instance")
        if X.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {X.size(1)}")

        H = self.inst_level_fc(X)
        A_norm = self.global_attn(H)
        M = torch.sum(A_norm * H, dim=0, keepdim=True)
        logits = self.bag_level_classifier(M)
        return logits, A_norm


def basic_test():
    """Basic test to verify implementation completeness."""
    print("Running basic ABMIL test...")

    torch.manual_seed(42)
    X = torch.randn(50, 320)
    model = ABMIL(input_dim=320, hidden_dim=64)
    model.eval()

    with torch.no_grad():
        logits, attention = model(X)

    print("✓ Forward pass completed successfully!")
    print(f"Logits shape: {logits.shape}, Attention shape: {attention.shape}")
    return True


if __name__ == "__main__":
    basic_test()
