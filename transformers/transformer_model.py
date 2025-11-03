# model/transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptionTransformer(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_layers=2, n_heads=4):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Two heads: one for classification, one for regression
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # Buy / Sell / Hold
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # Predicted hold days
        )

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        buy_sell_logits = self.class_head(x)
        hold_days = self.reg_head(x)
        return buy_sell_logits, hold_days
        
   