import torch
import torch.nn as nn

class OptionTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: probability of profitable trade
        )
        self.hold_regressor_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: predicted hold days
        )

    def forward(self, x):
        x = self.feature_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        profitability = self.classifier_head(x)
        hold_days = self.hold_regressor_head(x)
        return profitability.squeeze(), hold_days.squeeze()
