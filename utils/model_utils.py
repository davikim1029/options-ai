import torch
import pandas as pd
from transformers.transformer_model import OptionTransformer

FEATURE_COLUMNS = [
    "optionType", "strikePrice", "lastPrice", "bid", "ask", "bidSize", "askSize",
    "volume", "openInterest", "nearPrice", "inTheMoney", "delta", "gamma",
    "theta", "vega", "rho", "iv", "spread", "midPrice", "moneyness", "daysToExpiration"
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].fillna(0)

def prepare_labels(df: pd.DataFrame):
    # expected columns: 'target_action' (0=buy,1=sell,2=hold) and 'target_hold_days'
    if "target_action" not in df or "target_hold_days" not in df:
        raise ValueError("Training data must contain 'target_action' and 'target_hold_days'")
    return df["target_action"].values, df["target_hold_days"].values

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptionTransformer(feature_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device
