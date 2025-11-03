import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.model_core import OptionTransformerModel

MODEL_PATH = "model/option_transformer.pt"

def load_training_data(path="data/mock_option_data.csv"):
    df = pd.read_csv(path)

    features = df.drop(columns=["profitable", "optimal_hold_days"])
    target_profit = df["profitable"].astype(float)
    target_hold = df["optimal_hold_days"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    torch.save(scaler, "model/scaler.pkl")

    return (
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(target_profit.values, dtype=torch.float32),
        torch.tensor(target_hold.values, dtype=torch.float32),
    )

def train_model(data_path="data/mock_option_data.csv", epochs=25, batch_size=32):
    X, y_profit, y_hold = load_training_data(data_path)

    model = OptionTransformerModel(input_dim=X.shape[1])
    criterion_profit = nn.BCELoss()
    criterion_hold = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(X, y_profit, y_hold)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_profit, batch_hold in loader:
            optimizer.zero_grad()
            pred_profit, pred_hold = model(batch_X)
            loss_profit = criterion_profit(pred_profit, batch_profit)
            loss_hold = criterion_hold(pred_hold, batch_hold)
            loss = loss_profit + 0.5 * loss_hold
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")
