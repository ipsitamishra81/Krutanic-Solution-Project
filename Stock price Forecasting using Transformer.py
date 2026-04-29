import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# =========================
# LOAD DATA
# =========================
df = yf.download(
    "AAPL",
    start="2010-01-01",
    end="2024-01-01"
)

# Use ONLY Close price (best baseline)
data = df[["Close"]].values


# =========================
# SCALE DATA
# =========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(data, seq_len):
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    return np.array(X), np.array(y)


SEQ_LEN = 60   # increased context
X, y = create_sequences(data_scaled, SEQ_LEN)


# =========================
# TRAIN-TEST SPLIT
# =========================
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# =========================
# TENSORS
# =========================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)


# =========================
# TRANSFORMER MODEL (TUNED)
# =========================
class TransformerModel(nn.Module):

    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=3):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]   # last timestep
        return self.fc(x)


model = TransformerModel()


# =========================
# TRAINING
# =========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        preds = model(xb)
        loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")


# =========================
# EVALUATION
# =========================
model.eval()

with torch.no_grad():
    preds = model(X_test).numpy()

y_test_np = y_test.numpy()


# =========================
# INVERSE SCALING
# =========================
preds_actual = scaler.inverse_transform(preds)
y_actual = scaler.inverse_transform(y_test_np)


# =========================
# METRICS
# =========================
rmse = np.sqrt(np.mean((preds_actual - y_actual) ** 2))
print(f"RMSE: {rmse:.2f}")


# =========================
# PLOT
# =========================
plt.figure(figsize=(12, 6))

plt.plot(y_actual, label="Actual Price", linewidth=2)
plt.plot(preds_actual, label="Predicted Price", linestyle="--")

plt.title("Stock Price Prediction (Transformer - Tuned)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()

plt.show()