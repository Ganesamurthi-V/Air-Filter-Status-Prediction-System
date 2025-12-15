import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

# Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load & Normalize Data
df = pd.read_csv("power_per_deg_formatted.csv")
df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")


scaler = MinMaxScaler()
df['power_per_deg_norm'] = scaler.fit_transform(df[['power_per_deg']])


# joblib.dump(scaler, "scaler.pkl")

# Create Sequences
seq_len = 10
def create_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

X_np, y_np = create_sequences(df['power_per_deg_norm'].values, seq_len)
X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

# MixerBlock & TSMixer Definition
class MixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim=64):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len)
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        y = x + self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        y = y + self.channel_mixer(y)
        return y

class TSMixerRegressor(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.mixer = MixerBlock(seq_len)
        self.output_layer = nn.Linear(seq_len, 1)

    def forward(self, x):
        x = self.mixer(x)
        x = x.squeeze(-1)
        return self.output_layer(x).squeeze()

# Regression to determine best threshold
device = torch.device("cpu")
model_reg = TSMixerRegressor(seq_len=seq_len).to(device)
optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=0.001)
loss_fn_reg = nn.MSELoss()
epochs = 100
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(epochs):
    model_reg.train()
    epoch_loss = 0.0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer_reg.zero_grad()
        preds = model_reg(batch_X)
        loss = loss_fn_reg(preds, batch_y)
        loss.backward()
        optimizer_reg.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Get predictions and threshold
with torch.no_grad():
    predictions = model_reg(X_tensor.to(device)).cpu().numpy()
    true_vals = y_tensor.numpy()

threshold = np.quantile(true_vals, 0.7)
y_binary = (y_tensor > threshold).long()

# Optional F1 optimization
best_f1 = 0
for t in np.arange(0.1, 0.91, 0.01):
    y_pred_temp = (predictions > t).astype(int)
    f1 = f1_score((true_vals > threshold).astype(int), y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Classification Model
class TSMixerClassifier(nn.Module):
    def __init__(self, seq_len, num_blocks=3, hidden_dim=64):
        super().__init__()
        self.mixers = nn.Sequential(*[MixerBlock(seq_len, hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(seq_len, 1)

    def forward(self, x):
        x = self.mixers(x)
        x = x.squeeze(-1)
        return self.output_layer(x).squeeze()

# Focal Loss for imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Prepare classifier
dataset_cls = TensorDataset(X_tensor, y_binary)
loader_cls = DataLoader(dataset_cls, batch_size=32, shuffle=True)
model_cls = TSMixerClassifier(seq_len=seq_len).to(device)
optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=0.001)
loss_fn_cls = FocalLoss(alpha=0.5, gamma=2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cls, mode='min', patience=5, factor=0.5)

# Training
best_loss_cls = float('inf')
patience_cls = 10
patience_counter_cls = 0
best_model_state = None

for epoch in range(epochs):
    model_cls.train()
    total_loss = 0.0
    for batch_X, batch_y in loader_cls:
        batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)
        optimizer_cls.zero_grad()
        logits = model_cls(batch_X)
        loss = loss_fn_cls(logits, batch_y)
        loss.backward()
        optimizer_cls.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader_cls)
    scheduler.step(avg_loss)

    if avg_loss < best_loss_cls:
        best_loss_cls = avg_loss
        best_model_state = model_cls.state_dict()
        patience_counter_cls = 0
    else:
        patience_counter_cls += 1
        if patience_counter_cls >= patience_cls:
            break

if best_model_state:
    model_cls.load_state_dict(best_model_state)

# Evaluate
model_cls.eval()
with torch.no_grad():
    logits = model_cls(X_tensor.to(device))
    probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs > best_threshold).astype(int)
    y_true = y_binary.numpy()

# torch.save(model_cls.state_dict(), "best_tsmixer_model.pth")