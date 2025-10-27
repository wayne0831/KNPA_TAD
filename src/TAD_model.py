###########################################################################################################
# import libraries
###########################################################################################################

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from config import *
import pickle

###########################################################################################################
# set MTST
###########################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Transformer Block ─────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))

# ───── MTST Layer ─────
class MTSTLayer(nn.Module):
    def __init__(self, resolutions, input_dim, d_model, n_heads, seq_len):
        super().__init__()
        self.resolutions = resolutions
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                TransformerBlock(d_model, n_heads)
            ) for _ in resolutions
        ])

        total_flat_len = sum([(seq_len + r - 1) // r for r in resolutions]) * d_model
        self.linear_fusion = nn.Linear(total_flat_len, seq_len * input_dim)

    def forward(self, x):
        branch_outputs = []
        for i, r in enumerate(self.resolutions):
            x_down = x[:, ::r, :]
            x_proj = self.branches[i][0](x_down)
            x_attn = self.branches[i][1](x_proj)
            branch_outputs.append(x_attn.flatten(start_dim=1))
        x_concat = torch.cat(branch_outputs, dim=1)
        x_fused = self.linear_fusion(x_concat)
        return x_fused.view(x.size(0), self.seq_len, self.input_dim)

# ───── Autoencoder ─────
class MTSTAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, seq_len=30, resolutions=[1, 5, 10], n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            MTSTLayer(resolutions, input_dim, d_model, n_heads, seq_len)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.decoder(x)

# ───── Train ─────
def train(model, dataset, epochs=10, lr=1e-3, base_dim=5, pkl_save_path=PICKLE_PATH['TAD']['tr_loss_stat']):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device).train()

    epoch_losses = []  # DHY, 각 batch 평균 loss 기록용
    
    for ep in range(epochs):
        #total = 0
        batch_losses = [] # DHY
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            recon = model(x)
            loss = criterion(recon[:, :, :base_dim], y[:, :, :base_dim])  # 🔥 only original 5 features
            opt.zero_grad()
            loss.backward()
            opt.step()
            #total += loss.item()
            batch_losses.append(loss.item()) # DHY

        # DHY
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {ep+1}/{epochs} | Loss: {epoch_loss:.6f}")

        # DHY, loss 통계 계산 후 pickle로 저장
        if ep + 1 == epochs:
            tr_loss_stats = {
                "mean": float(np.mean(epoch_losses)),
                "std": float(np.std(epoch_losses)),
            }

            #print(tr_loss_stats)
            
            with open(pkl_save_path, "wb") as f:
                pickle.dump(tr_loss_stats, f)
            print(f"📁 Loss stats saved to {pkl_save_path}")

# ───── Dataset Loader ─────
def load_dataset(csv_path, seq_len=30, stride=15):
    df = pd.read_csv(csv_path)

    base_features = ['TRF_QNTY', 'AVG_SPD', 'OCPN_RATE']
    onehot_features = [col for col in df.columns if col.startswith('TimeInt_')]
    features = base_features + onehot_features

    for col in features:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.upper().map({'TRUE':1,'FALSE':0}).fillna(0)
    df[features] = df[features].astype(np.float32)

    X, y, meta = [], [], []
    grouped = df.groupby(['LINK_ID','lane'])

    for (_, _), group in grouped:    
        group = group.sort_values(by=['DAY', 'TIME']).reset_index(drop=True)
        vals = group[features].values
        pred_seq = group['pred'].values if 'pred' in group.columns else np.zeros(len(group))

        # <-- stride 적용: 0부터 끝까지, seq_len 단위로 윈도우를 stride 간격으로 이동
        for i in range(0, len(vals) - seq_len + 1, stride):
            X.append(vals[i:i+seq_len])
            y.append(vals[i:i+seq_len])
            row = group.iloc[i+seq_len-1].copy()
            row['pred'] = int(np.any(pred_seq[i:i+seq_len]))
            meta.append(row)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return TensorDataset(X, y), pd.DataFrame(meta).reset_index(drop=True)

###########################################################################################################
# set RL-MTST
###########################################################################################################

