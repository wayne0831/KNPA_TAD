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

###########################################################################################################
# set user defined funcitons
###########################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Detect Anomalies ─────
def detect_anomalies(model, dataset, meta_df, threshold=0.0, base_dim=5):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    all_recons = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            all_recons.append(recon.cpu().numpy())
    recons = np.concatenate(all_recons)
    targets = dataset.tensors[1].numpy()
    errors = np.mean((recons[:, :, :base_dim] - targets[:, :, :base_dim]) ** 2, axis=(1, 2))
    meta_df['recon_error'] = errors
    meta_df['anomaly'] = (errors > threshold).astype(int)
    return meta_df

# ───── Threshold ─────
# link + lane 조합별 이상치 임계치 계산
def get_group_thresholds(val_result_df):
    group_thresholds = val_result_df.groupby(['LINK_ID', 'lane'])['recon_error'].max().reset_index()
    group_thresholds.rename(columns={'recon_error': 'threshold'}, inplace=True)
    return group_thresholds

def apply_group_threshold(test_result_df, group_thresholds):
    merged = test_result_df.merge(group_thresholds, on=['LINK_ID', 'lane'], how='left')
    merged['anomaly'] = (merged['recon_error'] > merged['threshold']).astype(int)
    return merged

# ───── Average Compare & Domain Filter ─────
def filter_by_domain(model, dataset, meta_df, base_dim=3):
    """
    1) anomaly==1 인 윈도우만 대상
    2) 시퀀스 전체 평균 예측 vs 실제 계산
    3) 도메인 조건:
       avg_pred_VEHS  < avg_true_VEHS  AND
       avg_pred_SPEED < avg_true_SPEED AND
       avg_pred_OCC   > avg_true_OCC
    4) final_anomaly 플래그 추가
    """
    df = meta_df.copy().reset_index(drop=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds, trues = [], []

    model.to(device).eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            rec = model(x)
            preds.append(rec[:, :, :base_dim].mean(dim=1).cpu().numpy())
            trues.append(y[:, :, :base_dim].mean(dim=1).cpu().numpy())

    pred_avg = np.concatenate(preds, axis=0)
    true_avg = np.concatenate(trues, axis=0)

    df[['avg_pred_VEHS','avg_pred_SPEED','avg_pred_OCC']] = pred_avg
    df[['avg_true_VEHS','avg_true_SPEED','avg_true_OCC']] = true_avg

    cond = (
        (df['avg_pred_VEHS']  > df['avg_true_VEHS']) &
        (df['avg_pred_SPEED'] > df['avg_true_SPEED']) &
        (df['avg_pred_OCC']   < df['avg_true_OCC'])
    )
    df['final_anomaly'] = ((df['anomaly'] == 1) & cond).astype(int)
    return df

# ───── Aggregate Link+Time ─────
def aggregate_link_time(df, col='final_anomaly'):
    """
    같은 LINK_ID, date 그룹에서 col 을 max 집계
    """
    return (
        df
        .groupby(['LINK_ID','date'])[col]
        .max()
        .reset_index(name=f'link_time_{col}')
    )

# === END NEW ===