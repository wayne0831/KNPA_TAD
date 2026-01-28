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
from src.config import *

###########################################################################################################
# set user defined funcitons
###########################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TAD_result_analysis.py 또는 상단에 정의되어야 하는 함수 (Device 오류 해결 포함 최종 수정)

# ───── Detect Anomalies ─────
def detect_anomalies(model, dataset, meta_df, threshold_df=None, threshold=0.0, base_dim=3):
    """
    모델을 사용하여 재구성 오류를 계산하고, threshold_df 또는 단일 threshold를 적용하여 이상 여부를 판정합니다.
    (CPU/CUDA 장치 일치 오류 수정)

    Args:
        model (nn.Module): 학습된 이상 탐지 모델.
        dataset (TensorDataset): 시퀀스 데이터.
        meta_df (pd.DataFrame): 시퀀스 메타 정보 (TOT_DT, LINK_ID, LANE_NO 등).
        threshold_df (pd.DataFrame, optional): LINK_ID 및 LANE_NO별 임계값을 포함하는 DataFrame.
        threshold (float): threshold_df가 제공되지 않을 때 사용할 단일 임계값. 기본값은 0.0.
        base_dim (int): 재구성 오류 계산에 사용할 특징의 차원 수.
    """
    # 1. 모델과 데이터를 같은 장치로 이동
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 모델을 명시적으로 device로 이동시키고 평가 모드 설정
    model.to(device).eval() 
    
    all_recons = []
    
    # 2. 모델 예측 (재구성 오류 계산)
    with torch.no_grad():
        # 데이터셋이 (x, y) 형태라고 가정하고 y도 로드
        # (TensorDataset의 tensors[0]은 x, tensors[1]은 y(target)라고 가정)
        for x, _ in loader: 
            # ⭐️⭐️ 핵심 수정: 입력 데이터 x를 device로 이동 ⭐️⭐️
            x = x.to(device) 
            
            # 모델 호출
            recon = model(x)
            all_recons.append(recon.cpu().numpy())
            
    recons = np.concatenate(all_recons)
    targets = dataset.tensors[1].numpy() 
    
    # 3. 재구성 오류 계산
    errors = np.mean((recons[:, :, :base_dim] - targets[:, :, :base_dim]) ** 2, axis=(1, 2))
    
    result_df = meta_df.copy().reset_index(drop=True)
    result_df['recon_error'] = errors
    
    # run_test에서 필요한 'pred' (True Anomaly Label) 처리
    if 'pred' not in result_df.columns:
        result_df['pred'] = 0 

    # 4. 임계값 적용 (유연한 로직 유지)
    if threshold_df is not None:
        threshold_col_name = 'threshold' if 'threshold' in threshold_df.columns else 'Thresholds'
        
        result_df = result_df.merge(
            threshold_df[['LINK_ID', 'LANE_NO', threshold_col_name]],
            on=['LINK_ID', 'LANE_NO'],
            how='left'
        )
        result_df['Thresholds_applied'] = result_df[threshold_col_name].fillna(threshold) 
        result_df['anomaly'] = (result_df['recon_error'] > result_df['Thresholds_applied']).astype(int)
    
    else:
        # 단일 임계값 적용 (예: Validation Reconstruction Error 계산 시)
        result_df['Thresholds_applied'] = threshold 
        result_df['anomaly'] = (result_df['recon_error'] > threshold).astype(int) 

    return result_df[['TOT_DT', 'LINK_ID', 'LANE_NO', 'recon_error', 'Thresholds_applied', 'anomaly', 'pred']]


# ───── Threshold ─────
# link + lane 조합별 이상치 임계치 계산
def get_group_thresholds(val_result_df):
    group_thresholds = val_result_df.groupby(['LINK_ID', 'LANE_NO'])['recon_error'].max().reset_index()
    group_thresholds.rename(columns={'recon_error': 'threshold'}, inplace=True)
    return group_thresholds

def apply_group_threshold(test_result_df, group_thresholds):
    merged = test_result_df.merge(group_thresholds, on= ['LINK_ID', 'LANE_NO'], how='left')
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