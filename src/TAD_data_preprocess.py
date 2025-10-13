


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:24:42 2025

@author: user
"""

from config import *
import pandas as pd
import numpy as np
import re
from datetime import timedelta, datetime
import os
import locale

def preprocess(data_path, infer=False, seq_len=30, tr_ratio=0.7, val_ratio=0.2, te_ratio=0.1, event_rules=None, start_time=None):
    """
    데이터 전처리 함수.

    Args:
        data_path (str): DATA_PATH 딕셔너리에서 사용할 데이터셋의 키.
        infer (bool): 추론 모드 여부.
        seq_len (int): 추론 모드일 때 추출할 데이터의 개수.
        tr_ratio (float): 학습 데이터 비율.
        val_ratio (float): 검증 데이터 비율.
        te_ratio (float): 테스트 데이터 비율.
        event_rules (list): 이벤트 라벨링을 위한 규칙 리스트.
        start_time (datetime): 이벤트 라벨링에 필요한 시작 시간.
    """
    
    try:
        locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')
    except locale.Error:
        locale.setlocale(locale.LC_TIME, 'Korean_Korea.949')

    raw_path = DATA_PATH[data_path]['raw']
    tr_path = DATA_PATH[data_path]['tr']
    val_path = DATA_PATH[data_path]['val']
    te_path = DATA_PATH[data_path]['te']
    infer_path = DATA_PATH[data_path]['infer']

    col_mapping = {
        '집계 일시': 'TIME',
        'NODE_ID': 'NODE_ID',
        'ACSR_ID': 'LINK_ID',
        'LANE_NO': 'lane',
        'TRF_QNTY': 'TRF_QNTY',
        'AVG_SPD': 'AVG_SPD',
        'OCPN_RATE': 'OCPN_RATE',
    }
    
    input_cols = ['TRF_QNTY', 'AVG_SPD', 'OCPN_RATE']
    group_cols = ['NODE_ID', 'LINK_ID', 'lane']
    
    print(f"데이터 로드: {raw_path}")
    df = pd.read_csv(raw_path, encoding='cp949')
    
    df = df.rename(columns=col_mapping)
    
    # 1. 정규표현식을 사용한 날짜/시간 파싱 로직
    time_parts = df['TIME'].str.extract(r'(오전|오후)\s+(\d+):(\d+):(\d+)')
    time_parts.columns = ['ampm', 'hour', 'minute', 'second']
    
    time_parts['hour'] = pd.to_numeric(time_parts['hour'])
    
    time_parts.loc[time_parts['ampm'] == '오후', 'hour'] += 12
    time_parts.loc[(time_parts['ampm'] == '오전') & (time_parts['hour'] == 12), 'hour'] = 0
    time_parts.loc[(time_parts['ampm'] == '오후') & (time_parts['hour'] == 24), 'hour'] = 12

    df['temp_dt'] = pd.to_datetime(
        time_parts['hour'].astype(str).str.zfill(2) + ':' +
        time_parts['minute'].astype(str).str.zfill(2) + ':' +
        time_parts['second'].astype(str).str.zfill(2),
        format='%H:%M:%S',
        errors='coerce'
    )
    
    # 2. DAY와 TIME 컬럼 생성
    is_midnight_start = (df['temp_dt'].shift(-1).dt.hour == 0) & (df['temp_dt'].dt.hour == 23) & (df['temp_dt'].dt.minute >= 55)
    
    df['is_new_day'] = is_midnight_start.astype(int)
    df['DAY'] = df['is_new_day'].cumsum() + 1
    df.drop('is_new_day', axis=1, inplace=True)
    
    df['TIME'] = df['temp_dt'].dt.strftime('%H:%M:%S')

    df.dropna(subset=['temp_dt'], inplace=True)
    df.drop('temp_dt', axis=1, inplace=True)
    
    if df.empty:
        print("[경고] 데이터 전처리 후 DataFrame이 비어있습니다. 원본 데이터의 '집계 일시' 컬럼을 확인하세요.")
        return

    df[input_cols] = df[input_cols].fillna(0)
    
    df['pred'] = 0
    
    def get_time_interval(hour):
        if    0 <= hour <= 7:  return 0
        elif  8 <= hour <= 9:  return 1
        elif 10 <= hour <= 17: return 2
        elif 18 <= hour <= 19: return 3
        elif 20 <= hour <= 23: return 4
        return -1

    df['TimeInterval'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.hour.apply(get_time_interval)
    df = pd.get_dummies(df, columns=['TimeInterval'], prefix='TimeInt')
    
    final_cols = ['DAY', 'TIME'] + input_cols + [col for col in df.columns if 'TimeInt' in col] + ['pred']
    df = df.reindex(columns=group_cols + final_cols, fill_value=0).drop_duplicates()
    
    if infer:
        print("추론용 데이터 추출 모드...")
        infer_df_list = []
        groups = df.groupby(group_cols)
        for _, group in groups:
            group_sorted = group.sort_values(by=['DAY', 'TIME'])
            latest_data = group_sorted.tail(seq_len)
            infer_df_list.append(latest_data)
        infer_df = pd.concat(infer_df_list)
        infer_df.to_csv(infer_path, index=False)
        print(f"추론용 데이터 저장 완료: {infer_path}")
    else:
        print("데이터 분할 모드...")
        
        # ⭐ 수정된 부분: 정렬 로직 제거
        
        total_rows = len(df)
        tr_end = int(total_rows * tr_ratio)
        val_end = tr_end + int(total_rows * val_ratio)
        
        tr_df = df.iloc[:tr_end]
        val_df = df.iloc[tr_end:val_end]
        te_df = df.iloc[val_end:]
        
        tr_df.to_csv(tr_path, index=False)
        val_df.to_csv(val_path, index=False)
        te_df.to_csv(te_path, index=False)
        
        print(f"학습 데이터 저장 완료: {tr_path}")
        print(f"검증 데이터 저장 완료: {val_path}")
        print(f"테스트 데이터 저장 완료: {te_path}")
preprocess('SIHEUNG_REAL', infer=False, seq_len=30, tr_ratio=0.7, val_ratio=0.2, te_ratio=0.1, event_rules=None, start_time=None)
