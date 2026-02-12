###########################################################################################################
# import libraries
###########################################################################################################

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:24:42 2025

@author: user
"""

from src.config import *
import pandas as pd
import numpy as np
import re
from datetime import timedelta, datetime
import os
import pickle
from sklearn.preprocessing import StandardScaler  

###########################################################################################################
# set user-defined functions
###########################################################################################################

def split_dataset_siheung_sim(data_path, infer=False, seq_len=30, tr_ratio=0.7, val_ratio=0.2, te_ratio=0.1, event_rules=None, start_time=None):
    """
    시흥시 시뮬레이션 교통 데이터셋 분할(Tr, Val, Te, Infer)  함수
    """
    
    # 데이터 경로 설정
    raw_path    = DATA_PATH[data_path]['raw']
    tr_path     = DATA_PATH[data_path]['tr']
    val_path    = DATA_PATH[data_path]['val']
    te_path     = DATA_PATH[data_path]['te']
    infer_path  = DATA_PATH[data_path]['infer']

    print(f"Raw 데이터 로드: {raw_path}")
    df = pd.read_csv(raw_path) 
    
    # Text 타입의 TOT_DT datetime 타입으로의 변환
    df['TOT_DT'] = pd.to_datetime(df['TOT_DT'], errors='coerce')
    
    # 데이터셋 분할 및 저장
    group_by_cols = [col for col in GRP_COLS if col != 'TOT_DT']

    if infer:
        print("추론용 데이터 추출 모드...")
        infer_df_list = []
        groups = df.groupby(group_by_cols) 
        
        for _, group in groups:
            group_sorted = group.sort_values(by='TOT_DT') 
            latest_data = group_sorted.tail(seq_len)
            infer_df_list.append(latest_data)
            
        infer_df = pd.concat(infer_df_list)
        infer_df.to_csv(infer_path, index=False)
        print(f"추론용 데이터 저장 완료: {infer_path}")

    else:
        print("데이터 분할 모드...")
        # 시계열 순서대로 분할하기 위해 정렬 확인
        df = df.sort_values(by='TOT_DT')
        
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

def generate_dataset_siheung_sim(data_path):
    """
    시흥시 시뮬레이션 교통 데이터 전처리 함수(TimeInterval 인코딩, 컬럼 정렬)
    """

    df = pd.read_csv(data_path)

    # Text 타입의 TOT_DT datetime 타입으로의 변환
    df['TOT_DT'] = pd.to_datetime(df['TOT_DT'], errors='coerce')
    
    # 'pred' 열 생성
    df['pred'] = 0 
    
    # TimeInterval 함수 정의
    def _get_time_interval(hour):
        if    0 <= hour <= 7:  return 0
        elif  8 <= hour <= 9:  return 1
        elif 10 <= hour <= 17: return 2
        elif 18 <= hour <= 19: return 3
        elif 20 <= hour <= 23: return 4
        return -1
    
    # TimeInterval 열 생성 (.dt 접근자 사용 가능)
    df['TimeInterval'] = df['TOT_DT'].dt.hour.apply(_get_time_interval)
    
    all_intervals = [0, 1, 2, 3, 4]
    
    # 범주형 변환 및 더미 변수 생성
    df['TimeInterval'] = pd.Categorical(df['TimeInterval'], categories=all_intervals)
    df = pd.get_dummies(df, columns=['TimeInterval'], prefix='TimeInt')

    # 최종 컬럼 정리
    final_timeint_cols = [col for col in df.columns if 'TimeInt' in col]
    final_cols = GRP_COLS + INPUT_COLS + final_timeint_cols + ['pred']
    
    # 컬럼 순서 재배열 및 중복 제거
    df_fin = df.reindex(columns=final_cols, fill_value=0).drop_duplicates()
    
    return df_fin

# 스케일링 코드(1119)
def scale_data(data, data_type, scaler_path, group_col = ['LINK_ID', 'LANE_NO'], input_cols = INPUT_COLS):
    
    """
    데이터셋을 그룹별로 스케일링하고 스케일링된 DataFrame을 반환합니다.
    'tr' 타입일 경우 그룹별 통계를 pickle로 저장합니다.    
    """
    #original_csv_path = data_path
    #df = pd.read_csv(original_csv_path)
    df = data
    df_scaled = df.copy()

    if data_type == 'tr':
         print('  - Calculating group-wise StandardScaler for training set...')
         group_scalers = {}
         
         for group_name, group_df in df.groupby(group_col):
             scaler = StandardScaler()
             # StandardScaler 객체에 통계(mean, std)를 fit하고 객체 저장
             # 2차원 배열 형태로 전달 (values)
             scaler.fit(group_df[input_cols].values) 
             group_scalers[group_name] = scaler
             
             # 스케일링 적용
             df_scaled.loc[group_df.index, input_cols] = scaler.transform(group_df[input_cols].values)
         # 스케일러 저장
         #os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
         with open(scaler_path, 'wb') as f:
             pickle.dump(group_scalers, f)
         print(f"  - Group Scalers (dict of StandardScaler) successfully SAVED to {scaler_path}")   
    
    else:
        # 2. Validation/Test/Infer Set: 저장된 StandardScaler 로드 및 적용
        try:
            with open(scaler_path, 'rb') as f:
                group_scalers = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Error: Scaler file not found at {scaler_path}. Run with data_type='tr' first!")
        print(f"  - Group Scalers loaded from {scaler_path}")
        
        # 스케일링 적용
        for group_name, group_df in df.groupby(group_col):
            if group_name in group_scalers:
                scaler = group_scalers[group_name]
                # 스케일링 (transform 사용)
                df_scaled.loc[group_df.index, input_cols] = scaler.transform(group_df[input_cols].values)
            else:
                print(f"⚠️ Warning: Group {group_name} not found in training scalers. Skipping scaling for this group.")

    # 스케일링된 데이터는 파일로 저장하지 않고 리턴
    return df_scaled


def preprocess_siheung_real(data_path, infer=False, seq_len=30, tr_ratio=0.7, val_ratio=0.2, te_ratio=0.1, event_rules=None, start_time=None):
    """
    데이터 전처리 함수.
    """
    
    raw_path = DATA_PATH[data_path]['raw']
    tr_path = DATA_PATH[data_path]['tr']
    val_path = DATA_PATH[data_path]['val']
    te_path = DATA_PATH[data_path]['te']
    infer_path = DATA_PATH[data_path]['infer']

    print(f"데이터 로드: {raw_path}")
    df = pd.read_csv(raw_path, encoding='cp949')
    
    # 🚨 수정: 열 이름의 앞뒤 공백을 제거하고, 'ACRS_ID' 대신 'ACSR_ID'를 사용합니다.
    df.columns = df.columns.str.strip() 

    # TODO: NODE_ID + ACSR_ID -> LINK_ID 생성(수정 완료)
    df['LINK_ID'] = df['NODE_ID'].astype(str) + '_' + df['ACSR_ID'].astype(str)
    
    # -----------------------------------------------------------
    # 1. TOT_DT Datetime 변환 및 교체
    # -----------------------------------------------------------
    time_parts = df['TOT_DT'].str.extract(r'(\d+-\d+-\d+)\s+(오전|오후)\s+(\d+):(\d+)')
    time_parts.columns = ['date_str', 'ampm', 'hour', 'minute']
    
    time_parts['hour'] = pd.to_numeric(time_parts['hour'])
    time_parts['minute'] = pd.to_numeric(time_parts['minute'])
    
    # 24시간 형식 시(Hour) 계산
    time_parts.loc[
        (time_parts['ampm'] == '오후') & (time_parts['hour'] < 12),
        'hour'
    ] += 12
    
    time_parts.loc[
        (time_parts['ampm'] == '오전') & (time_parts['hour'] == 12),
        'hour'
    ] = 0
    
    # 표준 Datetime 문자열 조합
    def expand_year(date_str):
        parts = date_str.split('-')
        return f"20{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    
    time_parts['DATE_STANDARD'] = time_parts['date_str'].apply(expand_year)
    time_parts['TIME_STANDARD'] = (
        time_parts['hour'].astype(str).str.zfill(2) + ':' +
        time_parts['minute'].astype(str).str.zfill(2) + ':00'
    )
    
    df['TOT_DT_DATETIME'] = pd.to_datetime(
        time_parts['DATE_STANDARD'] + ' ' + time_parts['TIME_STANDARD'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # 기존 TOT_DT 삭제 및 Datetime 열로 교체
    df = df.drop(columns=['TOT_DT'])
    df = df.rename(columns={'TOT_DT_DATETIME': 'TOT_DT'})
    
    # -----------------------------------------------------------
    # 2. 시간 정규화 및 누락 시간 채우기
    # -----------------------------------------------------------

    # 2.1. 시간 범위 결정 및 1분 간격 기준 시간표 생성
    min_dt = df['TOT_DT'].min()
    max_dt = df['TOT_DT'].max()

    time_index = pd.date_range(start=min_dt, end=max_dt, freq='1min')
    group_keys_base = df[['LINK_ID', 'LANE_NO']].drop_duplicates().reset_index(drop=True)
    
    master_index = pd.MultiIndex.from_product(
        [time_index, group_keys_base['LINK_ID'], group_keys_base['LANE_NO']],
        names=['TOT_DT', 'LINK_ID', 'LANE_NO']
    ).to_frame(index=False)
    
    # 2.2. 기존 데이터 집계 (중복 시간 처리)
    agg_funcs = {
        'TRF_QNTY': 'sum',
        'AVG_SPD': 'mean',
        'OCPN_RATE': 'mean'
    }

    aggregated_df = df.groupby(GRP_COLS)[INPUT_COLS].agg(agg_funcs).reset_index()

    # 2.3. 기준 테이블과 집계 데이터 병합 (Outer Merge)
    final_df = pd.merge(
        master_index, 
        aggregated_df, 
        on=GRP_COLS, 
        how='left'
    )

    # 2.4. 누락된 교통량 데이터 (NaN)를 0으로 채우기
    final_df[INPUT_COLS] = final_df[INPUT_COLS].fillna(0)
    
    # 최종 데이터프레임을 df에 할당
    df = final_df

    # -----------------------------------------------------------
    # 3. TimeInterval, pred 생성 및 더미 변수화
    # -----------------------------------------------------------
    
    # 'pred' 열 생성
    df['pred'] = 0 
    
    # TimeInterval 함수 정의
    def get_time_interval(hour):
        if    0 <= hour <= 7:  return 0
        elif  8 <= hour <= 9:  return 1
        elif 10 <= hour <= 17: return 2
        elif 18 <= hour <= 19: return 3
        elif 20 <= hour <= 23: return 4
        return -1
    
    # TimeInterval 열 생성
    df['TimeInterval'] = df['TOT_DT'].dt.hour.apply(get_time_interval)
    
    all_intervals = [0, 1, 2, 3, 4]
    
    # 범주형 변환 및 더미 변수 생성
    df['TimeInterval'] = pd.Categorical(df['TimeInterval'], categories=all_intervals)
    df = pd.get_dummies(df, columns=['TimeInterval'], prefix='TimeInt')

    # -----------------------------------------------------------
    # 4. 최종 컬럼 정리
    # -----------------------------------------------------------
    
    final_timeint_cols = [col for col in df.columns if 'TimeInt' in col]
    final_cols = GRP_COLS + INPUT_COLS + final_timeint_cols + ['pred']
    
    # 최종 컬럼만 남기고 순서 재배열, 누락된 컬럼 0 채우기, 중복 행 제거
    df = df.reindex(columns=final_cols, fill_value=0).drop_duplicates()
    
    # -----------------------------------------------------------
    # 5. 데이터 분할 및 저장 (요청하신 최종 로직)
    # -----------------------------------------------------------
    
    # 정렬 기준 컬럼 설정 (TOT_DT가 Datetime 타입으로 정렬 기준이 됨)
    group_by_cols = [col for col in GRP_COLS if col != 'TOT_DT'] # ['LINK_ID', 'LANE_NO']

    if infer:
        print("추론용 데이터 추출 모드...")
        infer_df_list = []
        
        # LINK_ID, LANE_NO 별로 그룹화
        groups = df.groupby(group_by_cols) 
        
        for _, group in groups:
            # TOT_DT 기준으로 정렬 (이미 정규화 단계에서 정렬되어 있을 가능성이 높지만 안전하게 재정렬)
            group_sorted = group.sort_values(by='TOT_DT') 
            latest_data = group_sorted.tail(seq_len)
            infer_df_list.append(latest_data)
            
        infer_df = pd.concat(infer_df_list)
        infer_df.to_csv(infer_path, index=False)
        print(f"추론용 데이터 저장 완료: {infer_path}")


    else:
        print("데이터 분할 모드...")
        
        # 시계열 순서대로 순차 분할
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

