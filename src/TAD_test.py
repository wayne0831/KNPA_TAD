# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 18:38:21 2025

@author: user
"""

###########################################################################################################
# test.py
###########################################################################################################

# import libraries
import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import TensorDataset

# 가정: 아래 모듈 및 변수들은 src 디렉토리에서 import 됩니다.
from src.config import *
from src.TAD_data_preprocess import * 
from src.TAD_model import * 
from src.TAD_result_analysis import * 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt # 시각화를 위해 추가 (Confusion Matrix)
from sklearn.preprocessing import StandardScaler # scale_data 함수 내에서 사용됨을 가정


# =========================================================================================================
# Helper Functions (TAD_data_preprocess.py 또는 TAD_result_analysis.py에 정의되어야 하나, 편의상 여기에 통합)
# =========================================================================================================

def calculate_link_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    TOT_DT와 LINK_ID가 동일한 그룹 내에서 'anomaly'가 하나라도 1이면 'link_anomaly'를 1로 설정합니다.
    """
    # 'TOT_DT'와 'LINK_ID'를 기준으로 그룹화하여 이상 판정(1)이 있는지 확인합니다.
    link_anomaly_check = df.groupby(['TOT_DT', 'LINK_ID'])['anomaly'].transform(lambda x: x.any()).astype(int)
    
    # 결과를 'link_anomaly' 열에 반영합니다.
    df['link_anomaly'] = link_anomaly_check
    return df


# =========================================================================================================
# 1. run_test 함수: testResult.csv 생성
# =========================================================================================================

def run_test(model, base_dim, threshold_df, scaler_path, data_path_key, test_type='te'):
    """
    테스트 데이터셋을 사용하여 모델 성능을 평가하고 testResult.csv를 생성합니다.
    """
    print(f"\n======== Running Test on {test_type.upper()} Data ========")
    
    # 1. 데이터 로드 및 스케일링 
    test_df_scaled = scale_data(data_path=DATA_PATH[data_path_key][test_type], 
                                data_type='te', 
                                scaler_path=scaler_path)
    
    # 2. 시퀀스 데이터셋 생성 
    test_set, test_meta = load_dataset(df=test_df_scaled, 
                                       seq_len=SEQ_LEN, 
                                       stride=STRIDE)

    # 3. 이상 탐지 수행
    print("⚙️ Detecting anomalies...")
    test_results = detect_anomalies(
        model, 
        test_set, 
        test_meta.copy(), 
        threshold_df=threshold_df, 
        base_dim=base_dim
    )
    
    # 4. 결과 칼럼명 변경 및 정리
    test_results['Thresholds'] = test_results['Thresholds_applied'] 
    test_results['recon_error'] = test_results['error']            
    test_results['anomaly'] = test_results['anomaly'].astype(int)  # 이상여부
    test_results['true_anomaly'] = test_results['pred'].astype(int) # 실제 이상여부 (pred)
    
    # 5. 정답 여부 계산
    test_results['is_correct'] = (test_results['anomaly'] == test_results['true_anomaly']).astype(int)
    
    # 6. 링크 이상 여부 계산
    test_results = calculate_link_anomaly(test_results)
    
    # 7. 최종 결과 DataFrame 정리
    final_cols = ['TOT_DT', 'LINK_ID', 'LANE_NO', 'Thresholds', 'recon_error', 
                  'anomaly', 'link_anomaly', 'true_anomaly', 'is_correct']
    
    result_df = test_results[final_cols]
    
    # 8. CSV 파일 저장
    result_df.to_csv(RES_PATH['TAD']['te_res'], index=False)
    print(f"✅ Test Results saved to {RES_PATH['TAD']['te_res']}")
    
    # 9. 성능 지표 계산 및 출력
    y_true = result_df['true_anomaly'].values
    y_pred = result_df['anomaly'].values
    
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    if PIPELINE['visualize_conf_mat']:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Test)')
        plt.savefig('confusion_matrix_test.png') # 경로 지정 필요
        print(f"✅ Confusion Matrix saved to {'confusion_matrix_test.png'}")


# =========================================================================================================
# 2. run_inference 함수: inferenceReslt.csv 생성
# =========================================================================================================

def run_inference(model, base_dim, threshold_df, scaler_path, data_path_key, infer_type='infer'):
    """
    추론 데이터셋을 사용하여 이상 여부를 판정하고 inferenceReslt.csv를 생성합니다.
    """
    print(f"\n======== Running Inference on {infer_type.upper()} Data ========")
    
    # 1. 데이터 로드 및 스케일링
    infer_df_scaled = scale_data(data_path=DATA_PATH[data_path_key][infer_type], 
                                 data_type='val', # 추론 시에는 validation 통계 사용
                                 scaler_path=scaler_path)
    
    # 2. 시퀀스 데이터셋 생성
    infer_set, infer_meta = load_dataset(df=infer_df_scaled, 
                                         seq_len=SEQ_LEN, 
                                         stride=STRIDE)

    # 3. 이상 탐지 수행
    print("⚙️ Detecting anomalies...")
    infer_results = detect_anomalies(
        model, 
        infer_set, 
        infer_meta.copy(), 
        threshold_df=threshold_df, 
        base_dim=base_dim
    )
    
    # 4. 결과 칼럼명 변경 및 정리
    infer_results['Thresholds'] = infer_results['Thresholds_applied']
    infer_results['recon_error'] = infer_results['error']
    infer_results['anomaly'] = infer_results['anomaly'].astype(int) # 이상여부
    
    # 5. 링크 이상 여부 계산
    infer_results = calculate_link_anomaly(infer_results)
    
    # 6. 최종 결과 DataFrame 정리 (추론 결과는 정답 여부/실제 이상여부 제외)
    final_cols = ['TOT_DT', 'LINK_ID', 'LANE_NO', 'Thresholds', 'recon_error', 
                  'anomaly', 'link_anomaly']
    
    result_df = infer_results[final_cols]
    
    # 7. CSV 파일 저장

    result_df.to_csv(RES_PATH['TAD']['infer_res'], index=False)
    print(f"✅ Inference Results saved to {RES_PATH['TAD']['infer_res']}")


# =========================================================================================================
# 메인 실행 로직 (main 함수)
# =========================================================================================================

def main():
    
    # 1. 실행 조건 확인
    if not PIPELINE['is_test'] and not PIPELINE['is_infer']:
        print("PIPELINE['is_test'] 또는 PIPELINE['is_infer']가 True여야 실행됩니다.")
        return

    print("================== Starting TAD Test/Inference ==================")
    
    # 2. 모델 로드
    model = MTSTAutoencoder(input_dim=INPUT_DIM, d_model=D_MODEL, 
                            n_heads=N_HEADS, seq_len=SEQ_LEN, 
                            resolutions=RESOLUTIONS, n_layers=N_LAYERS)
    
    model_path = CHK_PATH['TAD']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval() # 평가 모드 설정
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"❌ Model checkpoint not found at {model_path}. Please train the model first.")
        return

    # 3. 임계값 로드
    threshold_path = PICKLE_PATH['TAD']['threshold']
    try:
        # ⭐️⭐️ CSV 로드를 Pickle 로드로 변경 ⭐️⭐️
        with open(threshold_path, 'rb') as f:
            threshold_df = pickle.load(f)
        # ⭐️⭐️ -------------------------------- ⭐️⭐️
        print(f"✅ Thresholds loaded from {threshold_path}")
    except FileNotFoundError:
        print(f"❌ Threshold file not found at {threshold_path}. Run validation for threshold calculation first.")
        return
    except Exception as e:
        print(f"❌ Error loading thresholds from pickle: {e}. Check if the file is correctly saved as a DataFrame pickle.")
        return

    # 4. 파이프라인 실행
    
    # 테스트 실행 (testResult.csv 생성)
    if PIPELINE['is_test']:
        run_test(model=model, 
                 base_dim=BASE_DIM, 
                 threshold_df=threshold_df, 
                 scaler_path=PICKLE_PATH['TAD']['scaler_stat'], 
                 data_path_key=TAD_VER,
                 test_type='te')

    # 추론 실행 (inferenceReslt.csv 생성)
    if PIPELINE['is_infer']:
        run_inference(model=model, 
                      base_dim=BASE_DIM, 
                      threshold_df=threshold_df, 
                      scaler_path=PICKLE_PATH['TAD']['scaler_stat'], 
                      data_path_key=TAD_VER,
                      infer_type='infer')
        
    print("================== Test/Inference Completed ==================")


if __name__ == '__main__':
    main()