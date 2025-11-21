# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 12:56:42 2025

@author: user
"""

###########################################################################################################
# import libraries
###########################################################################################################
import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import TensorDataset

# 가정: src.config, src.TAD_data_preprocess, src.TAD_model, src.TAD_result_analysis 함수들이 import 됨
from src.config import *
from src.TAD_data_preprocess import * # create_sequence_dataset_from_df 함수 가정
from src.TAD_model import * 
from src.TAD_result_analysis import * # detect_anomalies, get_group_thresholds 함수 가정
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# =========================================================================================================
# CONFIG 변수 설정 및 경로 정의 (메인 실행 환경과 동일하게 가정)
# =========================================================================================================
TAD_VER = 'SIHEUNG_REAL'
PIPELINE = {
    'is_train': False,
    'is_test': True,
    'visualize_conf_mat': True,
    'visualize_line_plot': False,
    'is_infer': True,
    'is_rl': True
}

# CONFIG: 데이터 경로, 모델 체크포인트, 스케일러/임계값 pickle 경로
CHK_PATH = {'TAD': f'../checkpoint/TAD/checkpoint_TAD_{TAD_VER}.pt'}
PICKLE_PATH = {
    'TAD': {
        'scaler_stat': f'../pickle/TAD/scaler_stat_{TAD_VER}.pkl',
        'threshold': f'../pickle/TAD/threshold_{TAD_VER}.csv', # 임계값은 CSV로 저장했다고 가정
    }
}

RES_PATH = { # 모델 예측 결과 경로
    'TAD': { # 이상탐지 모델
        'tr':       './result/TAD/train/result_train.csv',
        'val':      './result/TAD/valid/result_val.csv',
        'te':       './result/TAD/test/result_test.csv',
        'infer':    './result/TAD/inference/result_infer.csv',
        
        # TODO: 네이밍 변경 필요
        #'grp_thr':  '../result/TAD/group_threshold.csv',
        #'cmp_df':   '../result/TAD/compare_data.csv',
        #'agg_link': '../result/TAD/link_time_final.csv',
        'te_res': './result/TAD/testResult.csv',
        'infer_res': './result/TAD/inferenceResult.csv',
    }, 
    'RL': { # 강화학습 모델
        'tr':       './result/RL/train/result_train.csv',
        'val':      './result/RL/valid/result_val.csv',
        'te':       './result/RL/test/result_test.csv',
        'infer':    './result/RL/inference/result_infer.csv',
        
        # TODO 네이밍 변경 필요 / 필요한가?
        #'grp_thr':  './result/RL/group_threshold.csv',
        #'cmp_df':   './result/RL/compare_data.csv',
        #'agg_link': './result/RL/link_time_final.csv',
    }

}


# DATA_PATH는 config에 정의되어 있고, 'te'와 'infer' 키가 있다고 가정합니다.
# DATA_PATH[TAD_VER]['te'] 와 DATA_PATH[TAD_VER]['infer'] 는 원본 CSV 파일 경로를 가리킵니다.
OUTPUT_DIR = '../result/TAD'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = CHK_PATH['TAD']
threshold_path = 'threshold_pickle.pkl'


# =========================================================================================================
# Helper Function: 링크 이상 여부 계산 (test와 infer 모두 사용)
# =========================================================================================================

def calculate_link_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    TOT_DT와 LINK_ID가 동일한 그룹 내에서 '이상여부'가 하나라도 1이면 '링크 이상 여부'를 1로 설정합니다.
    """
    # 1. 'TOT_DT'와 'LINK_ID'를 기준으로 그룹화하여 이상 판정(1)이 있는지 확인합니다.
    link_anomaly_check = df.groupby(['TOT_DT', 'LINK_ID'])['이상여부'].transform(lambda x: x.any()).astype(int)
    
    # 2. 결과를 '링크 이상 여부' 열에 반영합니다.
    df['링크 이상 여부'] = link_anomaly_check
    return df


# =========================================================================================================
# 1. Test 함수: testResult.csv 생성
# =========================================================================================================

def run_test(model, base_dim, threshold_df, scaler_path, data_path, test_type='te'):
    """
    테스트 데이터셋을 사용하여 모델 성능을 평가하고 testResult.csv를 생성합니다.
    """
    print(f"\n======== Running Test on {test_type.upper()} Data ========")
    
    # 1. 데이터 로드 및 스케일링
    # 스케일링되지 않은 원본 CSV를 로드하여 그룹별 스케일러를 적용합니다.
    test_df_scaled = scale_data(data_path=DATA_PATH[data_path]['te'], 
                                data_type='te', # 'tr'의 스케일러 사용
                                scaler_path=scaler_path)
    
    # 2. 시퀀스 데이터셋 생성 (create_sequence_dataset_from_df 함수 가정)
    test_set, test_meta = create_sequence_dataset_from_df(df=test_df_scaled, 
                                                         seq_len=SEQ_LEN, 
                                                         stride=STRIDE)

    # 3. 이상 탐지 수행 (복원 오차 계산 및 이상 판정)
    print("⚙️ Detecting anomalies...")
    # detect_anomalies 함수는 임계값 적용 후 '이상여부'를 포함한 결과를 반환한다고 가정
    test_results = detect_anomalies(
        model, 
        test_set, 
        test_meta.copy(), 
        threshold_df=threshold_df, # 그룹별 임계값 DataFrame 전달
        base_dim=base_dim
    )
    
    # 4. 결과 컬럼 추가 및 정리
    test_results['Thresholds'] = test_results['Thresholds_applied'] # 임계값 (Thresholds)
    test_results['recon_error'] = test_results['error']            # 복원 오차 (recon_error)
    test_results['이상여부'] = test_results['anomaly'].astype(int)  # 이상여부 (anomaly: 0 또는 1)
    test_results['실제 이상여부'] = test_results['pred'].astype(int) # 실제 이상여부 (pred)
    
    # 5. 정답 여부 계산
    test_results['정답 여부'] = (test_results['이상여부'] == test_results['실제 이상여부']).astype(int)
    
    # 6. 링크 이상 여부 계산
    test_results = calculate_link_anomaly(test_results)
    
    # 7. 최종 결과 DataFrame 정리
    final_cols = ['TOT_DT', 'LINK_ID', 'LANE_NO', 'Thresholds', 'recon_error', 
                  '이상여부', '링크 이상 여부', '실제 이상여부', '정답 여부']
    
    result_df = test_results[final_cols]
    
    # 8. CSV 파일 저장
    result_path = os.path.join(OUTPUT_DIR, 'testResult.csv')
    result_df.to_csv(result_path, index=False)
    print(f"✅ Test Results saved to {result_path}")
    
    # 9. 성능 지표 계산 및 출력 (시각화 옵션에 따라 Confusion Matrix 포함)
    y_true = result_df['실제 이상여부'].values
    y_pred = result_df['이상여부'].values
    
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    if PIPELINE['visualize_conf_mat']:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot()
        # 시각화 저장 (matplotlib을 사용하여 저장하는 로직이 추가되어야 함)
        # plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))


# =========================================================================================================
# 2. Inference 함수: inferenceReslt.csv 생성
# =========================================================================================================

def run_inference(model, base_dim, threshold_df, scaler_path, data_path_key, infer_type='infer'):
    """
    추론 데이터셋을 사용하여 이상 여부를 판정하고 inferenceReslt.csv를 생성합니다.
    """
    print(f"\n======== Running Inference on {infer_type.upper()} Data ========")
    
    # 1. 데이터 로드 및 스케일링
    infer_df_scaled = scale_data(data_path=DATA_PATH[data_path_key][infer_type], 
                                 data_type='val', # 'tr'의 스케일러 사용
                                 scaler_path=scaler_path)
    
    # 2. 시퀀스 데이터셋 생성 (create_sequence_dataset_from_df 함수 가정)
    infer_set, infer_meta = create_sequence_dataset_from_df(df=infer_df_scaled, 
                                                            seq_len=SEQ_LEN, 
                                                            stride=STRIDE)

    # 3. 이상 탐지 수행 (복원 오차 계산 및 이상 판정)
    print("⚙️ Detecting anomalies...")
    infer_results = detect_anomalies(
        model, 
        infer_set, 
        infer_meta.copy(), 
        threshold_df=threshold_df, # 그룹별 임계값 DataFrame 전달
        base_dim=base_dim
    )
    
    # 4. 결과 컬럼 추가 및 정리
    infer_results['Thresholds'] = infer_results['Thresholds_applied']
    infer_results['recon_error'] = infer_results['error']
    infer_results['이상여부'] = infer_results['anomaly'].astype(int)
    
    # 5. 링크 이상 여부 계산
    infer_results = calculate_link_anomaly(infer_results)
    
    # 6. 최종 결과 DataFrame 정리 (추론 결과는 정답 여부/실제 이상여부 제외)
    final_cols = ['TOT_DT', 'LINK_ID', 'LANE_NO', 'Thresholds', 'recon_error', 
                  '이상여부', '링크 이상 여부']
    
    result_df = infer_results[final_cols]
    
    # 7. CSV 파일 저장
    result_path = os.path.join(OUTPUT_DIR, 'inferenceReslt.csv')
    result_df.to_csv(result_path, index=False)
    print(f"✅ Inference Results saved to {result_path}")


# =========================================================================================================
# 메인 실행 로직
# =========================================================================================================

def main():
    
    # 1. 모델 로드 및 경로 설정
    if not PIPELINE['is_test'] and not PIPELINE['is_infer']:
        print("PIPELINE['is_test'] 또는 PIPELINE['is_infer']가 True여야 실행됩니다.")
        return

    print("================== Starting TAD Test/Inference ==================")
    
    # 모델 정의 및 체크포인트 로드
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

    # 2. 임계값 로드
    threshold_path = PICKLE_PATH['TAD']['threshold']
    try:
        threshold_df = pd.read_csv(threshold_path)
        print(f"✅ Thresholds loaded from {threshold_path}")
    except FileNotFoundError:
        print(f"❌ Threshold file not found at {threshold_path}. Run validation for threshold calculation first.")
        return

    # 3. 파이프라인 실행
    
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


# if __name__ == '__main__':
#     main()