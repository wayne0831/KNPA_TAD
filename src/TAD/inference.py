# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 18:38:21 2025

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

from src.config import *
from src.TAD.data_preprocess import * 
from src.TAD.model import * 
from src.TAD.result_analysis import * 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt # 시각화를 위해 추가 (Confusion Matrix)
from sklearn.preprocessing import StandardScaler # scale_data 함수 내에서 사용됨을 가정

###########################################################################################################
# 새로 입력받는 데이터에 대해 이상탐지 수행
###########################################################################################################


def main():
    print("================== Load trained anomaly detection model ==================")
    
    # 이상탐지 모델 세팅값  로드
    model = MTSTAutoencoder(input_dim   = INPUT_DIM,
                            d_model     = D_MODEL, 
                            n_heads     = N_HEADS, 
                            seq_len     = SEQ_LEN, 
                            resolutions = RESOLUTIONS, 
                            n_layers    = N_LAYERS)
    
    # 저장된 이상탐지 모델 로드
    model_path = CHK_PATH['TAD']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval() # 평가 모드 설정
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    # 그룹별(링크 및 차로) 이상 임계값 로드
    threshold_path = PICKLE_PATH['TAD']['anomaly_threshold']
    try:
        with open(threshold_path, 'rb') as f:
            threshold_df = pickle.load(f)
        print(f"Anomaly thresholds loaded from {threshold_path}")
    except FileNotFoundError:
        print(f"Anomaly threshold file not found at {threshold_path}. Run validation for threshold calculation first.")
        return
    except Exception as e:
        print(f"Error loading thresholds from pickle: {e}. Check if the file is correctly saved as a DataFrame pickle.")
        return

    # 추론 실행 (inferenceReslt.csv 생성) - 실제 운영 레벨에서는 inference만 수행 가능
    run_inference(model         = model, 
                  base_dim      = BASE_DIM, 
                  threshold_df  = threshold_df, 
                  scaler_path   = PICKLE_PATH['TAD']['scaler_stat'], 
                  data_path_key = TAD_VER,
                  infer_type    = 'infer')
    
    #final_df = filter_by_domain(model=model, dataset, meta_df, base_dim=BASE_DIM)

    print("================== Inference Completed ==================")

if __name__ == '__main__':
    main() 

