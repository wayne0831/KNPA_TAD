###########################################################################################################
# import libraries
###########################################################################################################

import os
from src.config import *
from src.TAD.data_preprocess import *
from src.TAD.model import *
from src.TAD.result_analysis import *
from src.RL_model import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from sklearn.preprocessing import StandardScaler  

###########################################################################################################
# 이상탐지 모델 학습 및 저장
###########################################################################################################

print('==============Load training set for training anomaly detection model==============')
# 저장된 training set 스케일링
train_data = generate_dataset_siheung_sim(data_path=DATA_PATH[TAD_VER]['tr'])

train_data_scl = scale_data(data        = train_data, 
                            data_type   = 'tr', 
                            scaler_path = PICKLE_PATH['TAD']['scaler_stat'],
                            group_col   = ['LINK_ID', 'LANE_NO'],
                            input_cols  = INPUT_COLS)

#train_set, _ = load_dataset(csv_path=train_data, seq_len=SEQ_LEN, stride=STRIDE)
train_set, _ = load_dataset(train_data_scl, seq_len=SEQ_LEN, stride=STRIDE)
print('==============Data loaded!==============')


# 이상탐지 모델 학습
print('==============Train anomaly detection model==============')
model = MTSTAutoencoder(input_dim   = INPUT_DIM,
                        d_model     = D_MODEL, 
                        n_heads     = N_HEADS, 
                        seq_len     = SEQ_LEN, 
                        resolutions = RESOLUTIONS, 
                        n_layers    = N_LAYERS)

train(model     = model,
      dataset   = train_set,
      epochs    = EPOCH,
      lr        = LR,
      base_dim  = BASE_DIM,
      pkl_save_path = PICKLE_PATH['TAD']['tr_loss_stat'])
print('==============Model trained!==============')

# 이상탐지 모델 저장
print('==============Save anomaly detection model==============')
os.makedirs(os.path.dirname(CHK_PATH['TAD']), exist_ok=True)
torch.save(model.state_dict(), CHK_PATH['TAD'])
print(f"('============== Model saved to {CHK_PATH['TAD']} ==============")

###########################################################################################################
# 링크 및 차로 조합별 이상 임계치 계산
###########################################################################################################

# 저장된 validation set 스케일링
print('==============Load validation sat for calcuulating anomaly threshold==============')
val_data = generate_dataset_siheung_sim(data_path=DATA_PATH[TAD_VER]['val'])

val_data_scl = scale_data(data        = val_data, 
                          data_type   = 'val', 
                          scaler_path = PICKLE_PATH['TAD']['scaler_stat'],
                          group_col   = ['LINK_ID', 'LANE_NO'],
                          input_cols  = INPUT_COLS)

#val_set, val_meta = load_dataset(csv_path=val_data, seq_len=SEQ_LEN, stride=STRIDE)
val_set, val_meta = load_dataset(val_data_scl, seq_len=SEQ_LEN, stride=STRIDE)

# 이상 임계치 계산을 위한 재구성 오차 계산 
# 임계값은 0.0으로 설정하여 순수 복원 오차만 계산
print('==============Calculate reconstruction error on validation set...==============')
val_results = detect_anomalies(model         = model, 
                                dataset      = val_set, 
                                meta_df      = val_meta.copy(), 
                                threshold_df = None,
                                threshold    = 0.0, 
                                base_dim     = BASE_DIM)

print(f"==============Calculated {len(val_results)} reconstruction errors.==============")

# 3. 그룹별 임계값 계산 (get_group_thresholds 함수 사용)
# LINK_ID와 lane별 최대 복원 오차를 임계값으로 설정합니다.
print('==============Calculate group anomaly thresholds...==============')
group_thresholds = get_group_thresholds(val_results)

# 4. 임계값 저장
threshold_path =  PICKLE_PATH['TAD']['anomaly_threshold'] 
with open(threshold_path, 'wb') as f:
    pickle.dump(group_thresholds, f)

print(f"==============Group anomaly thresholds saved to {threshold_path}==============")
