###########################################################################################################
# import libraries
###########################################################################################################

import os
from src.config import *
from src.TAD_data_preprocess import *
from src.TAD_model import *
from src.TAD_result_analysis import *
from src.RL_model import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 추가패키지
import pickle
from sklearn.preprocessing import StandardScaler  

# TAD_VER = 'SIHEUNG_REAL'
# PIPELINE = {
#     # 모델 학습
#     'is_train': True, 
# }
# PICKLE_PATH = { # pickle 경로
#     # 모델 training loss 평균/표준편차
#     'TAD': {
#         'tr_loss_stat': f'../pickle/TAD/tr_loss_stat_{TAD_VER}.pkl', 
#     },
#     'RL': {
#         'q_table': f'../pickle/RL/q_table_{RL_VER}.pkl', 
#     }
# }
# CHK_PATH = {
#     'TAD': f'../checkpoint/TAD/checkpoint_TAD_{TAD_VER}.pt',  # 이상탐지
#     'RL':  f'../checkpoint/RL/checkpoint_RL_{RL_VER}.pt',   # 강화학습
# }

###########################################################################################################
# load/preprcoess data
###########################################################################################################

# raw data불러와서 train/valid/test로 구분
# TODO: 여기서는 scaling 전 데이터가 폴더에 저장되어야 함
preprocess(data_path   = TAD_VER, 
           infer       = False, 
           seq_len     = SEQ_LEN, 
           tr_ratio    = 0.7, 
           val_ratio   = 0.2, 
           te_ratio    = 0.1, # 실제 운영시에는 testset 필요없음
           event_rules = None, 
           start_time  = None)

###########################################################################################################
# train model
###########################################################################################################

# 데이터 로드
print('==============Data load for TAD==============')
# TODO: training set scaling하고 training set의 mean/std을 pickle로 저장해야함
# data_scaling 함수 구현 -> TAD_data_preprocess.py에 구현
# pickle로 저장해야만 TAD_test.py에서 validation/test/infer set에 적용가눙 
scaler_path = 'standard_scaler_pickle.pkl'
train_data = scale_data(DATA_PATH[TAD_VER]['tr'], 'tr', scaler_path)


train_set, _ = load_dataset(train_data, seq_len=SEQ_LEN, stride=STRIDE)
print('==============Data loaded!==============')


# 모델 학습
print('==============Model training==============')
model = MTSTAutoencoder(input_dim   = INPUT_DIM,
                        d_model     = D_MODEL, 
                        n_heads     = N_HEADS, 
                        seq_len     = SEQ_LEN, 
                        resolutions = RESOLUTIONS, 
                        n_layers    = N_LAYERS)

model_path = CHK_PATH['TAD']

train(model     = model,
        dataset   = train_set,
        epochs    = EPOCH,
        lr        = LR,
        base_dim  = BASE_DIM,
        pkl_save_path=PICKLE_PATH['TAD']['tr_loss_stat'])

#if os.path.exists(model_path):
## 2. 파일이 존재하면 기존 모델의 상태(weights) 로드
#    print(f'✅ Found existing checkpoint at {model_path}. Loading model state...')
#
#    # torch.load를 사용하여 저장된 상태 딕셔너리를 로드
#    # map_location은 학습 환경에 따라 CPU 또는 GPU를 지정할 수 있습니다.
#    try:
#        model.load_state_dict(torch.load(model_path))
#        print('✅ Model state loaded successfully! Resuming training...')            # d
#    except RuntimeError as e:
#        print(f"⚠️ Error loading model state: {e}")
#        print("⚠️ Model architecture might have changed. Starting fresh training.")
#else:
#    # 3. 파일이 존재하지 않으면 새로 학습 시작
#    print(f'❌ No checkpoint found at {model_path}. Starting fresh training...')
#    train(model     = model,
#            dataset   = train_set,
#            epochs    = EPOCH,
#            lr        = LR,
#            base_dim  = BASE_DIM,
#            pkl_save_path=PICKLE_PATH['TAD']['tr_loss_stat'])

print('==============Model trained!==============')

# 모델 저장
print('==============Model save==============')
os.makedirs(os.path.dirname(CHK_PATH['TAD']), exist_ok=True)
torch.save(model.state_dict(), CHK_PATH['TAD'])
print(f"📁 Model saved to {CHK_PATH['TAD']}")

###########################################################################################################
# Calculate Group Thresholds using Validation Data
###########################################################################################################

# 1. Validation 데이터 로드
print('\n==============Loading Validation Data for Threshold Calculation==============')
val_data = scale_data(DATA_PATH[TAD_VER]['val'], 'val', scaler_path)
val_set, val_meta = load_dataset(val_data, seq_len=SEQ_LEN, stride=STRIDE)

# 2. 복원 오차 계산 (detect_anomalies 함수 사용)
# 임계값은 0.0으로 설정하여 순수 복원 오차만 계산합니다.
print('⚙️ Calculating Reconstruction Error on Validation Set...')
val_results = detect_anomalies(
    model, 
    val_set, 
    val_meta.copy(), 
    threshold=0.0, 
    base_dim=BASE_DIM
)
print(f"✅ Calculated {len(val_results)} reconstruction errors.")

# 3. 그룹별 임계값 계산 (get_group_thresholds 함수 사용)
# LINK_ID와 lane별 최대 복원 오차를 임계값으로 설정합니다.
print('⚙️ Calculating Group Thresholds...')
group_thresholds = get_group_thresholds(val_results)


# 4. 임계값 저장

threshold_path = 'threshold_pickle.pkl' # 경로 지정 필요
#os.makedirs(os.path.dirname(threshold_path), exist_ok=True)

# CSV 파일로 저장
group_thresholds.to_csv(threshold_path, index=False)
print(f"📁 Group thresholds saved to {threshold_path}")
print('=============================================================================')