###########################################################################################################
# import libraries
###########################################################################################################

import os

###########################################################################################################
# set configurations
###########################################################################################################

# 대상 데이터: SIHEUNG_SIM, SIHEUNG_REAL
TAD_VER = 'SIHEUNG_REAL' 
# RL_VER  = '0904_v1'

###########################################################################################################
# set path configurations
###########################################################################################################

# path
DATA_PATH = { # 데이터셋 경로
    'SIHEUNG_SIM': { # 시흥 시뮬레이션 데이터
        'raw':   './data/siheung_sim/raw/SIHEUNG_SIM_RAW.csv',
        'tr':    './data/siheung_sim/train/SIHEUNG_SIM_TR.csv', 
        'val':   './data/siheung_sim/valid/SIHEUNG_SIM_VAL.csv',
        'te':    './data/siheung_sim/test/SIHEUNG_SIM_TE.csv',
        'infer': './data/siheung_sim/inference/SIHEUNG_SIM_INFER.csv',
    },
    'SIHEUNG_REAL': { # 시흥 스마트교차로 데이터
        'raw':   './data/siheung_real/raw/SIHEUNG_REAL_RAW.csv',  
        'tr':    './data/siheung_real/train/SIHEUNG_REAL_TR.csv', 
        'val':   './data/siheung_real/valid/SIHEUNG_REAL_VAL.csv',
        'te':    './data/siheung_real/test/SIHEUNG_REAL_TE.csv',
        'infer': './data/siheung_real/inference/SIHEUNG_REAL_INFER.csv',
        'adj':   './data/siheung_real/adjacency/SIHEUNG_REAL_ADJ.csv' # 260506 추가: 인접링크 정보 (링크ID, 인접링크ID)
    },
    # 'PANGYO_REAL': { # 판교 스마트교차로 데이터
    #     'raw':   './data/pangyo_real/raw/pangyo_14days_raw.csv',   
    #     'tr':    './data/pangyo_real/train/pangyo_14days_tr.csv', 
    #     'val':   './data/pangyo_real/valid/pangyo_14days_val.csv',
    #     'te':    './data/pangyo_real/test/pangyo_14days_te.csv',
    #     'infer': './data/pangyo_real/inference/pangyo_14days_infer.csv',
    # }
}

PICKLE_PATH = { # pickle 경로
    # 모델 training loss 평균/표준편차
    'TAD': {
        'tr_loss_stat': f'./pickle/TAD/tr_loss_stat_{TAD_VER}.pkl', 
        'scaler_stat': f'./pickle/TAD/scaler_stat_{TAD_VER}.pkl',
        'anomaly_threshold': f'./pickle/TAD/anomaly_threshold{TAD_VER}.pkl'
    },
    # 'RL': {
    #     'q_table': f'./pickle/RL/q_table_{RL_VER}.pkl', 
    # }
}

RES_PATH = { # 모델 예측 결과 경로
    'TAD': { # 이상탐지 모델
        'te_res': f'./result/TAD/test/testResult_{TAD_VER}.csv',
        'infer_res': f'./result/TAD/inference/inferenceResult_{TAD_VER}.csv',
    }, 
    # 'RL': { # 강화학습 모델
    #     'tr':       './result/RL/train/result_train.csv',
    #     'val':      './result/RL/valid/result_val.csv',
    #     'te':       './result/RL/test/result_test.csv',
    #     'infer':    './result/RL/inference/result_infer.csv',
    # }
}

# 모델 오브젝트 경로
CHK_PATH = {
    'TAD': f'./checkpoint/TAD/checkpoint_TAD_{TAD_VER}.pt',  # 이상탐지
    #'RL':  f'./checkpoint/RL/checkpoint_RL_{RL_VER}.pt',   # 강화학습
}

###########################################################################################################
# set data configurations
###########################################################################################################

## model input features
GRP_COLS   = ['TOT_DT', 'LINK_ID', 'LANE_NO']
INPUT_COLS = ['TRF_QNTY', 'AVG_SPD', 'OCPN_RATE']

## model output features
# 테스트결과: 모델 운영에는 활용안함
TE_RES_COLS  = []

# 추론결과: 시간, 링크ID, 차로번호, 차로이상임계치, 차로재구성오차, 차로이상여부, 링크이상여부
INFER_RES_COLS = ['TOT_DT', 'LINK_ID', 'LANE_NO', 'LANE_ANOMALY_THR', 'LANE_REC_ERROR', 'LANE_DET_RES', 'LINK_DET_RES']

###########################################################################################################
# set model configurations
###########################################################################################################

# 수집주기에 따라 하이퍼파라미터 조절 필요
# MTST hyperparameters
SEQ_LEN     = 6
STRIDE      = 3
INPUT_DIM   = 8
D_MODEL     = 64
N_HEADS     = 4
RESOLUTIONS = [1, 5, 10]
N_HEADS     = 4
N_LAYERS    = 2
EPOCH       = 10 # 10
LR          = 1e-3
BASE_DIM    = 3 # 3 base features (TRF_QNTY, AVG_SPD, OCPN_RATE)
BATCH_SIZE  = 32

# # RL hyperparameters
# #STATE       = ['NI_0', '0_1', '1_2', '2_3', '3_PI']
# STATE       = ['NI_0', '0_1.5', '1.5_3', '3_PI']
# ACTIONS     = ['FT_NONE', 'FT_ENC', 'FT_DEC', 'FT_ALL']
# EPISODES    = 100
# BUDGET_STEPS = None
# EPSILON     = 0.3
# EPS_START   = 0.3
# EPS_END     = 0.05
# EPS_DECAY   = 'exp'
# EPS_K       = 3.0
# ALPHA_Q     = 0.2
# GAMMA_Q     = 0.9
# LR_FT       = 5e-4
# MICRO_STEPS = 1 
# ALLOW_NEG_REWARD = True
# SEED        = 42

###########################################################################################################
# set pipeline configurations
###########################################################################################################