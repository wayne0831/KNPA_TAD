###########################################################################################################
# import libraries
###########################################################################################################

import os

###########################################################################################################
# set configurations
###########################################################################################################

# 대상 데이터: SIHEUNG_SIM, SIHEUNG_REAL, PANGYO_REAL
TAD_VER = 'SIHEUNG_SIM' 
RL_VER  = '0904_v1' ## No

###########################################################################################################
# set path configurations
###########################################################################################################

# path
DATA_PATH = { # 데이터셋 경로
    # raw(원시): data scaling 이전 
    # tr(학습), val(검증), te(테스트), infer(추론): data scaling 이후 TODO: 논의 필요
    'SIHEUNG_SIM': { # 시흥 시뮬레이션 데이터
        'raw':   '../data/siheung_sim/raw/siheung_14days_sim.csv',   
        'tr':    '../data/siheung_sim/train/converted_train.csv', 
        'val':   '../data/siheung_sim/valid/converted_val.csv',
        'te':    '../data/siheung_sim/test/converted_test.csv',
        'infer': '../data/siheung_sim/inference/converted_infer.csv',
    },
    'PANGYO_REAL': { # 판교 스마트교차로 데이터
        'raw':   '../data/pangyo_real/raw/pangyo_14days_raw.csv',   
        'tr':    '../data/pangyo_real/train/pangyo_14days_tr.csv', 
        'val':   '../data/pangyo_real/valid/pangyo_14days_val.csv',
        'te':    '../data/pangyo_real/test/pangyo_14days_te.csv',
        'infer': '../data/pangyo_real/inference/pangyo_14days_infer.csv',
    },
    'SIHEUNG_REAL': { # 시흥 스마트교차로 데이터
        'raw':   '../data/siheung_real/raw/siheung_14days_raw.csv',   
        'tr':    '../data/siheung_real/train/siheung_14days_tr.csv', 
        'val':   '../data/siheung_real/valid/siheung_14days_val.csv',
        'te':    '../data/siheung_real/test/siheung_14days_te.csv',
        'infer': '../data/siheung_real/inference/siheung_14days_infer.csv',
    },
}

PICKLE_PATH = { # pickle 경로
    # 모델 training loss 평균/표준편차
    'TAD': {
        'tr_loss_stat': f'../pickle/TAD/tr_loss_stat_{TAD_VER}.pkl', 
    },
    'RL': {
        'q_table': f'../pickle/RL/q_table_{RL_VER}.pkl', 
    }
}

RES_PATH = { # 모델 예측 결과 경로
    'TAD': { # 이상탐지 모델
        'tr':       '../result/TAD/train/result_train.csv',
        'val':      '../result/TAD/valid/result_val.csv',
        'te':       '../result/TAD/test/result_test.csv',
        'infer':    '../result/TAD/inference/result_infer.csv',
        
        # TODO: 네이밍 변경 필요
        #'grp_thr':  '../result/TAD/group_threshold.csv',
        #'cmp_df':   '../result/TAD/compare_data.csv',
        #'agg_link': '../result/TAD/link_time_final.csv',
        'te_res': '../result/TAD/testResult.csv',
        'infer_res': '../result/TAD/inferenceResult.csv',
    }, 
    'RL': { # 강화학습 모델
        'tr':       '../result/RL/train/result_train.csv',
        'val':      '../result/RL/valid/result_val.csv',
        'te':       '../result/RL/test/result_test.csv',
        'infer':    '../result/RL/inference/result_infer.csv',
        
        # TODO 네이밍 변경 필요 / 필요한가?
        #'grp_thr':  './result/RL/group_threshold.csv',
        #'cmp_df':   './result/RL/compare_data.csv',
        #'agg_link': './result/RL/link_time_final.csv',
    }

}

# 모델 오브젝트 경로
CHK_PATH = {
    'TAD': f'../checkpoint/TAD/checkpoint_TAD_{TAD_VER}.pt',  # 이상탐지
    'RL':  f'../checkpoint/RL/checkpoint_RL_{RL_VER}.pt',   # 강화학습
}

# 인접행렬 경로
#MAP_PATH  = './data/siheung_map.csv'
#ADJ_PATH  = './data/siheung_adj.csv'

###########################################################################################################
# set data configurations
###########################################################################################################

# data preprocess input
#MELT_COLS  = ['TOT_DT', 'LINK_ID', 'LANE_NO']
#PIVOT_COL  = 'TOT_DT'

# model input features
INPUT_COLS = ['TRF_QNTY', 'AVG_SPD', 'OCPN_RATE']

TE_RES_COLS    = []
INFER_RES_COLS = []

###########################################################################################################
# set model configurations
###########################################################################################################

# TODO: 수집주기에 따라 하이퍼파라미터 조절 필요
# MTST hyperparameters
SEQ_LEN     = 30
STRIDE      = 15
INPUT_DIM   = 8
D_MODEL     = 64
N_HEADS     = 4
RESOLUTIONS = [1, 5, 10]
N_HEADS     = 4
N_LAYERS    = 2
EPOCH       = 10 # 10
LR          = 1e-3
BASE_DIM    = 5
BATCH_SIZE  = 32

# RL hyperparameters
#STATE       = ['NI_0', '0_1', '1_2', '2_3', '3_PI']
STATE       = ['NI_0', '0_1.5', '1.5_3', '3_PI']
ACTIONS     = ['FT_NONE', 'FT_ENC', 'FT_DEC', 'FT_ALL']
EPISODES    = 100
BUDGET_STEPS = None
EPSILON     = 0.3
EPS_START   = 0.3
EPS_END     = 0.05
EPS_DECAY   = 'exp'
EPS_K       = 3.0
ALPHA_Q     = 0.2
GAMMA_Q     = 0.9
LR_FT       = 5e-4
MICRO_STEPS = 1 
ALLOW_NEG_REWARD = True
SEED        = 42

###########################################################################################################
# set pipeline configurations
###########################################################################################################

# pipeline
PIPELINE = {
    #'save_dataset': True,

    # 모델 학습
    'is_train': True, # 평소에는 False, 2주에 한번씩 True

    # 테스트
    'is_test': True,              # 성능 테스트
    'visualize_conf_mat': True,  # confusion matrix 시각화
    'visualize_line_plot': False, # line plot 

    # 모델 추론
    'is_infer': True, # 계속 True

    # 강화학습
    'is_rl': True
}

