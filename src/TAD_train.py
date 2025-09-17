###########################################################################################################
# import libraries
###########################################################################################################

import os
from config import *
from TAD_model import *
from TAD_result_analysis import *
from RL_model import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###########################################################################################################
# load/preprcoess data
###########################################################################################################

############### TODO 데이터 load, 가공(scaling 포함), 저장: HJL 지원 필요

# 발견 에러
# 1. converted_test 7월1일부터 시작
## start_time 관련 작업 필요a

### 최종: scaling된 데이터셋 저장

###########################################################################################################
# train model
###########################################################################################################

if PIPELINE['is_train']: 
    # 데이터 로드
    print('==============Data load for TAD==============')
    train_set, _ = load_dataset(csv_path=DATA_PATH[TAD_VER]['tr'], seq_len=SEQ_LEN, stride=STRIDE)
    print('==============Data loaded!==============')

    # 모델 학습
    print('==============Model training==============')
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
          pkl_save_path=PICKLE_PATH['TAD']['tr_loss_stat'])
    
    print('==============Model trained!==============')

    # 모델 저장
    print('==============Model save==============')
    os.makedirs(os.path.dirname(CHK_PATH['TAD']), exist_ok=True)
    torch.save(model.state_dict(), CHK_PATH['TAD'])
    print(f"📁 Model saved to {CHK_PATH['TAD']}")
# end if