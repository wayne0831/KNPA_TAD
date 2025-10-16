###########################################################################################################
# import libraries
###########################################################################################################

import os
from config import *
from TAD_data_preprocess import *
from TAD_model import *
from TAD_result_analysis import *
from RL_model import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

if PIPELINE['is_train']: 
    # 데이터 로드
    print('==============Data load for TAD==============')
    train_set, _ = load_dataset(csv_path=DATA_PATH[TAD_VER]['tr'], seq_len=SEQ_LEN, stride=STRIDE)
    print('==============Data loaded!==============')

    # TODO: training set scaling하고 training set의 mean/std을 pickle로 저장해야함
    # data_scaling 함수 구현 -> TAD_data_preprocess.py에 구현
    # pickle로 저장해야만 TAD_test.py에서 validation/test/infer set에 적용가눙 

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