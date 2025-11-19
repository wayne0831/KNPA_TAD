###########################################################################################################
# import libraries
###########################################################################################################

import os
from config import *
from TAD_model import *
from TAD_result_analysis import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from RL_model import *  # run_q_learning, QTable 등

###########################################################################################################
# load/preprcoess data
###########################################################################################################

############### TODO 데이터 load, 가공(scaling 포함), 저장: HJL 지원 필요
# 발견 에러
# 1. converted_test 7월1일부터 시작
## start_time 관련 작업 필요
### 최종: scaling된 데이터셋 저장

###########################################################################################################
# utils
###########################################################################################################

def print_q_table(q_table):
    """(옵션) 학습된 Q 테이블 간단 출력"""
    print("\n=== Learned Q(s,a) ===")
    states, actions = q_table.states, q_table.actions
    header = "state \\ action".ljust(12) + " | " + " | ".join(a.ljust(8) for a in actions)
    print(header); print("-" * len(header))
    for s in states:
        row = s.ljust(12) + " | " + " | ".join(f"{q_table.Q[(s,a)]:+8.3e}" for a in actions)
        print(row)
    print("=" * len(header))


###########################################################################################################
# run modeling - model.py, result_analysis.py
###########################################################################################################

def main():
    # 1) 검증 데이터 로드 (μ,σ는 pickle에서 직접 로드하므로 train_set 불필요)
    print('============== Data load for RL ==============')
    val_set, val_meta = load_dataset(csv_path=DATA_PATH[TARGET_DATA]['val'], seq_len=SEQ_LEN, stride=STRIDE)

    # 2) Q-learning 실행 (ε 감쇠 스케줄 인자 전달)
    best_state_dict, q_table = run_q_learning(
        val_set        = val_set,
        tr_loss_stat_path= PICKLE_PATH['TAD']['tr_loss_stat'],
        q_table_path   = PICKLE_PATH['RL']['q_table'],
        states         = STATE,
        actions        = ACTIONS,
        val_batch_size = BATCH_SIZE,
        episodes       = EPISODES,
        budget_steps   = BUDGET_STEPS,
        # --- ε-greedy 감쇠 설정 (config 없으면 기본값으로) ---
        epsilon        = EPSILON,       # eps_decay="constant"일 때 사용
        eps_start      = EPS_START,
        eps_end        = EPS_END,
        eps_decay      = EPS_DECAY,     # 'exp' | 'linear' | 'constant'
        eps_k          = EPS_K,
        # ---------------------------------------------------
        alpha_q        = ALPHA_Q,
        gamma_q        = GAMMA_Q,
        lr_ft          = LR_FT,
        micro_steps    = MICRO_STEPS,
        allow_neg_reward = ALLOW_NEG_REWARD,
        seed           = SEED
    )

    # (옵션) Q-테이블 출력
    print_q_table(q_table)

    # 3) RL 체크포인트 저장
    os.makedirs(os.path.dirname(CHK_PATH['RL']), exist_ok=True)
    torch.save(best_state_dict, CHK_PATH['RL'])
    print(f"\n📁 RL-finetuned model saved to {CHK_PATH['RL']}")

if __name__ == "__main__":
    main()
