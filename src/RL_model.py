# -*- coding: utf-8 -*-
"""
RL_model.py — Q-learning 기반으로 Validation 배치마다 '파인튜닝 실행 여부/범위'를 결정해
모델을 미세조정하는 모듈.

구성:
- 상태(STATES): ['NI_1','1_2','2_3','3_PF']  (train loss의 μ, σ로 정의한 구간)
- 행동(ACTIONS): ['FT_NONE','FT_DEC','FT_ALL']  (무FT / decoder만 FT / 전부 FT)
- 보상(REWARD):  Δbatch loss = Loss_before - Loss_after  (해당 배치의 재구성 MSE 감소량)

외부 의존:
- config.py: CHK_PATH, PICKLE_PATH, 하이퍼파라미터(SEQ_LEN 등)
- model(MTSTAutoencoder)
- load_tr_loss_stat(): PICKLE_PATH['tr_loss_stat']에서 μ, σ를 바로 로드 (예외 처리 없음)
"""

###########################################################################################################
# import libraries
###########################################################################################################

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from config import *
import pickle
import copy, random, numpy as np, torch
from typing import Dict, List, Tuple, Optional
from typing import Dict, List, Tuple, Optional
from TAD_model import MTSTAutoencoder
import torch.nn.functional as F

###########################################################################################################
# set user defined functions
###########################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: Optional[int] = None):
    """
    재현성을 위해 파이썬/넘파이/파이토치 시드를 고정한다.

    Args:
        seed (int|None): 고정할 시드. None이면 아무 작업도 하지 않음.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 완전한 재현성을 위해 아래 두 옵션을 비결정적으로 만드는 최적화를 끔
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_ft_by_action(model: nn.Module, action: str):
    """
    행동(action)에 따라 어떤 파라미터를 학습할지 스위칭한다.

    ACTIONS:
        - 'FT_NONE': 모든 파라미터 동결 (평가만)
        - 'FT_ENC': encoder만 학습
        - 'FT_DEC' : decoder만 학습
        - 'FT_ALL' : 모든 파라미터 학습

    Args:
        model (nn.Module): MTSTAutoencoder 인스턴스
        action (str): 위 ACTIONS 중 하나
    """
    # freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    if action == 'FT_NONE':
        pass
    elif action == 'FT_ENC': # fine-tune encoder (self.layers)
        for lyr in model.layers:
            for p in lyr.parameters():
                p.requires_grad = True
    elif action == 'FT_DEC': # fine-tune decoder parameters
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif action == 'FT_ALL': # fine-tune all parameters
        for p in model.parameters():
            p.requires_grad = True
    # end if


def make_optimizer(model: nn.Module, lr: float = 5e-4, wd: float = 0.0):
    """
    현재 requires_grad=True인 파라미터만 모아 Adam 옵티마이저를 만든다.

    Args:
        model (nn.Module): 학습 대상 모델
        lr (float): 학습률
        wd (float): weight decay

    Returns:
        torch.optim.Optimizer | None: 학습 파라미터가 없다면 None 반환
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=wd) if params else None


@torch.no_grad()
def batch_loss_eval(model: nn.Module, xb: torch.Tensor, base_dim: int = BASE_DIM) -> float:
    """
    배치 텐서 xb에 대해 재구성 MSE 손실을 계산(평가 모드)한다.

    Args:
        model (nn.Module): 평가할 모델
        xb (Tensor): shape (B, L, D) 배치 입력
        base_dim (int): 손실 계산에 사용할 입력 피처 차원 수 (앞쪽 BASE_DIM만 사용)

    Returns:
        float: 배치 평균 재구성 MSE
    """
    model.eval()
    rec = model(xb)
    
    # loss 계산: 전체 요소 평균 (B,L,D 모두 평균)
    loss = F.mse_loss(rec[:, :, :base_dim], xb[:,  :, :base_dim], reduction='mean')

    return float(loss.item())


def load_tr_loss_stat(path: str) -> Tuple[float, float]:
    """
    피클 파일에서 학습 손실 통계 (mu, std)를 '직접' 로드한다. 예외 처리 없음.

    허용 포맷:
        - {'mean': <float>, 'std': <float>}
        - {'loss': {'mean': <float>, 'std': <float>}}

    Args:
        path (str): PICKLE_PATH['tr_loss_stat']

    Returns:
        (mu, sigma) (float, float): 평균과 표준편차

    Raises:
        KeyError: 허용 포맷이 아닌 경우
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)

    def to_float(x):  # numpy scalar 등 안전 변환
        return float(np.array(x).astype('float64'))

    if isinstance(d, dict) and 'mean' in d and 'std' in d:
        mu, sigma = to_float(d['mean']), to_float(d['std'])
    elif isinstance(d, dict) and 'loss' in d and isinstance(d['loss'], dict):
        mu, sigma = to_float(d['loss']['mean']), to_float(d['loss']['std'])
    else:
        raise KeyError("train loss pickle must contain {'mean','std'} or {'loss':{'mean','std'}}")
    # end if

    # 표준편차 0/음수 보호
    if sigma <= 0:
        sigma = 1e-12
    return mu, sigma


def bin_state(batch_loss: float, mean: float, std: float) -> str:
    """
    배치 손실을 학습 통계 μ, σ 기준 구간으로 매핑해 상태(state)를 반환한다.

    STATES:
        - 'NI_1' : x < μ+1σ
        - '1_2'  : μ+1σ ≤ x < μ+2σ
        - '2_3'  : μ+2σ ≤ x < μ+3σ
        - '3_PF' : x ≥ μ+3σ

    Args:
        batch_loss (float): 현재 배치의 손실
        mean (float): 학습 손실 평균 μ
        std (float): 학습 손실 표준편차 σ

    Returns:
        str: 상태 라벨 중 하나
    """
    s = max(float(std), 1e-12)
    m = float(mean)
    
    # t0, t1, t2, t3 = m, m + s, m + 2*s, m + 3*s
    # x = float(batch_loss)
    # if x < t0:
    #     return 'NI_0'
    # if x < t1: 
    #     return '0_1'
    # elif x < t2: 
    #     return '1_2'
    # elif x < t3: 
    #     return '2_3'
    # else: 
    #     return '3_PI'

    t0, t1_5, t3 = m, m + 1.5*s, m + 3*s
    x = float(batch_loss)
    if x < t0:
        return 'NI_0'
    if x < t1_5: 
        return '0_1.5'
    elif x < t3: 
        return '1.5_3'
    else: 
        return '3_PI'

class QTable:
    """
    문자열 상태/행동 공간에 대한 간단한 Q(s,a) 테이블 구현.

    Attributes:
        states (List[str]): 상태 라벨 목록
        actions(List[str]): 행동 라벨 목록
        Q (Dict[(str,str), float]): (state, action) → Q값
    """
    def __init__(self, states: List[str], actions: List[str]):
        self.states  = list(states)
        self.actions = list(actions)
        self.Q: Dict[Tuple[str,str], float] = {(s,a): 0.0 for s in self.states for a in self.actions}

    def greedy_action(self, s: str) -> str:
        """
        주어진 상태 s에서 Q값이 최대인 행동을 반환한다(탐욕 정책).

        Args:
            s (str): 상태 라벨

        Returns:
            str: 행동 라벨
        """
        return max(self.actions, key=lambda a: self.Q[(s,a)])

    def update(self, s: str, a: str, r: float, s_next: str, alpha: float = 0.2, gamma: float = 0.9):
        """
        Q-learning 업데이트 식으로 Q(s,a)를 갱신한다.

        Q(s,a) ← (1-α)Q(s,a) + α[r + γ max_a' Q(s',a')]

        Args:
            s (str): 현재 상태
            a (str): 실행한 행동
            r (float): 즉시 보상
            s_next (str): 다음 상태
            alpha (float): 학습률 α
            gamma (float): 할인율 γ
        """
        best_next = max(self.Q[(s_next, a_next)] for a_next in self.actions)
        self.Q[(s,a)] = (1 - alpha)*self.Q[(s,a)] + alpha*(r + gamma*best_next)

        return None

def build_base_model() -> nn.Module:
    """
    config에 정의된 하이퍼파라미터로 MTSTAutoencoder를 생성해 반환한다.

    Returns:
        nn.Module: 초기화된 MTSTAutoencoder (아직 state_dict 로드 전)
    """
    model = MTSTAutoencoder(
        input_dim=INPUT_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        seq_len=SEQ_LEN, resolutions=RESOLUTIONS, n_layers=N_LAYERS
    ).to(device)
    return model

def get_epsilon(
    global_step: int,
    total_steps: int,
    *,
    eps_decay: str = "exp",   # "exp" | "linear" | "constant"
    epsilon: float = 0.2,     # constant일 때 사용
    eps_start: float = 0.30,  # exp/linear 시작값
    eps_end: float = 0.05,    # 바닥값
    eps_k: float = 3.0        # exp 감쇠 강도 (클수록 빨리 줄어듦)
) -> float:
    """
    0-based global_step과 총 스텝 수(total_steps)를 받아 ε를 반환한다.
    """
    T = max(int(total_steps), 1)
    gs = max(0, min(int(global_step), T - 1))  # [0, T-1]로 클램프

    dec = eps_decay.lower()
    if dec == "constant":
        return float(epsilon)

    if dec == "exp":
        r = gs / (T - 1 if T > 1 else 1)
        e = eps_end + (eps_start - eps_end) * np.exp(-eps_k * r)
    elif dec == "linear":
        e = eps_end + (eps_start - eps_end) * (1.0 - gs / (T - 1 if T > 1 else 1))
    else:
        raise ValueError("eps_decay must be one of {'exp','linear','constant'}")

    lo, hi = (min(eps_start, eps_end), max(eps_start, eps_end))

    return float(min(max(e, lo), hi))

def run_q_learning(
    val_set,
    tr_loss_stat_path,
    q_table_path,
    states: List[str],
    actions: List[str],
    val_batch_size: int = 32,
    episodes: int = 3,
    budget_steps: Optional[int] = None,
    # ----- ε-greedy 감쇠 설정 -----
    epsilon: float = 0.2,         # eps_decay="constant"일 때 사용되는 상수 ε
    eps_start: float = 0.30,      # 감쇠 시작값 (exp/linear에서 사용)
    eps_end: float = 0.05,        # 바닥값 (exp/linear에서 사용)
    eps_decay: str = "exp",       # "exp" | "linear" | "constant"
    eps_k: float = 3.0,           # exp 감쇠 강도(클수록 빨리 줄어듦)
    # ------------------------------
    alpha_q: float = 0.2,
    gamma_q: float = 0.9,
    lr_ft: float = 5e-4,
    micro_steps: int = 1,
    allow_neg_reward: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[dict, QTable]:

    set_seed(seed)

    # 1) base model load (기준 모델 1번만 생성+로드)
    base_model = build_base_model()
    base_state = torch.load(CHK_PATH['TAD'], map_location=device)
    base_model.load_state_dict(base_state)  # ← 기준 가중치
    base_model.to(device)

    # 2) μ, σ (pickle direct load)
    mu, sigma = load_tr_loss_stat(path=tr_loss_stat_path)

    # 3) validation batches
    idx_all = np.arange(len(val_set.tensors[0]))
    np.random.shuffle(idx_all)
    val_batches = [idx_all[i:i+val_batch_size] for i in range(0, len(idx_all), val_batch_size)]

    # 에피소드 스텝 상한
    if budget_steps is None:
        budget_steps = len(val_batches)
    budget_steps = min(budget_steps, len(val_batches))

    # ε 스케줄 총 스텝 수
    total_steps = max(episodes * budget_steps, 1)

    # 4) Q-learning
    Q = QTable(states=states, actions=actions)
    best_total_reward = -1e18
    best_state_dict   = copy.deepcopy(base_model.state_dict())

    for ep in range(episodes):
        print(f"\n[Episode {ep+1}/{episodes}]")

        # 기준 모델의 작업 사본으로 에피소드 시작
        model = copy.deepcopy(base_model).to(device)

        total_reward = 0.0
        order = np.random.permutation(len(val_batches))[:budget_steps]

        for step, bidx in enumerate(order, 1):
            batch_index = val_batches[bidx]
            xb = val_set.tensors[0][batch_index].to(device)

            # 글로벌 스텝(0-base) & 현재 ε (외부 get_epsilon 사용)
            gstep = ep * budget_steps + (step - 1)
            eps_now = get_epsilon(
                gstep, total_steps,
                eps_decay=eps_decay, epsilon=epsilon,
                eps_start=eps_start, eps_end=eps_end, eps_k=eps_k
            )

            # 상태: FT 전 배치 손실 → 구간화
            loss_before = batch_loss_eval(model, xb, BASE_DIM)
            s = bin_state(loss_before, mu, sigma)

            # ε-greedy 행동 선택
            a = random.choice(actions) if (random.random() < eps_now) else Q.greedy_action(s)

            # 행동 실행 (FT_NONE 제외 fine tuning)
            if a not in ('FT_NONE'):
                set_ft_by_action(model, a)
                opt = make_optimizer(model, lr=lr_ft)
                if opt is not None:
                    for _ in range(micro_steps):
                        model.train()
                        rec  = model(xb)
                        loss = ((rec[:, :, :BASE_DIM] - xb[:, :, :BASE_DIM])**2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()

            # 보상: Δbatch loss
            loss_after = batch_loss_eval(model, xb, BASE_DIM)
            r = (loss_before - loss_after) / sigma
            if not allow_neg_reward:
                r = max(r, 0.0)

            # Q 업데이트
            s_next = bin_state(loss_after, mu, sigma)
            Q.update(s, a, r, s_next, alpha=alpha_q, gamma=gamma_q)

            total_reward += r
            if step % 10 == 0 or step == budget_steps:
                print(f"  step {step:03d} | eps={eps_now:.3f} | state={s:>4} → act={a:<7} | Δloss={r:+.6e} | next={s_next:>4}")

        print(f"Episode reward sum = {total_reward:+.6e}")
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_state_dict   = copy.deepcopy(model.state_dict())
    # end for

    # save Q-table
    with open(q_table_path, "wb") as f:
        pickle.dump(Q, f)
    print(f"📁 Q table saved to {q_table_path}")

    return best_state_dict, Q
