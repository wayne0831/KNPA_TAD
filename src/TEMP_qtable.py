import pickle, math
import pandas as pd

class QTableStub:
    def __init__(self):
        self.__dict__ = {}
    def __setstate__(self, state):
        self.__dict__.update(state)

class RedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'RL_model' and name == 'QTable':
            return QTableStub
        return super().find_class(module, name)

with open('./pickle/RL/q_table_0904_v1.pkl', 'rb') as f:
    stub = RedirectUnpickler(f).load()

d = getattr(stub, '__dict__', {})
print("keys:", d.keys())  # 디버그용

# Case A: states/actions/Q가 모두 들어있는 경우 (당신 케이스)
if all(k in d for k in ('states','actions','Q')):
    states  = d['states']
    actions = d['actions']
    Q       = d['Q']            # {(state, action): value}

# Case B: Q만 있는 경우 → states/actions를 Q의 키로부터 유도
elif 'Q' in d and isinstance(d['Q'], dict):
    Q = d['Q']
    states  = sorted({s for (s, _) in Q.keys()})
    actions = sorted({a for (_, a) in Q.keys()})

else:
    raise ValueError(f"알 수 없는 구조입니다: keys={list(d.keys())}")

# 행렬로 변환
df = (
    pd.DataFrame(
        [(s, a, Q.get((s, a), math.nan)) for s in states for a in actions],
        columns=['state', 'action', 'value']
    )
    .pivot(index='state', columns='action', values='value')
    .reindex(index=states, columns=actions)  # 순서 보장
)

# 보기 좋게 출력(소수 6자리, NaN은 공란)
print(df.to_string(float_format=lambda x: f"{x: .6f}"))

# 필요 시 저장
# df.to_csv('q_table_matrix.csv', encoding='utf-8-sig')
