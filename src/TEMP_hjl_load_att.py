import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============ 경로 설정 ============
att_normal = r"./data/siheung_normal.att"
att_event  = r"./data/siheung_anomaly.att"
map_path   = r"./data/siheung_map_info.xlsx"
save_dir   = r"./data/"

start_time = datetime(2025, 7, 1, 0, 0, 0)

header = [
    "SIMRUN", "TIMEINT", "DATACOLLECTIONMEASUREMENT",
    "ACCELERATION(ALL)", "DIST(ALL)", "LENGTH(ALL)", "VEHS(ALL)", "PERS(ALL)",
    "QUEUEDELAY(ALL)", "SPEEDAVGARITH(ALL)", "SPEEDAVGHARM(ALL)", "OCCUPRATE(ALL)"
]
agg_cols = ['VEHS(ALL)', 'SPEEDAVGARITH(ALL)', 'OCCUPRATE(ALL)']

def get_time_interval(hour):
    if 0 <= hour <= 7: return 0
    elif 8 <= hour <= 9: return 1
    elif 10 <= hour <= 17: return 2
    elif 18 <= hour <= 19: return 3
    elif 20 <= hour <= 23: return 4
    return -1

def convert_att_to_df(att_path):
    with open(att_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('$DATACOLLECTIONMEASUREMENTEVALUATION'):
            data_start = i + 1
            break
    else:
        raise ValueError("데이터 시작 지점을 찾을 수 없습니다.")
    data_lines = lines[data_start:]
    data_rows = [line.strip().split(";") for line in data_lines if line.strip() and not line.startswith("*")]
    df = pd.DataFrame(data_rows, columns=header)
    return df

def preprocess(df, mapping_df):
    # agg_cols 중에서 퍼센트가 섞여 있는 컬럼 리스트 (필요에 따라 추가)
    percent_cols = ['OCCUPRATE(ALL)']

    for col in agg_cols:
        # 1) 문자열 타입으로 만든 뒤, '%'와 공백 제거
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
        # 2) 숫자로 변환 (coerce → NaN), 빈 문자열은 NaN 처리
        df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
        # 3) 퍼센트 컬럼이면 100으로 나눠서 비율로 변환
        if col in percent_cols:
            df[col] = df[col] / 100

    # 나머지 결측치는 0으로 채우기
    df[agg_cols] = df[agg_cols].fillna(0)

    # 이하 기존 로직 그대로…
    df['TIMEINT'] = df['TIMEINT'].astype(str)
    df['start_sec'] = df['TIMEINT'].str.extract(r'(\d+)-').astype(int)
    df['date'] = df['start_sec'].apply(lambda x: start_time + timedelta(seconds=x))
    df['DAY'] = df['date'].dt.day
    df['TimeInterval'] = df['date'].dt.hour.apply(get_time_interval)

    df['DATACOLLECTIONMEASUREMENT'] = pd.to_numeric(
        df['DATACOLLECTIONMEASUREMENT'], errors='coerce'
    ).astype('Int64')
    df = df.merge(mapping_df, how='left',
                  left_on='DATACOLLECTIONMEASUREMENT',
                  right_on='현재 레인 ID')
    df = df[~df['DATACOLLECTIONMEASUREMENT'].isin(
        mapping_df[mapping_df['변환 링크 ID'].isna()]['현재 레인 ID']
    )]

    df.rename(columns={'변환 링크 ID': 'LINK_ID', '레인번호': 'lane'}, inplace=True)
    return df.dropna(subset=['LINK_ID', 'lane'])


def apply_event_labels(df, start_time):
    df = df.copy()
    df['pred'] = 0
    df['elapsed_sec'] = (df['date'] - start_time).dt.total_seconds().astype(int)

    # 반으로 줄일 컬럼, 두 배로 늘릴 컬럼
    half_cols   = ["VEHS(ALL)", "SPEEDAVGARITH(ALL)"]
    double_cols = ["OCCUPRATE(ALL)"]

    # 안전한 숫자형 강제 변환
    for col in set(half_cols + double_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    event_rules = [
        {'day': 1, 'start': 33300,  'end': 35100,  'link_id': 51,  'lane': [4,5], 'pred': 1},
        {'day': 2, 'start': 119700, 'end': 120600, 'link_id': 164, 'lane': 2,     'pred': 1},
        {'day': 3, 'start': 220500, 'end': 221400, 'link_id': 113, 'lane': 2,     'pred': 1},
        {'day': 4, 'start': 321300, 'end': 323100, 'link_id': 136, 'lane': [3,4], 'pred': 1},
        {'day': 5, 'start': 346500, 'end': 432900, 'link_id': 'ALL','lane': 'ALL','pred': 1},
        {'day': 6, 'start': 461700, 'end': 468900, 'link_id': 42,  'lane': 1,     'pred': 1},
        {'day': 7, 'start': 580500, 'end': 581400, 'link_id': 67,  'lane': 2,     'pred': 1},
    ]

    for rule in event_rules:
        cond = (
            (df['DAY'] == rule['day']) &
            df['elapsed_sec'].between(rule['start'], rule['end'])
        )
        if rule['link_id'] != 'ALL':
            cond &= (df['LINK_ID'] == rule['link_id'])
        if rule['lane'] != 'ALL':
            if isinstance(rule['lane'], (list, tuple, set)):
                cond &= df['lane'].isin(rule['lane'])
            else:
                cond &= (df['lane'] == rule['lane'])

        # pred 할당
        df.loc[cond, 'pred'] = rule['pred']

        # 반으로 줄이기
        for col in half_cols:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                df.loc[cond, col] = df.loc[cond, col] * 0.5

        # 두 배로 늘리기
        for col in double_cols:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                df.loc[cond, col] = df.loc[cond, col] * 2

    return df

def apply_bigo_mapping(df):
    has_note = df['비고'].notna()
    df_with_note = df[has_note].copy()
    df_with_note[agg_cols] = df_with_note[agg_cols].fillna(0)
    df_with_note['LINK_ID'] = df_with_note['비고']

    group_cols = ['date', 'lane', 'LINK_ID']
    df_grouped_mean = df_with_note.groupby(group_cols, as_index=False)[agg_cols].mean()
    df_grouped_mean['VEHS(ALL)'] = df_grouped_mean['VEHS(ALL)'].round().astype(int)
    meta_cols = ['DAY', 'TimeInterval', 'pred']
    df_grouped_meta = df_with_note.groupby(group_cols, as_index=False).first()[group_cols + meta_cols]
    df_note_final = pd.merge(df_grouped_mean, df_grouped_meta, on=group_cols)

    df_without_note = df[~has_note][['date', 'LINK_ID', 'DAY', 'TimeInterval', 'lane'] + agg_cols + ['pred']]
    return pd.concat([df_note_final, df_without_note], ignore_index=True)

# ✅ One-hot encoding 함수
def one_hot_encode_time_interval(df):
    one_hot = pd.get_dummies(df['TimeInterval'], prefix='TimeInt')
    df = pd.concat([df.drop(columns=['TimeInterval']), one_hot], axis=1)
    return df

# ======================== 실행 ========================
print(".att 파일 로딩")
df_normal = convert_att_to_df(att_normal)
df_event  = convert_att_to_df(att_event)


mapping_df = pd.read_excel(map_path, sheet_name="변환")
adj_df = pd.read_excel(map_path, sheet_name="인접")

print("전처리 수행 중")
df_normal_cleaned = preprocess(df_normal, mapping_df)
df_event_cleaned = preprocess(df_event, mapping_df)

print(df_normal_cleaned.head())

"""
print("pred 라벨링 적용 중")
df_normal_cleaned['pred'] = 0
df_event_labeled = apply_event_labels(df_event_cleaned, start_time)

print("링크 그룹 통합 중")
df_normal_final = apply_bigo_mapping(df_normal_cleaned)
df_event_final = apply_bigo_mapping(df_event_labeled)

# ✅ One-hot encoding 적용
print("One-hot encoding 적용 중 (TimeInterval)")
df_train = df_normal_final[df_normal_final['DAY'] <= 5]
df_val   = df_normal_final[df_normal_final['DAY'] > 5]
df_test  = df_event_final

df_train = one_hot_encode_time_interval(df_train)
df_val   = one_hot_encode_time_interval(df_val)
df_test  = one_hot_encode_time_interval(df_test)

from sklearn.preprocessing import StandardScaler  # ← 추가

scaler = StandardScaler()
# 1) train 으로 fit + transform
df_train[agg_cols] = scaler.fit_transform(df_train[agg_cols])

# 2) val/test 는 transform 만
df_val[agg_cols]   = scaler.transform(df_val[agg_cols])
df_test[agg_cols]  = scaler.transform(df_test[agg_cols])

# 저장
train_path = os.path.join(save_dir, "converted_train_0812.csv")
val_path = os.path.join(save_dir, "converted_val_0812.csv")
test_path = os.path.join(save_dir, "converted_test_0812.csv")

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"\n✅ 전체 처리 완료!\n📁 저장 경로:\n- {train_path}\n- {val_path}\n- {test_path}")
"""