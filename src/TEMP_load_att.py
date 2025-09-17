###########################################################################################################
# import libraries
###########################################################################################################

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


###########################################################################################################
# set user defined funcitons
###########################################################################################################

def convert_att_to_df(att_path, header):
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

###########################################################################################################
# load and save file
###########################################################################################################

if __name__ == "__main__":
    print('load att file')

    header = [
        "SIMRUN", "TIMEINT", "DATACOLLECTIONMEASUREMENT",
        "ACCELERATION(ALL)", "DIST(ALL)", "LENGTH(ALL)", "VEHS(ALL)", "PERS(ALL)",
        "QUEUEDELAY(ALL)", "SPEEDAVGARITH(ALL)", "SPEEDAVGHARM(ALL)", "OCCUPRATE(ALL)"
    ]

    att_norm_path   = r'./data/siheung_normal.att'
    att_abnrom_path = r'./data/siheung_anomaly.att'
    map_path        = r'./data/siheung_map_info.xlsx'

    df_norm   = convert_att_to_df(att_path=att_norm_path, header=header)
    df_abnorm = convert_att_to_df(att_path=att_abnrom_path, header=header)
    df_map    = pd.read_excel(map_path, sheet_name="변환")
    df_adj    = pd.read_excel(map_path, sheet_name="인접")

    print('save csv file')
    df_all = pd.concat([df_norm, df_abnorm], axis=0, ignore_index=True)
    df_norm.to_csv('./data/siheung_14days.csv', index=False)

    #df_norm.to_csv('./data/siheung_normal.csv')
    #df_abnorm.to_csv('./data/siheung_anomaly.csv')
    df_map.to_csv('./data/siheung_map.csv', encoding="utf-8-sig", index=False)
    df_adj.to_csv('./data/siheung_adj.csv', encoding="utf-8-sig", index=False)

    #print(df_norm.head())
    #print(df_abnorm.head())
    #print(df_map)
    #print(df_adj)
