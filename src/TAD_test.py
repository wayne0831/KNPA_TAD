###########################################################################################################
# import libraries
###########################################################################################################

import os
from config import *
#from TAD_train import target
from TAD_model import *
from TAD_result_analysis import *
from RL_model import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###########################################################################################################
# model test
###########################################################################################################

model_type = 'TAD' # 'RL'
model_path = CHK_PATH[model_type]
model_name = os.path.basename(model_path)
res_path   = RES_PATH[model_type]

if PIPELINE['is_test']: 
    # 데이터 로드
    print('==============Data load for test==============')
    val_set, val_meta   = load_dataset(csv_path=DATA_PATH[TAD_VER]['val'], seq_len=SEQ_LEN, stride=STRIDE)
    test_set, test_meta = load_dataset(csv_path=DATA_PATH[TAD_VER]['te'], seq_len=SEQ_LEN, stride=STRIDE)
    print('==============Data loaded!==============')

    # 모델 불러오기
    print('==============Model load==============')
    # 1) base model load
    model = build_base_model()

    model.load_state_dict(torch.load(model_path))
    print('==============Model loaded!==============')
    print(f'Model name: {model_name}')

    # 결과 산출
    print('==============Test and get results==============')
    # validation/test set의 link + lane별 이상치 임계치 도출, test set 이상여부 판단
    val_results      = detect_anomalies(model, val_set, val_meta.copy())
    group_thresholds = get_group_thresholds(val_results) 
    test_results_raw = detect_anomalies(model, test_set, test_meta.copy()) 
    test_results     = apply_group_threshold(test_results_raw, group_thresholds) # link + lane별 이상여부 판단

    # 결과 저장
    print('==============Save results==============')
    val_results.to_csv(res_path['val'], index=False)
    test_results.to_csv(res_path['te'], index=False)
    group_thresholds.to_csv(res_path['grp_thr'], index=False)

    # 4) === NEW === 평균 비교 후 도메인 필터링
    cmp_df = filter_by_domain(model, test_set, test_results.copy())
    cmp_df.to_csv(res_path['cmp_df'], index=False)

    # 5) === NEW === 링크+시간 단위 최종 이상 집계
    link_time_df = aggregate_link_time(cmp_df, col='final_anomaly')
    link_time_df.to_csv(res_path['agg_link'], index=False)
    print('==============Results saved!==============')
# end if

###########################################################################################################
# visulaize results
###########################################################################################################

# 결과 시각화
if PIPELINE['visualize_conf_mat']:
    # 정확한 평가를 위한 조건 확인
    if 'pred' in test_results.columns:
        y_true = test_results['pred']
        y_pred = test_results['anomaly']

        # 정밀도, 재현율, F1
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n📊 Test 평가 지표:")
        print(f" - Precision : {precision:.4f}")
        print(f" - Recall    : {recall:.4f}")
        print(f" - F1 Score  : {f1:.4f}")

        # 분류 리포트 출력
        print("\n📋 상세 리포트:")
        print(classification_report(y_true, y_pred, digits=4))

        # ROC-AUC & PR-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
            print(f"\n🔵 ROC-AUC     : {roc_auc:.4f}")
            print(f"🟢 PR-AUC      : {pr_auc:.4f}")
        except ValueError as e:
            print(f"⚠️ AUC 계산 오류: {e}")

        # ========================
        # 🔶 Confusion Matrix (counts)
        # ========================
        labels = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=labels)  # counts

        print("\n🧮 Confusion Matrix (counts):")
        print(pd.DataFrame(
            cm,
            index=[f"True {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels]
        ))

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # 시각화 (counts만)
        #fig, ax = plt.subplots(figsize=(5, 4))
        #disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        #disp.plot(ax=ax, values_format='d', colorbar=False)
        #ax.set_title('Confusion Matrix ')
        #plt.tight_layout()
        #plt.savefig("confusion_matrix.png", dpi=150)
        #plt.show()
        #print("🖼️ 혼동행렬 이미지를 'confusion_matrix.png'로 저장했습니다.")

    else:
        print("⚠️ Test 데이터에 'pred' 열이 없습니다! 평가 지표 생략됨.")
# end if

if PIPELINE['visualize_line_plot']:

    # 기존 merged 생성은 그대로 두고...
    merged = test_results_raw.merge(group_thresholds, on=['LINK_ID', 'lane'], how='left')

    # 타입 정리
    merged['recon_error'] = merged['recon_error'].astype(float)
    merged['threshold']   = merged['threshold'].astype(float)
    merged['date']        = pd.to_datetime(merged['date'])

    # ✅ cmp_df에서 final_anomaly만 끌어와서 합치기 (키는 LINK_ID, lane, date)
    tmp = cmp_df[['LINK_ID','lane','date','final_anomaly']].copy()
    tmp['date'] = pd.to_datetime(tmp['date'])

    merged = merged.merge(tmp, on=['LINK_ID','lane','date'], how='left')
    merged['final_anomaly'] = merged['final_anomaly'].fillna(0).astype(int)

    # 이벤트 룰 정의
    event_rules = [
        {'day': 1, 'start': 33300,  'end': 35100,  'link_id': 51,  'lane': [4,5], 'pred': 1},
        {'day': 2, 'start': 119700, 'end': 120600, 'link_id': 164, 'lane': 2,     'pred': 1},
        {'day': 3, 'start': 220500, 'end': 221400, 'link_id': 113, 'lane': 2,     'pred': 1},
        {'day': 4, 'start': 321300, 'end': 323100, 'link_id': 136, 'lane': [3,4], 'pred': 1},
        {'day': 5, 'start': 346500, 'end': 432900, 'link_id': 'ALL','lane': 'ALL','pred': 1},
        {'day': 6, 'start': 461700, 'end': 468900, 'link_id': 42,  'lane': 1,     'pred': 1},
        {'day': 7, 'start': 580500, 'end': 581400, 'link_id': 67,  'lane': 2,     'pred': 1},
    ]

    # 'ALL' 제외한 고유 link_id 추출
    target_links = sorted({rule['link_id'] for rule in event_rules if rule['link_id'] != 'ALL'})

    # 그룹별 시각화
    for link in target_links:
        group_df = merged[merged['LINK_ID'] == link].copy()
        if group_df.empty:
            print(f"[!] LINK_ID={link}에 해당하는 데이터가 없습니다.")
            continue

        group_df = group_df.sort_values('date')
        threshold = group_df['threshold'].iloc[0]
        anomaly_df = group_df[group_df['pred'] == 1]  # 실제 이상 라벨

        lanes = group_df['lane'].unique()
        for lane in lanes:
            lane_df = group_df[group_df['lane'] == lane]
            anomaly_df_t =  anomaly_df[anomaly_df['lane'] == lane]
            plt.figure(figsize=(10, 4))
            plt.plot(lane_df['date'], lane_df['recon_error'], label='Reconstruction Error', color='black', alpha = 0.5)
            plt.scatter(
                lane_df[lane_df['pred'] == 1]['date'],
                lane_df[lane_df['pred'] == 1]['recon_error'],
                color='red', s=25, label='Actual Anomaly (Label==1)'
            )
            plt.scatter(
                lane_df[lane_df['final_anomaly'] == 1]['date'],
                lane_df[lane_df['final_anomaly'] == 1]['recon_error'],
                color='blue', s=10, label='pred final Anomaly (Label==1)'
            )
            plt.axhline(threshold, color='orange', linestyle='--', label=f'Threshold: {threshold:.4e}')
            plt.title(f"LINK_ID={link} | Lane={lane}")
            plt.xlabel("Time")
            plt.ylabel("Reconstruction Error")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.show()
# end if