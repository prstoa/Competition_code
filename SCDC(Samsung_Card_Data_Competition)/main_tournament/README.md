# The top 14 teams competed.
1. Preprocess
  1) 주어진 1월 y 데이터 각각의 label을 1D array 로 변환
  2) 기존 val 데이터에서 train 데이터와 중복되는 값을 제거하여 검증 시 과적합 방지
  3) 위에서 만든 train_y, val_y를 통해 1월 학습 데이터를 train, val으로 분리
  4) train, train_y에 MLSMOTE 후 저장
  - train_preprocess.csv : MLSMOTE를 적용한 1월 데이터 중 학습에 사용된 x 데이터
  - train_target_preprocess.csv : MLSMOTE를 적용한 1월 데이터 중 학습에 사용된 y 데이터
  - val_preprocess.csv : 1월 데이터 중 검증에 사용된 x 데이터
  - val_target_preprocess.csv : 1월 데이터 중 검증에 사용된 y 데이터

2. Model
  1) LIFT 점수 모델 구현을 위해 Robust scale 적용
  2) AUROC 점수 모델 catboost와 LIFT 점수 모델 lightgbm 각각에 multi-label 분석을 위한 OneVsRestClassifier와
     cross_validation을 위한 MultilabelStratifiedKFold을 적용
  3) KFold를 통해 학습된 각각 5개의 모델을 저장(joblib 사용)
  - ovr_none_cat_model1~5.pkl : AUROC 점수를 위한 catboost 모델
  - ovr_rs_lgbm_model1~5.pkl : LIFT 점수를 위한 LightGBM 모델

3. Predict
  1) model 로드 후 예측(LIFT 점수 모델에는 예측 데이터에 Robust Scale을 적용)
  2) 예측된 AUROC 점수 모델과 LIFT 점수 모델을 5:5로 앙상블
  3) 제공된 quiz에 맞는 형식으로 입력 후 저장
  - quiz_s.csv : 제출 할 quiz 데이터
