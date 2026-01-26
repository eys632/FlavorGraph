# [ver2 | 상관관계 분석 | gemini_ver1] 실험 설계

> 이 문서는 **gemini_ver1** 실험이 “무엇을 검증하려고 했고”, “어떤 데이터/전처리/분석/모델링/평가 흐름으로 진행되었는지”를 처음부터 끝까지 **재현 가능한 수준으로** 설명한다.

---

## 0) 이 실험(ver2)의 핵심 목표

### 문제 정의
- 입력 **X**: 맥주의 **화학 성분(수치형 feature들)**  
- 출력 **Y**: 맥주의 **감각(향/맛) 관련 수치형 target들(다중 타깃)**  
- 목표: `X → Y` 예측 성능을 높이면서, 데이터의 **상관관계 구조(correlation structure)**를 깨지 않도록 한다.

### ver2의 관점(왜 “상관관계 분석”인가?)
이 실험은 단순히 모델을 한 번 돌려보는 것이 아니라,
1) 데이터의 분포/왜도/정규성/상관관계를 먼저 진단하고  
2) 증강(augmentation) 또는 합성(synthetic) 데이터가 **원래 상관관계를 얼마나 보존하는지**를 “정량 지표”로 체크하며  
3) “잘 만든 합성 데이터만” 골라서 학습에 섞어 성능을 개선할 수 있는지를 본다.

---

## 1) 폴더 구조 & 산출물(Outputs) 맵

### 주요 산출 폴더(예시)
- `gemini_ver1_output_0123_0124/`  
  → 이 실험이 생성한 결과물(표/그림/CSV/모델)이 모여 있음.

### 산출물 파일이 의미하는 분석 단계 (파일명 → 역할)
아래는 “파일명을 보면 실험이 무엇을 했는지”를 바로 알 수 있도록 맵핑한 것이다.

#### A. 기초 분포/정규성/왜도 진단
- `analysis_1_normality.csv` : 변수별 정규성(또는 유사 진단) 결과 테이블
- `plot_1_skewness.png` : 변수 왜도(skewness) 시각화

#### B. 상관관계 구조 진단
- `plot_2_1_XX_correlation.png` : X(화학성분)끼리의 상관행렬/히트맵
- `plot_2_2_YY_correlation.png` : Y(감각타깃)끼리의 상관행렬/히트맵
- `plot_2_correlation_compare.png` : (전/후) 상관구조 비교 플롯(주로 “합성 데이터 섞기 전후” 비교에 사용)

#### C. 구조(차원) 진단
- `plot_3_pca_dimension.png` : PCA로 본 데이터 유효 차원/설명분산
- `plot_4_tsne_structure.png` : t-SNE로 본 군집/구조

#### D. 스타일(Style) 관점 분포 차이
- `plot_5_1_style_chemical_boxplot.png` : 스타일별 화학성분 분포
- `plot_5_2_style_sensory_boxplot.png` : 스타일별 감각타깃 분포

#### E. 합성/증강 데이터 품질 보정(calibration) & 필터링
- `plot_6_calibration_dist.png` : 합성 데이터 품질 지표 분포
- `experiment_1_calibration_results.csv` : 품질 지표 요약(평균/표준편차/임계값 등)
- `plot_7_filtering_result.png` : 필터링 전/후 비교(남긴 합성 데이터의 품질)

#### F. 학습 데이터셋/트랙별 결과
- `track_A_random_400.csv` : Track A(랜덤 방식 합성/증강)로 만든 학습 데이터(또는 후보)
- `track_B_filtered_400.csv` : Track B(필터링 적용) 버전
- `final_training_data_X.csv` : 최종적으로 학습에 사용된 X(실제+합성 포함)

#### G. 성능 평가/비교
- `final_performance_results.csv` : 최종 성능(트랙별 R2/MSE/개선량)
- `performance_comparison_2stage.csv` : 2-stage 방식 비교표
- `paper_final_result_table.csv` : 논문/발표 표 형태로 정리한 최종 결과 테이블
- `plot_8_final_performance.png` : 최종 성능 플롯
- `plot_9_2stage_performance.png` : 2-stage 성능 플롯

#### H. 증강 비율/하이퍼파라미터 스윕
- `ratio_sweep_results.csv` + `plot_10_ratio_sweep.png` : “실제 대비 합성 비율” 탐색 결과
- `ultimate_mixup_sweep.csv` + `plot_15_mixup_heatmap.png` : Mixup 관련 설정 탐색(heatmap)

#### I. 상관관계 변화(부작용) 확인
- `plot_14_correlation_delta.png` : 합성 데이터 섞은 뒤 상관관계가 얼마나 바뀌었는지(Δ correlation)

#### J. (후속) 민감도/특성 중요도 분석
- `Auto_Sensitivity_A_malt_all.csv`
- `Final_Sensitivity_Result_Full_231.csv`
  → “어떤 화학성분이 특정 감각 특성(예: malt 등)에 민감/중요한가”를 정리한 결과

#### K. 저장된 최종 모델
- `champion_model_mixup.zip` 또는 `champion_model_mixup.pkl`
  → 최종 선택된 “챔피언 모델” 저장본

---

## 2) 데이터(입력/타깃/메타) 구성 가정과 확인 포인트

### X (입력 feature)
- 화학성분 중심 수치 feature들
- `Final_Sensitivity_Result_Full_231.csv` 파일명으로 보아 **X 차원이 231개 내외**로 구성된 것으로 해석 가능  
  (정확한 컬럼 리스트는 X 관련 CSV 또는 노트북의 전처리 셀에서 확인)

### Y (예측 타깃)
- 감각 특성(다중 타깃)
- 코드에서 `MultiOutputRegressor`를 쓰는 형태이므로 **Y가 여러 컬럼(다중 target)**임을 전제

### Style(스타일)
- 스타일별 boxplot이 생성되는 것으로 보아, 원본 데이터에 스타일 컬럼(카테고리)이 존재
- ver2에서 스타일은 “층화분할”의 핵심이라기보단,
  - 분포가 스타일에 따라 얼마나 달라지는지(도메인 shift)
  - 합성 데이터가 특정 스타일 분포를 망가뜨리는지  
  를 확인하는 역할로 쓰였을 가능성이 큼

---

## 3) 전처리(Preprocessing) & 분할(Splitting)

### 결측치 처리
- 코드 내 변수명이 `X_filled`, `Y_filled`로 등장하므로,
  - 결측치 제거(drop) 또는 대체(impute) 과정을 거친 뒤 학습에 투입

### Train/Test 분할
- 기본적으로 `test_size=0.2`, `random_state=42`로 80/20 분할을 사용
- 출력 로그 기준 Train이 **200개**로 고정됨  
  → 전체 데이터가 약 250개 규모일 가능성이 높음

> ⚠️ 중요: “스타일 기반 층화(stratify by style)”는 ver2 코드에서는 기본 `train_test_split` 형태로 보이며,
> 스타일 층화를 하고 싶다면 `stratify=style_label`을 별도로 넣어야 한다.

---

## 4) 분석/모델링 파이프라인(한 눈에 보기)

아래는 ver2 실험의 전체 흐름을 “입력 → 진단 → 합성/필터링 → 학습 → 평가 → 민감도”로 정리한 것이다.

(1) Raw Data 로드
↓
(2) 전처리: 컬럼 선택 / 결측치 처리 → X_filled, Y_filled 생성
↓
(3) Train/Test 분할 (기본 80/20, seed 고정)
↓
(4) EDA/진단
- 왜도/정규성 진단
- XX/YY 상관행렬 분석
- PCA/t-SNE 구조 확인
- Style별 분포(화학/감각) 비교
↓
(5) 증강/합성 데이터 생성(여러 방식)
- Random / Mixup / (필요 시 Copula 등)
↓
(6) 합성 데이터 품질 점검 & 필터링
- 상관구조 보존(예: PCD = correlation diff)
- 거리 기반(예: DCR = real과의 거리/암기 위험)
↓
(7) 모델 학습/평가 (다중출력 회귀)
- Baseline(Real only)
- Track A/B/C (증강 방식·필터링 유무 비교)
↓
(8) 스윕(비율/파라미터) → 최적 설정(챔피언) 선택
↓
(9) 최종 산출물 저장 + 해석
- final_training_data 저장
- champion_model 저장
- sensitivity/importance 결과 테이블 생성

---

## 5) “Mixup + Teacher 라벨링” 방식(핵심 설계)

### 개념
- **Mixup으로 X를 합성**
  - `x_new = λ x1 + (1-λ) x2`
- **Y는 Teacher 모델로 라벨링**
  - 합성된 `x_new`에 대해 Teacher 모델이 예측한 `ŷ_new`를 라벨로 사용

### 설계 의도
- “X의 분포/상관구조”를 크게 깨지 않으면서
- 작은 데이터(Train 200)의 한계를 완화하여
- 최종 예측모델의 일반화 성능을 끌어올리는 것이 목표

### 대표 설정(코드에서 확인되는 값)
- Train/Test: `test_size=0.2`, `random_state=42`
- Teacher 모델: `MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))`
- Mixup λ: `Beta(0.2, 0.2)`
- 최적 합성 개수 예: `n_aug = 150`

---

## 6) 평가 지표(무엇으로 “좋다/나쁘다”를 판단했나)

### 성능 지표(모델 관점)
- `R2` (설명력)
- `MSE` (오차)
- `Gain_R2` (Baseline 대비 개선량)

### 합성 데이터 품질 지표(데이터 관점)
- 상관관계 보존 정도(예: PCD = correlation diff류)
- 거리 기반 지표(예: DCR = real 데이터와의 거리/간격)
- 임계값(예: 95% 기준)을 정해 필터링

---

## 7) 재현(실행) 가이드

### 실행 순서(추천)
1) 노트북/코드에서 경로(BASE_DIR/OUTPUT_DIR)만 프로젝트 구조에 맞게 설정
2) 전처리 셀 실행 → `X_filled`, `Y_filled` 생성 확인
3) EDA/상관관계/구조 분석 셀 실행 → plot & analysis CSV 생성 확인
4) 증강 생성 → 품질 측정 → 필터링 수행
5) Track별 학습/평가 수행 → `final_performance_results.csv` 생성 확인
6) 스윕 결과 기반으로 챔피언 설정 결정 → 모델 저장
7) (후속) 민감도 분석 실행 → `Auto_Sensitivity_*.csv`, `Final_Sensitivity_Result_*.csv` 생성 확인

### 예상 실수 포인트
- (1) 경로/상대경로: output이 엉뚱한 위치에 생김
- (2) 컬럼 정렬: train/test 또는 합성/실제 concat 시 column order mismatch
- (3) 스케일링/변환을 train에 fit하고 test에 transform 안 하는 leakage
- (4) Teacher 라벨링은 Teacher 편향을 그대로 학습시키므로, 합성 비율이 커지면 성능이 오히려 떨어질 수 있음
