# 맥주데이터실험 (Beer PCA / PCR Experiments)

이 폴더는 “맥주 화학(231개 변수) → 관능평가(50개 타깃)” 예측을 위해,
- (1) 논문 방식/분할을 최대한 유지하며
- (2) PCA로 화학 변수 간 상관(공분산)을 제거한 뒤
- (3) PCR(주성분 회귀) 등 선형 모델의 성능과 과적합/일반화 문제를 점검하기 위한 실험들을 모아둔 곳이다.

핵심 질문:
1. **화학 변수 간 상관이 큰 상태에서 선형모델을 쓰는 것이 위험한가?**
2. **PCA로 decorrelation(비상관화)을 하고도 예측 성능이 유지/개선되는가?**
3. **PCA 축이 train에만 “맞춰진 축”이 되어 test에서 무너지는 현상(분포 이동, rank deficiency)이 있는가?**
4. **(k개 PC 누적)에서 k 증가에 따라 train은 좋아지고 test는 나빠지는 전형적 과적합이 나타나는가?**
5. **k 선택을 CV로 해야 하는가, 반복 stratified split으로 안정성을 봐야 하는가?**

---

## 데이터(로컬) 전제

실험 노트북들은 로컬 서버에 있는 논문 부록 엑셀(`Supplemental Files and Figure source files.xlsx`)을 읽는 형태로 작성되어 있다.
예) `/home/.../data/Supplemental Files and Figure source files.xlsx`

일반적으로 다음이 필요하다:
- 화학 데이터 X: shape = (N_beer, 231)
- 관능평가 Y: shape = (N_beer, 50)  (타깃 50개를 하나씩 회귀)
- 맥주 스타일(층화용): `tasting_category_fine` (혹은 유사 컬럼)

---

## 큰 실험 흐름(공통)

### (A) Outer split (논문 train/test 유지)
- 전체 250개 맥주를 **train 175 / test 75**로 분할한다.
- 스타일 불균형을 고려해 가능하면 `tasting_category_fine`로 **층화(stratified) 분할**한다.
- 분할 결과를 `split_assignment.csv` 같은 파일로 저장하여 재현성을 확보한다.

### (B) PCA (train으로만 fit)
- PCA는 **train 데이터로만 fit** 한다. (test를 PCA fit에 쓰면 누설)
- 보통 표준화(센터링+스케일링)를 먼저 한다.
- PCA 결과로 얻는 것:
  - `components`(주성분 축; 직교기저)
  - `explained_variance_ratio` 및 누적 설명분산(cum EVR)
  - (옵션) eigenvalue(고유값)

### (C) PCR (k개 주성분 누적)
- 타깃 1개(Y_j)에 대해,
  - PC1~PCk로 투영한 Z_k로 선형회귀 학습
  - train/test RMSE, R², corr(=accuracy로 저장한 경우)을 계산
  - k를 1부터 k_keep(=유효 PC 수)까지 증가시키며 곡선으로 분석

### (D) k 선택(중요)
- **test 기반으로 best_k를 고르는 것은 “평가 누설”**이다.
  - 논문 재현/비교용으로는 “test best”를 따로 기록할 수 있으나,
  - “모델 선택”은 train 내부 CV(또는 별도 validation)로 해야 논리적으로 깔끔하다.
- 따라서 이 폴더에는
  - CV로 k를 고르는 실험
  - test R²로 “사후적으로” best를 보는 비교표
  둘 다 존재할 수 있다.

---

## 폴더 구조 안내(요약)

- `pca/`
  - PCA 자체(축/EVR/저장) sanity check 및 기초 분석.
- `pca_add/0225/`
  - 50개 타깃 전체에 대해 “k를 누적시키는 PCR”을 수행하고, per-target 결과와 종합 CSV를 저장.
- `pca_origin_fold_0227/`
  - 분할을 여러 번 반복하거나(K회),
  - 또는 train 내부를 KFold로 나눠 PCA 축의 일반화(재구성 성능 등)를 점검.

각 폴더 README를 먼저 읽으면 결과 해석이 빨라진다.

---

## 결과를 볼 때 가장 먼저 확인할 것 (체크리스트)

1) `explained_variance_ratio_all_pcs.csv`  
   - EVR이 빠르게 80~90%에 도달하는지, tail이 긴지.

2) `cumulative_pcr_all_targets.csv` 또는 per-target `cumulative_pcr_*.csv`  
   - k 증가에 따른 **train↑ / test↓** 전형적 과적합이 언제부터 시작되는지.

3) `reconstruction_check_train_test.csv` 또는 `reconstruction_metrics.csv`  
   - PCA가 “정보 보존(재구성)” 관점에서 test에 대해 붕괴하는지 확인.
   - 특히 train 재구성은 완벽한데 test 재구성이 매우 나쁘면,
     - (i) train에서 0-분산이었던 방향이 test에서는 변동이 있는 경우(=nullspace 문제),
     - (ii) 스케일링/결측처리 방식이 train/test에 다르게 영향을 준 경우,
     - (iii) test가 out-of-distribution인 경우
     를 의심해야 한다.

---

## CSV/지표 용어 요약

- RMSE: sqrt(mean((y - ŷ)^2))
- R²: 1 - SSE/SST. 0은 “평균 예측과 비슷”, 음수는 “평균보다 더 못함”.
- accuracy(이 실험들에서): 분류 정확도가 아니라 **corr(y, ŷ)**를 저장한 경우가 많다(이름만 accuracy).

(자세한 컬럼 설명은 각 실험 폴더 README 참고)
