# pca_add/0225/ (Cumulative PCR for 50 targets)

## 한 줄 요약
- Train(175)로 PCA를 fit하고,
- PC를 1개부터 k개까지 **누적**하며 PCR(선형회귀)을 수행,
- 50개 관능평가 타깃 각각에 대해 k-성능 곡선을 저장하고,
- 종합 CSV로 비교/분석할 수 있게 만든 실험.

---

## 왜 이 실험을 하나?

PCA는 “분산이 큰 방향”부터 PC를 만든다.
따라서 “PC를 하나씩 독립적으로” 보는 것보다,
**PC1→PC2→… 누적하면서 성능이 어떻게 변하는지** 보는 흐름이 더 자연스럽다.

이 실험은 특히 다음을 보기 좋다:
- k가 늘수록 train은 좋아지는데 test는 나빠지는 과적합 패턴
- “최적 k”가 타깃마다 얼마나 다른지
- 누적 EVR이 어느 정도일 때 성능이 peak를 찍는지

---

## 주요 파일

### (1) 노트북
- `beer_cumulative_pcr_experiment_v3.ipynb`
  - PCA 학습(train fit), k 누적 PCR, 결과 저장
- (있다면) `결과확인.ipynb`
  - 종합 CSV를 읽어서 비교/정렬/시각화

---

## output 구조 예시

`output/20260225_220244/`
- `target_01_A_malt_all/`
  - `cumulative_pcr_A_malt_all.csv`
  - `best_k_summary_A_malt_all.txt`
  - `cumulative_rmse_curve_A_malt_all.png`
  - `cumulative_r2_curve_A_malt_all.png`
- ...
- `target_50_overall/` (50개 타깃 전부)
- `종합/`
  - `cumulative_pcr_all_targets.csv`
  - `best_k_per_target.csv`
  - `explained_variance_ratio_all_pcs.csv`
  - `pca_axes_full_231.npz`
  - `pca_axes_kept_nonzero.npz`
  - `reconstruction_check_train_test.csv`
  - `compare_paper_vs_experiment_best_test_r2_*.csv`

---

## CSV 컬럼 설명(중요)

### A) `cumulative_pcr_all_targets.csv` / `cumulative_pcr_<target>.csv`
(타깃별로 k 증가시키며 측정한 로그)

- `target`  
  타깃 코드 (예: `A_malt_all`)
- `target_index_1based`  
  1~50 타깃 인덱스(1-based)
- `k`  
  누적 사용 PC 개수 (PC1~PCk)
- `cum_evr_percent`  
  k개 PC까지의 누적 explained variance ratio(%)  
  → “X 분산의 몇 %를 PC1~PCk가 설명하나”
- `train_rmse`, `test_rmse`  
  train/test에서의 RMSE
- `train_r2`, `test_r2`  
  train/test에서의 R²
- `train_accuracy`, `test_accuracy`  
  (이 실험에서) 보통 **corr(y, ŷ)** 값을 저장해 둔 것  
  ※ 분류 정확도 아님
- `cv_rmse_mean`, `cv_rmse_std`  
  (옵션) train 내부 CV에서 RMSE 평균/표준편차
- `cv_r2_mean`, `cv_r2_std`  
  (옵션) train 내부 CV에서 fold별 R²의 평균/표준편차

> CV가 들어간 파일이라면 “모델 선택은 CV로, 최종 평가는 test로”가 정석이다.
> test로 best_k를 고르는 값은 비교표로는 쓸 수 있어도, 엄밀한 모델 선택으로 쓰면 누설이 된다.

---

### B) `best_k_per_target.csv`
(타깃별로 “best”로 선택된 k와 그때 성능 요약)

자주 나오는 컬럼:
- `best_k`  
  선택된 k
- `best_cum_evr_percent`  
  best_k에서의 누적 EVR(%)
- `best_train_rmse`, `best_train_r2`
- `best_test_rmse`, `best_test_r2`
- (있다면) `best_cv_rmse_mean`, `best_cv_r2_mean`

---

### C) `explained_variance_ratio_all_pcs.csv`
(PCA 자체의 PC별 정보)

- `pc_index_1based_full` : PC 번호(1-based, 원래 231차원 기준)
- `eigenvalue` : 공분산 고유값(분산 크기)
- `explained_variance_ratio` : PC별 EVR(0~1)
- `explained_variance_ratio_percent` : PC별 EVR(%)
- `kept_nonzero` : “0 분산(또는 tol 이하)”가 아니라서 유지했는지 여부

---

### D) `reconstruction_check_train_test.csv`
(PCA 축으로 투영했다가 inverse_transform 했을 때 복원 성능)

대표 컬럼 예:
- `n_train`, `n_test` : 샘플 수
- `p` : 원래 차원(231)
- `k_keep` : “유효 PC 수”(0-분산 PC 제거 후 남은 PC 수)
- `standardize_before_pca` : 표준화 여부(True/False)
- `eig_tol` : 0-분산 판정 tolerance
- `train_recon_rmse`, `train_recon_mae`, `train_recon_max_abs_err`
- `test_recon_rmse`, `test_recon_mae`, `test_recon_max_abs_err`

**중요 해석 포인트**
- train 재구성은 완벽한데 test 재구성이 매우 나쁘면,
  train에서는 0-분산이라 버린 축 방향이 test에서는 변동이 있을 수 있다(=nullspace 문제).
  이 경우 “train 정보를 100% 보존”은 맞지만, “미래(test) 정보를 100% 보존”은 아니다.

---

## 결과를 어떻게 읽나 (추천 순서)

1) `종합/explained_variance_ratio_all_pcs.csv` + 그래프
   - PC tail이 0에 가까운지, 누적 EVR이 어디서 포화되는지

2) `종합/cumulative_pcr_all_targets.csv`
   - 타깃별로 test_r2가 peak 찍는 k가 작은지/큰지 분포 확인

3) 타깃 1개를 잡고 `target_XX_*/cumulative_r2_curve_*.png`, `cumulative_rmse_curve_*.png`
   - “k 늘어날수록 과적합” 패턴이 명확한지 체크

4) `종합/reconstruction_check_train_test.csv`
   - PCA 자체가 test에서 불안정한지(=축이 train 전용인지) 확인

---

## 마지막으로(논문 재현성 관점)

- “논문 best와 실험 best” 비교는 `compare_paper_vs_experiment_best_test_r2_*.csv`로 한다.
- 다만, 엄밀한 학술적 서술에서는
  - **k 선택은 train 내부**(CV)로 결정하고,
  - test는 “최종 1회 평가”
  흐름이 더 안전하다.
