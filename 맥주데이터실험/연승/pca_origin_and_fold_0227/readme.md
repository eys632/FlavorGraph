# pca_origin_fold_0227/ (Origin repeats & KFold diagnostics)

이 폴더는 “PCA 축이 train에 과하게 맞춰져서 test에서 망가지는가?”를
**분할 반복 / KFold 내부검증**으로 진단하기 위한 실험을 모아둔다.

구성(크게 2가지):
1) `origin/` : (주로) outer split을 여러 번 바꿔가며 PCA/재구성을 반복 점검
2) `k_fold/` : outer test는 고정해 두고, train 내부를 KFold로 쪼개 PCA 축 일반화를 진단

---

## 1) origin/

### 목적
- train/test 분할을 한 번만 하면 “우연히” 특정 split에서만 좋은/나쁜 결과가 나올 수 있다.
- 그래서 **층화 유지 + 랜덤 시드 변경**으로 split을 여러 번 만들고,
  PCA EVR/재구성/이상치 등을 반복 확인한다.

### output 구조(예)
`origin/output/1/종합/`, `origin/output/2/종합/` ... (반복 횟수만큼)

대표 산출물:
- `evr_per_pc.png`, `cumulative_evr.png`
- `explained_variance_ratio_all_pcs.csv`
- `pca_axes_full_231.npz`, `pca_axes_kept_nonzero.npz`
- `reconstruction_check_train_test.csv`
- `split_assignment.csv`
- `test_outlier_report_top10.csv`

#### `test_outlier_report_top10.csv` (의도)
- test에서 PCA 재구성 오차가 큰 샘플 Top10을 뽑아,
  “어떤 스타일/어떤 맥주가 PCA 축 관점에서 OOD인지” 빠르게 본다.

---

## 2) k_fold/

### 목적(핵심)
- “PCA 축이 train에 overfit된 것 같다”는 의심이 들 때,
  단순히 회귀모델이 과적합인지, **PCA 축 자체가 불안정한지** 분리해서 봐야 한다.
- 그래서 outer test(75)는 고정해 두고,
  train(175) 안에서 KFold로 fold_train/fold_val을 만들고,
  **PCA를 fold_train으로만 fit**한 뒤
  fold_val과 outer_test에서 **재구성 성능이 얼마나 붕괴하는지** 확인한다.

### output 구조(예)
`k_fold/output/1/종합/` ... `k_fold/output/10/종합/`

대표 산출물(각 fold의 종합 폴더):
- `reconstruction_metrics.csv`
- `split_assignment.csv`
- `style_counts.csv`
- `explained_variance_ratio_all_pcs.csv`

그리고 전체 요약:
- `k_fold/output/summary_kfold10_reconstruction.csv`

---

## CSV 컬럼 설명 (k_fold 중심)

### A) `split_assignment.csv`
(각 샘플이 어느 파트에 들어갔는지 기록)

컬럼:
- `beer` : 내부 인덱스(대개 1~250)
- `beer_id` : 맥주 ID
- `tasting_category_fine` : 스타일(층화 기준)
- `part` : `fold_train` / `fold_val` / `outer_test`
- `fold` : fold 번호(1~K)

이 파일이 있어야 “이번 fold에서 누가 train/val/test였는지” 재현 가능.

### B) `style_counts.csv`
(스타일별 샘플 수 분포)

컬럼:
- `tasting_category_fine`
- `fold_train`
- `fold_val`
- `outer_test`

목적:
- 소수 클래스가 fold_val에서 0이 되는 등, 층화가 무너졌는지 빠르게 확인.

### C) `reconstruction_metrics.csv` / `summary_kfold10_reconstruction.csv`
(PCA 재구성 성능 요약)

대표 컬럼:
- `fold` : fold 번호
- `splitter` : 예: `KFold`
- `n_fold_train`, `n_fold_val`, `n_outer_test`
- `p` : 원래 차원(231)
- `k_keep` : kept PC 수(0-분산 제거 후)
- `standardize_before_pca` : 표준화 여부
- `eig_tol` : 0-분산 판정 tol
- `fold_train_recon_rmse`, `fold_train_recon_r2_flat`
- `fold_val_recon_rmse`, `fold_val_recon_r2_flat`
- `outer_test_recon_rmse`, `outer_test_recon_r2_flat`
- `fold_val_max_abs_err`, `outer_test_max_abs_err`

`*_recon_r2_flat` 의미:
- X를 (N,p)로 펼쳐서(flat) 전체 원소 기준으로 R²를 계산한 것(=재구성 품질을 “전체 스칼라”로 요약)

---

## 매우 중요한 해석 포인트: train만 완벽하고 val/test가 폭망하는 이유

이 현상이 나오면 흔히 다음 중 하나다:

1) **rank deficiency / nullspace 문제**
- fold_train에서 분산이 0인 방향을 “삭제(k_keep < p)”하면,
  fold_val/outer_test가 그 방향으로 변동이 있을 때 정보가 크게 손실된다.
- 결과적으로 fold_train 재구성은 완벽(그 방향이 0이라서),
  fold_val/outer_test 재구성은 크게 붕괴한다.

2) **분포 이동(OOD)**
- test에만 존재하는 패턴(스타일, 제조 특성)이 PCA 축 관점에서 멀리 있을 수 있다.

3) **전처리(결측치/스케일) 영향**
- imputer/standardizer가 fold_train 기준으로 fit될 때,
  fold_val/test에서 극단값이 크게 왜곡될 수 있다.

---

## 이 폴더를 통해 “논문 서술”을 어떻게 강화할 수 있나?

- “단일 split 성능”만 주장하면 재현성 약함.
- 반복 split / KFold 진단 결과를 함께 제시하면,
  - PCA 축/재구성의 불안정성 여부
  - 성능 변동(평균±표준편차)
  - 소수 클래스 영향
  을 논문에서 논리적으로 설명할 근거가 생긴다.
