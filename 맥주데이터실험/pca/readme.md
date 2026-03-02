# pca/ (PCA basics & axis saving)

이 폴더는 PCA를 “예측”이 아니라 “축 생성/저장/설명분산 확인/재구성” 관점에서 점검하는 기초 실험들을 모아둔다.

대표 산출물(예: `0216/`):
- `beer_pca_basics_axes231.ipynb`
- `pca_basics_YYYYMMDD_HHMMSS/`
  - `meta.json`: 실행 설정(경로, seed, 옵션 등)
  - `outer_split_indices.json`: (있다면) outer split 인덱스 기록
  - `pca_all_250/`:
    - `X_scaled.npy`: 표준화된 X
    - `Z_full.npy`: PCA 투영 점수(전체 PC)
    - `components.npy`: PCA components(축)
    - `cumulative_evr.png`: 누적 EVR 그래프
    - `explained_variance.csv`: PC별 eigenvalue/EVR
    - `imputer.joblib`: 결측치 처리 객체(사용 시)
    - `pca.joblib`: PCA 객체(사용 시)

---

## 이 실험의 목적

1) PCA 구현이 맞는지 확인(축 직교성, EVR 계산, 저장/로드)
2) PC별/누적 설명분산 곡선을 보고 “차원축소”를 할지 말지 감을 잡기
3) 이후 PCR 실험에서 쓸 저장 포맷(npz/joblib/npy)이 제대로 재현되는지 검증

---

## 해석 가이드(빠르게 보기)

- `cumulative_evr.png`:
  - PC 몇 개에서 EVR이 80/90/95%를 넘는지 확인.
- `components.npy`:
  - shape 보통 (231, 231) 혹은 (k, 231).
- `explained_variance.csv`:
  - tail 쪽 EVR이 0에 가까운 PC가 많은지(=rank deficiency 가능).

---

## 주의

이 폴더의 일부 실험은 “전체 250개”로 PCA를 하는 형태일 수 있다.
이 경우 PCA 축 자체가 test 정보를 본 것이므로,
**엄밀한 일반화 평가(논문 train/test 재현)**와는 별도로 “분석/탐색” 용도로만 본다.
