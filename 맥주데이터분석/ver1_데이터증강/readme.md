현재 연구과제에서 하고자 하는 방향과 가장 유사하면서 활용 가능한 데이터를 팀원이 찾았음. (맥주 데이터(GC + 관능평가) _ https://zenodo.org/records/10653704)

최종목표는 지역사회에 도움을 줄 수 있는 AI모델을 만드는 것이라고 이해 함.

문제 : 현재 활용 가능한 데이터셋이 절대적으로 적고 확보한 데이터셋이 맥주데이터 하나밖에 없음. 그러나 이 맥주데이터마저 데이터수가 데이터의 차원 수 보다도 매우 적어 이대로 활용하기에 어려움이 있음.

해결 방안 : 실제 모집단을 잘 따르도록 하는 데이터 생성 방법을 찾아내고, 더 나아가 생성한 데이터를 학습에 활용하여 실제 모델의 정확도가 향상되는지 검증이 필요.

먼저 이 문제와 해결방안에 대한 전반적인 이해를 위해 다음과 같은 실험을 설계, 실험, 분석을 진행함.

# 실험 설계

# 1. 연구 배경 및 목표

본 연구의 목표는 **고차원·소량(High-dimensional Small Data)** 환경에서, 단순한 샘플 복제나 표면적인 통계 근사에 그치지 않고 **모집단의 분포(Distribution)** 와 **특징 간 의존 구조(Dependency; feature–feature / feature–target 관계)** 를 함께 보존하는 데이터 생성/증강 방법을 비교·검증하는 것이다. 특히 \(N \ll D\) 환경에서는 관측 샘플 수가 적어 데이터 공간이 대부분 비어 있으며, 이때 생성 모델이 **실제 데이터가 놓인 매니폴드(Manifold) 주변의 타당한 영역**을 적절히 보강하여 (1) 데이터의 타당성(fidelity/validity)을 유지하면서도 (2) 다운스트림 예측 성능(utility)을 향상시키는지 확인하는 것이 핵심이다.

---

# 2. 데이터셋 (Beer Expert Panel)

본 실험은 맥주 데이터 중 **전문가 패널(Trained panel)의 관능평가 데이터만** 사용한다(일반인 리뷰 데이터는 사용하지 않음).

- 입력 \(X\): 맥주의 화학 성분(연속형 변수)
    - Excel “Supplementary File S1”에서 `acetaldehyde`부터 `sulfur_sum`까지의 구간 컬럼을 사용
    - 차원: \(D_X = 231\)
- 출력 \(Y\): 전문가 패널의 관능 평가 점수(다중 출력 회귀; multi-output regression)
    - Excel “Supplementary File S4”에서 `A_malt_all`부터 `overall`까지의 구간 컬럼을 사용
    - 차원: \(D_Y = 50\) (각 관능 항목을 **디스크립터(descriptor)** 라고 부름)
- 조인 키: `beer_id`

---

# 3. 데이터 전처리 및 분할 (재현성 있는 반복 실험)

## 3.1 결측치 처리 및 스케일링

- 결측치는 평균 대치(mean imputation)로 처리한다.
- 생성 모델 학습 안정성과 거리 기반 평가(DCR)를 위해 \(X, Y\) 모두 **Min–Max 스케일링(min–max scaling)** 으로 \([0,1]\) 범위로 변환한다.
- 전처리기는 **훈련 데이터(train)로만 fit**하고 테스트(test)에는 transform만 적용한다.

## 3.2 스타일 기반 층화 분할(Style-stratified split)

- 맥주 스타일(`tasting_category_fine`)을 기준으로 **층화 분할(stratified split)** 을 적용한다.
- test 비율은 0.2로 고정하여:
    - \(N_{\text{train}} = 200\)
    - \(N_{\text{test}} = 50\)

## 3.3 반복 실험(repeats = 3)

- 서로 다른 random seed로 **repeats = 3**회 반복 실험을 수행한다.
- 최종 결과는 **평균(mean) ± 표준편차(std)** 로 요약한다.

---

# 4. 생성 방법 (6가지 비교군)

## 4.1 통계적 방법 (Baselines)

### (1) 부트스트랩(Bootstrapping)

- 훈련 데이터를 복원추출하여 합성 데이터를 생성한다.
- 분포 보존은 우수하나, 복제에 가까워 일반화 한계가 있다.

### (2) KDE (Kernel Density Estimation)

- 각 변수별 독립 KDE를 학습하여 샘플링한다.
- 주변 분포는 맞출 수 있으나 변수 간 의존성은 약하다.

### (3) 다변량 가우시안 코퓰러(Gaussian Copula)

- 순위 기반 누적분포 변환 후 정규 공간에서 상관 구조를 학습한다.
- 공분산은 **Ledoit–Wolf 수축(shrinkage)** 으로 안정화한다.

## 4.2 생성형 AI 모델

### (4) VAE (Variational Autoencoder)

- \([X|Y]\) 결합 벡터를 입력으로 잠재공간 분포를 학습한다.
- GPU 및 AMP(혼합정밀도)를 활용하며, early stopping을 적용한다.

### (5) CTGAN

- 정형 데이터 생성을 위한 GAN 기반 모델이다.
- SDV 또는 ctgan 라이브러리가 설치된 경우 실행한다.

### (6) TabDDPM (Tabular Diffusion Model)

- 확산모델을 정형 데이터에 적용한 생성 모델이다.
- \([0,1]\) → \([-1,1]\) 변환 후 확산 과정을 학습하며 GPU를 사용한다.

---

# 5. 증강 데이터 구성

- 훈련 데이터 수 \(N_{\text{train}} = 200\)
- 합성 데이터 수 \(N_{\text{synth}} = 2 \times N_{\text{train}} = 400\)
- 증강 학습 데이터 총량:
\[
N_{\text{aug}} = N_{\text{train}} + N_{\text{synth}} = 600
\]

---

# 6. 평가 지표 및 검증 프로토콜

## Phase 1. 모집단 적합성 및 구조 보존

### (1) KS-test (Kolmogorov–Smirnov test)

- 실제 데이터와 합성 데이터의 **분포 유사도**를 feature별로 측정한다.
- X(화학 성분)와 Y(관능 점수)에 대해 각각 수행한다.

### (2) PCD (Pairwise Correlation Difference)

- 실제 상관행렬과 합성 상관행렬의 차이를 측정한다.
- **X–X(화학–화학)** 및 **X–Y(화학–관능)** 블록을 분리하여 평가한다.
- Pearson 및 Spearman 상관을 모두 사용하며, **Ledoit–Wolf shrinkage** 로 안정화한다.

### (3) DCR (Distance to Closest Record)

- 합성 샘플과 가장 가까운 실제 샘플 간의 거리를 계산한다.
- 복제(암기) 여부 및 이상치 생성을 진단한다.

---

## Phase 2. 데이터 유용성 검증 (Utility)

### Downstream Predictor

- **GBDT (Gradient Boosting Decision Trees)**: 주력 예측기
- **MLP (Neural Network)**: 비교용 예측기

### 평가 시나리오

1. **TRTR**: 실제 데이터로 학습 → 실제 데이터로 테스트
2. **TSTR**: 합성 데이터로 학습 → 실제 데이터로 테스트
3. **AUG**: 실제 + 합성 데이터로 학습 → 실제 데이터로 테스트

### 평가 지표

- 결정계수 \(R^2\)
- 제곱평균제곱근오차(RMSE)
- 관능 디스크립터별 성능 및 **평균 순위(average rank)**

---

# 7. 재현성 및 결과 저장

- 모든 결과는 실행 시각 기반 `RUN_ID` 폴더에 저장된다.
- repeats별 결과, descriptor별 점수, 평균±표준편차 요약, 평균 순위 파일을 모두 기록한다.

---

# 8. 계산 자원

- VAE, TabDDPM, MLP는 GPU(CUDA) 및 AMP를 사용한다.
- DCR 계산은 GPU 기반 거리 계산으로 가속한다.
