# Seaborn 소개 및 머신러닝 활용

## Seaborn이란?

Seaborn은 Python 데이터 시각화 라이브러리로, matplotlib을 기반으로 하여 더 아름답고 정보가 풍부한 통계 그래픽을 만들 수 있도록 설계되었습니다. Michael Waskom에 의해 개발되었으며, 특히 **머신러닝 프로젝트의 데이터 탐색과 모델 해석**에 강력한 도구로 사용됩니다.

## 머신러닝에서 왜 Seaborn을 사용해야 할까요?

### 1. 머신러닝 워크플로우 전반에 걸친 시각화 지원

Seaborn은 머신러닝 프로젝트의 모든 단계에서 활용할 수 있습니다:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 문제 상황: 분류 모델링 전 데이터 특성 파악
# 데이터 로드
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Seaborn으로 클래스별 특성 분포 확인
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x=feature, hue='target', kde=True)
    plt.title(f'{feature} 분포 (클래스별)')

plt.tight_layout()
plt.show()
```

### 2. 통계적 기능을 통한 데이터 인사이트 발견

Seaborn은 복잡한 통계적 시각화를 간단한 코드로 구현하여 머신러닝 모델링 전 중요한 패턴을 발견할 수 있게 합니다.

```python
# 문제 상황: 피처 간 관계와 클래스 분리 패턴 파악
# 회귀선과 신뢰 구간이 포함된 산점도로 피처 관계 파악
sns.lmplot(data=df, x="feature_1", y="feature_2", hue="target",
           col="target", height=5, aspect=1)
plt.suptitle('클래스별 피처 관계 패턴', y=1.02)
plt.show()
```

### 3. 다변량 관계 시각화로 피처 중요도 파악

머신러닝 모델의 성능에 영향을 미치는 피처 간의 복잡한 관계를 효과적으로 시각화할 수 있습니다.

```python
# 문제 상황: 다차원 데이터에서 중요한 패턴 발견
# 다변량 관계 시각화
sns.pairplot(df, hue="target", diag_kind="kde", markers=["o", "s"])
plt.suptitle('다변량 관계 시각화', y=1.02)
plt.show()
```

### 4. 모델 평가 및 해석을 위한 시각화

모델 예측 결과, 특성 중요도, 오차 패턴 등을 시각화하여 모델 성능을 평가하고 해석할 수 있습니다.

```python
# 문제 상황: 모델 예측 결과와 특성 중요도 시각화
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 모델 훈련
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('특성 중요도')
plt.show()
```

## Seaborn vs Matplotlib (머신러닝 관점)

| 특징 | Matplotlib | Seaborn |
|------|------------|---------|
| 머신러닝 워크플로우 지원 | 직접 구현 필요 | 내장된 통계 기능으로 지원 |
| 코드 복잡도 | 복잡한 통계 플롯에 많은 코드 필요 | 간결한 코드로 복잡한 통계 플롯 |
| 데이터 탐색 | 기본적인 플롯 | 자동화된 통계 분석 기능 |
| 모델 평가 시각화 | 직접 구현 | 내장된 평가 시각화 도구 |
| DataFrame 지원 | 기본적 | 완벽한 통합으로 ML 파이프라인과 호환 |

## 머신러닝에서의 Seaborn 주요 구성 요소

### 1. 관계형 플롯 (Relational Plots) - 피처 관계 탐색
- `scatterplot()`: 피처 간 관계와 클래스 분리 패턴 파악
- `lineplot()`: 시계열 데이터와 성능 추이 분석
- `relplot()`: 다중 플롯으로 복잡한 관계 탐색

### 2. 범주형 플롯 (Categorical Plots) - 범주형 피처 분석
- `boxplot()`: 클래스별 피처 분포 비교
- `violinplot()`: 밀도 분포를 통한 클래스별 특성 파악
- `barplot()`: 범주별 통계적 요약
- `catplot()`: 복잡한 범주형 데이터 분석

### 3. 분포 플롯 (Distribution Plots) - 데이터 분포 이해
- `histplot()`: 피처 분포와 왜도 확인
- `kdeplot()`: 클래스별 밀도 함수 비교
- `ecdfplot()`: 누적 분포로 데이터 특성 파악
- `displot()`: 다중 분포 플롯으로 종합적 분석

### 4. 행렬 플롯 (Matrix Plots) - 피처 상관관계 분석
- `heatmap()`: 상관관계 행렬로 다중공선성 파악
- `clustermap()`: 피처 클러스터링으로 그룹화 패턴 발견

### 5. 회귀 플롯 (Regression Plots) - 회귀 분석 시각화
- `regplot()`: 회귀 관계와 신뢰 구간 시각화
- `lmplot()`: 다중 회귀 플롯으로 복잡한 관계 분석
- `residplot()`: 잔차 분석으로 모델 가정 검증

## 머신러닝에서의 Seaborn 활용 철학

1. **데이터 이해 기반**: 모델링 전 데이터의 본질을 파악하는 데 도움을 줍니다.
2. **특성 엔지니어링 지원**: 시각화를 통해 효과적인 피처 변환 아이디어를 제공합니다.
3. **모델 해석 용이성**: 복잡한 모델의 동작과 결과를 직관적으로 이해하게 합니다.
4. **성능 모니터링**: 모델 성능 변화와 문제점을 시각적으로 파악할 수 있게 합니다.

## 머신러닝에서의 실제 활용 사례

### 데이터 탐색적 분석 (EDA) - 모델링 전 데이터 이해
```python
# 문제: 분류 문제에서 클래스별 피처 분포 파악
# 해결: pairplot으로 다차원 관계 시각화
sns.pairplot(df, hue="target", diag_kind="kde", vars=feature_names)
```

### 피처 엔지니어링 - 피처 변환 효과 확인
```python
# 문제: 왜곡된 피처 분포 정규화 필요성 파악
# 해결: 원본과 변환 후 분포 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['feature_1'], kde=True, ax=axes[0]).set_title('원본 분포')
sns.histplot(np.log1p(df['feature_1']), kde=True, ax=axes[1]).set_title('로그 변환 후')
```

### 모델 평가 - 성능 지표 시각화
```python
# 문제: 회귀 모델의 예측 성능과 잔차 패턴 분석
# 해결: 실제값 vs 예측값 비교와 잔차 분석
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=y_test, y=y_pred, ax=axes[0])
sns.residplot(x=y_pred, y=y_test - y_pred, ax=axes[1])
```

### 특성 중요도 시각화 - 모델 해석
```python
# 문제: 어떤 피처가 예측에 중요한지 파악
# 해결: 특성 중요도를 막대 그래프로 시각화
feature_imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
sns.barplot(data=feature_imp.sort_values('importance', ascending=False).head(10),
            x='importance', y='feature')
```

## 머신러닝 문제 해결 워크플로우

Seaborn은 머신러닝 프로젝트의 전체 수명주기에 걸쳐 활용될 수 있습니다:

1. **데이터 수집 및 이해**: 데이터의 기본 구조와 품질 파악
2. **탐색적 데이터 분석**: 변수 간 관계와 패턴 발견
3. **피처 엔지니어링**: 피처 변환 효과 확인 및 최적의 피처 선택
4. **모델 선택 및 훈련**: 다양한 모델 성능 비교
5. **모델 평가**: 성능 지표와 오차 패턴 분석
6. **모델 해석**: 특성 중요도와 예측 결과 시각화
7. **모델 배포 및 모니터링**: 운영 중 성능 변화 추적

자세한 워크플로우와 예시는 [머신러닝 문제 해결 프레임워크](ml-problem-solution-framework.md)를 참조하세요.

## 다음 단계

Seaborn의 머신러닝 활용을 이해했다면, 다음으로 [설치 및 설정 가이드](02-installation.md)를 통해 개발 환경을 설정해보세요.

## 추가 자료

- [Seaborn 공식 문서](https://seaborn.pydata.org/introduction.html)
- [Seaborn 예제 갤러리](https://seaborn.pydata.org/examples/index.html)
- [Seaborn API 참조](https://seaborn.pydata.org/api.html)
- [머신러닝 문제 해결 프레임워크](ml-problem-solution-framework.md)