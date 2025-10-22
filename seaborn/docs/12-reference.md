# Seaborn 머신러닝 문제 해결 참조 요약

이 문서는 머신러닝 프로젝트에서 Seaborn을 활용하여 문제를 해결하는 데 필요한 주요 기능과 함수들을 빠르게 참조할 수 있는 치트 시트입니다. 함수 구문, 주요 매개변수, 문제 해결 패턴과 사용 예시를 포함합니다.

## 기본 설정

### 임포트 및 설정

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 테마 설정
sns.set_theme(style="whitegrid")  # darkgrid, whitegrid, dark, white, ticks
sns.set_context("notebook")       # paper, notebook, talk, poster
sns.set_palette("deep")           # deep, muted, pastel, bright, dark, colorblind

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

### 머신러닝 프로젝트 기본 설정

```python
# 머신러닝 라이브러리 임포트
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

# 재현성을 위한 시드 설정
np.random.seed(42)
```

## 머신러닝 문제 해결을 위한 플롯 함수 요약

### 관계형 플롯 (Relational Plots) - 피처 관계 탐색

| 함수 | ML 문제 유형 | 주요 매개변수 | 문제 해결 패턴 | 예시 코드 |
|------|--------------|--------------|--------------|----------|
| `scatterplot()` | 클래스 분리 패턴 파악 | `x, y, hue, size, style, data` | 분류 문제에서 클래스별 피처 관계 시각화 | `sns.scatterplot(data=df, x='feature1', y='feature2', hue='target')` |
| `lineplot()` | 시계열 성능 모니터링 | `x, y, hue, style, size, data, ci` | 모델 성능의 시간에 따른 변화 추적 | `sns.lineplot(data=perf_df, x='date', y='accuracy')` |
| `relplot()` | 다차원 관계 탐색 | `x, y, hue, col, row, kind, data` | 여러 피처 간 복잡한 관계 동시 분석 | `sns.relplot(data=df, x='f1', y='f2', col='category', hue='target')` |

### 범주형 플롯 (Categorical Plots) - 범주형 피처 분석

| 함수 | ML 문제 유형 | 주요 매개변수 | 문제 해결 패턴 | 예시 코드 |
|------|--------------|--------------|--------------|----------|
| `boxplot()` | 클래스별 피처 분포 비교 | `x, y, hue, data, order, width` | 분류 문제에서 각 클래스별 피처 분포 차이 분석 | `sns.boxplot(data=df, x='target', y='feature')` |
| `violinplot()` | 클래스별 밀도 분포 파악 | `x, y, hue, data, split, inner` | 분류 문제에서 클래스별 밀도 함수 비교 | `sns.violinplot(data=df, x='target', y='feature', split=True)` |
| `barplot()` | 범주별 통계 요약 | `x, y, hue, data, ci, estimator` | 범주형 피처의 통계적 특성 요약 및 비교 | `sns.barplot(data=df, x='category', y='value', estimator=np.mean)` |
| `countplot()` | 클래스 불균형 확인 | `x, y, hue, data` | 분류 문제에서 클래스 불균형 상태 시각화 | `sns.countplot(data=df, x='target')` |

### 분포 플롯 (Distribution Plots) - 데이터 분포 이해

| 함수 | ML 문제 유형 | 주요 매개변수 | 문제 해결 패턴 | 예시 코드 |
|------|--------------|--------------|--------------|----------|
| `histplot()` | 피처 분포 및 정규성 검증 | `x, y, hue, data, bins, kde, stat` | 회귀 모델 가정 검증 및 이상치 탐지 | `sns.histplot(data=df, x='feature', kde=True)` |
| `kdeplot()` | 클래스별 밀도 함수 비교 | `x, y, hue, data, shade, levels` | 분류 문제에서 클래스별 밀도 차이 분석 | `sns.kdeplot(data=df, x='feature', hue='target')` |
| `ecdfplot()` | 누적 분포 함수 분석 | `x, y, hue, data, stat` | 데이터의 분포 형태와 특성 파악 | `sns.ecdfplot(data=df, x='feature')` |

### 행렬 플롯 (Matrix Plots) - 상관관계 및 유사성 분석

| 함수 | ML 문제 유형 | 주요 매개변수 | 문제 해결 패턴 | 예시 코드 |
|------|--------------|--------------|--------------|----------|
| `heatmap()` | 피처 상관관계 분석 | `data, annot, fmt, cmap, center, linewidths` | 다중공선성 파악 및 피처 선택 | `sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')` |
| `clustermap()` | 피처 클러스터링 | `data, method, metric, cmap, row_cluster, col_cluster` | 유사한 피처 그룹화 및 차원 축소 | `sns.clustermap(data, cmap='viridis')` |

### 회귀 플롯 (Regression Plots) - 회귀 분석 시각화

| 함수 | ML 문제 유형 | 주요 매개변수 | 문제 해결 패턴 | 예시 코드 |
|------|--------------|--------------|--------------|----------|
| `lmplot()` | 다중 회귀 관계 분석 | `x, y, hue, col, row, data, ci, order` | 여러 그룹 간 회귀 관계 비교 | `sns.lmplot(data=df, x='feature', y='target', col='category')` |
| `regplot()` | 단일 회귀 관계 분석 | `x, y, data, ci, scatter, order` | 피처와 타겟 간 선형 관계 확인 | `sns.regplot(data=df, x='feature', y='target')` |
| `residplot()` | 잔차 분석 및 모델 가정 검증 | `x, y, data, lowess` | 회귀 모델의 잔차 패턴 분석 | `sns.residplot(x=y_pred, y=y_test-y_pred)` |

### 모델 평가 플롯 (Model Evaluation Plots)

| 함수 | ML 문제 유형 | 주요 매개변수 | 문제 해결 패턴 | 예시 코드 |
|------|--------------|--------------|--------------|----------|
| `confusion_matrix` + `heatmap()` | 분류 모델 성능 평가 | `y_true, y_pred` | 혼동 행렬 시각화로 분류 성능 상세 분석 | `sns.heatmap(cm, annot=True, fmt='d')` |
| `roc_curve` + `lineplot()` | 분류 모델 임계값 분석 | `y_true, y_score` | ROC 곡선으로 분류기 성능 비교 | `sns.lineplot(x=fpr, y=tpr)` |
| `scatterplot()` (실제 vs 예측) | 회귀 모델 성능 평가 | `x, y, data` | 실제값과 예측값 비교로 회귀 성능 평가 | `sns.scatterplot(x=y_true, y=y_pred)` |

## 머신러닝 문제 해결을 위한 색상 팔레트

### ML 문제 유형별 팔레트 선택

```python
# 분류 문제 - 클래스 구별을 위한 팔레트
sns.color_palette("Set2", 8)        # 클래스별 명확한 구분
sns.color_palette("colorblind", 8)  # 색맹 친화적 클래스 구분

# 회귀 문제 - 연속값 표현을 위한 팔레트
sns.color_palette("viridis", 10)    # 순차적 값 표현
sns.color_palette("coolwarm", 10)   # 발산적 값 표현 (양/음)

# 상관관계 분석 - 중앙값 기준 팔레트
sns.color_palette("RdBu_r", 10)     # 상관관계 히트맵용
sns.color_palette("vlag", 10)       # 양/음 값 명확한 구분
```

### 문제 해결 상황별 팔레트 사용

```python
# 이상치 탐지 - 정상/이상치 구별
normal_outlier_palette = ["#4C72B0", "#C44E52"]  # 파랑(정상)/빨강(이상치)

# 피처 중요도 - 중요도 순위 표현
importance_palette = sns.light_palette("navy", reverse=True, as_cmap=True)

# 모델 성능 비교 - 다중 모델 구별
model_palette = sns.color_palette("husl", len(models))

# 시계열 데이터 - 시간 흐름 표현
time_palette = sns.cubehelix_palette(start=0.5, rot=-0.75)
```

## 스타일링

### 테마 및 컨텍스트

```python
# 테마 설정
sns.set_theme(style="whitegrid")    # darkgrid, whitegrid, dark, white, ticks
sns.set_theme(style="white", rc={"axes.spines.right": False, "axes.spines.top": False})

# 컨텍스트 설정
sns.set_context("paper")            # paper, notebook, talk, poster
sns.set_context("notebook", font_scale=1.2)  # 폰트 크기 조정
```

### 개별 요소 제어

```python
# 축 스타일
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.3)

# 제목과 레이블
ax.set_title("제목", fontsize=14, fontweight='bold')
ax.set_xlabel("X축 레이블", fontsize=12)
ax.set_ylabel("Y축 레이블", fontsize=12)

# 범례
ax.legend(title="범례", bbox_to_anchor=(1.05, 1), loc='upper left')
```

## 데이터 처리

### 데이터 준비

```python
# 결측치 처리
df = df.dropna()                    # 결측치 제거
df = df.fillna(df.mean())           # 평균으로 대체

# 데이터 타입 변환
df['category'] = df['category'].astype('category')
df['numeric_col'] = pd.to_numeric(df['numeric_col'])

# 파생 변수 생성
df['new_col'] = df['col1'] / df['col2']
df['category'] = pd.cut(df['value'], bins=[0, 10, 20, 30], labels=['Low', 'Medium', 'High'])
```

### 데이터 집계

```python
# 그룹별 집계
grouped = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
pivot_table = df.pivot_table(values='value', index='category', columns='region', aggfunc='mean')

# 시계열 데이터
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
monthly_data = df.groupby('month')['value'].sum()
```

## 머신러닝 문제 해결을 위한 빠른 참조 코드

### 데이터 탐색 및 품질 진단 템플릿

```python
# 데이터 품질 진단
def diagnose_data_quality(df, target_col=None):
    """데이터 품질 종합 진단"""
    print(f"데이터 크기: {df.shape}")
    print(f"결측치: {df.isnull().sum().sum()}개 ({df.isnull().sum().sum()/df.size:.2%})")
    
    # 결측치 시각화
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('결측치 패턴')
    plt.show()
    
    # 수치형 변수 분포
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, col in enumerate(numeric_cols[:4]):
            row, col_idx = i // 2, i % 2
            sns.boxplot(data=df, y=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'{col} 분포 (이상치 확인)')
        plt.tight_layout()
        plt.show()

# 클래스별 피처 분포 비교 (분류 문제)
def compare_class_distributions(df, target_col, feature_cols):
    """클래스별 피처 분포 비교"""
    n_features = len(feature_cols)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, feature in enumerate(feature_cols[:4]):
        row, col_idx = i // 2, i % 2
        sns.boxplot(data=df, x=target_col, y=feature, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'클래스별 {feature} 분포')
    
    plt.tight_layout()
    plt.show()
```

### 모델 평가 및 해석 템플릿

```python
# 분류 모델 성능 평가
def evaluate_classification_model(y_true, y_pred, y_proba=None):
    """분류 모델 성능 종합 평가"""
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('혼동 행렬')
    plt.ylabel('실제')
    plt.xlabel('예측')
    plt.show()
    
    # ROC 곡선 (확률 제공 시)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=fpr, y=tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--')
        plt.title('ROC 곡선')
        plt.xlabel('위양성률')
        plt.ylabel('진양성률')
        plt.legend()
        plt.show()
    
    # 분류 보고서
    print(classification_report(y_true, y_pred))

# 회귀 모델 성능 평가
def evaluate_regression_model(y_true, y_pred):
    """회귀 모델 성능 종합 평가"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    # 실제값 vs 예측값
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('실제값 vs 예측값')
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    plt.show()
    
    # 잔차 분석
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, ax=axes[0])
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title('잔차 분석')
    axes[0].set_xlabel('예측값')
    axes[0].set_ylabel('잔차')
    
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title('잔차 분포')
    
    plt.tight_layout()
    plt.show()
    
    # 성능 지표
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

# 피처 중요도 시각화
def plot_feature_importance(feature_names, importances, top_n=10):
    """피처 중요도 시각화"""
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature')
    plt.title(f'상위 {top_n}개 중요 특성')
    plt.tight_layout()
    plt.show()
```

### 모델 성능 모니터링 템플릿

```python
# 시계열 성능 모니터링
def monitor_model_performance(performance_df, metric_col):
    """모델 성능 시계열 모니터링"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=performance_df, x='date', y=metric_col)
    plt.title(f'모델 {metric_col} 추이')
    plt.ylabel(metric_col)
    
    # 통계적 과정 관리 (SPC)
    mean_val = performance_df[metric_col].mean()
    std_val = performance_df[metric_col].std()
    
    plt.axhline(mean_val, color='green', linestyle='-', alpha=0.7, label='평균')
    plt.axhline(mean_val + 2*std_val, color='red', linestyle='--', alpha=0.7, label='관리 상한')
    plt.axhline(mean_val - 2*std_val, color='red', linestyle='--', alpha=0.7, label='관리 하한')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
```

## 머신러닝 문제 해결 체크리스트

### 데이터 탐색 및 준비 단계

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 데이터 분포 왜곡 | 정규분포 가정 위배 | 로그/제곱근 변환 또는 `PowerTransformer` 사용 |
| 클래스 불균형 | 한 클래스의 샘플 부족 | `SMOTE` 오버샘플링 또는 클래스 가중치 조정 |
| 다중공선성 | 피처 간 높은 상관관계 | `VIF` 분석 후 피처 제거 또는 차원 축소 |
| 이상치 다수 | 측정 오류 또는 극단값 | `IQR` 기반 제거 또는 `RobustScaler` 사용 |

### 모델링 단계

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 과적합 | 모델 복잡도 과도 | 교차 검증, 규제 또는 드롭아웃 적용 |
| 과소적합 | 모델 복잡도 부족 | 피처 엔지니어링 또는 더 복잡한 모델 사용 |
| 피처 중요도 불명확 | 비선형 관계 | `SHAP` 값 또는 `Permutation Importance` 사용 |
| 하이퍼파라미터 튜닝 어려움 | 탐색 공간 크기 | `GridSearchCV` 또는 `RandomizedSearchCV` 사용 |

### 모델 평가 단계

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 단일 지표만으로 평가 | 평가 지표의 한계 | 정밀도/재현율, ROC-AUC 등 다양한 지표 사용 |
| 교차 검증 결과 불안정 | 데이터 분할 편향 | `StratifiedKFold` 또는 반복 교차 검증 사용 |
| 잔차 패턴 존재 | 모델 가정 위배 | 비선형 모델로 전환 또는 피처 변환 |

### 모델 운영 및 모니터링 단계

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 모델 성능 저하 | 데이터 분포 변화 (드리프트) | 정기적 재학습 및 성능 모니터링 |
| 예측 결과 해석 어려움 | 블랙박스 모델 | `SHAP`, `LIME` 등 설명 가능한 AI 기법 사용 |
| 대용량 데이터 예측 지연 | 모델 복잡도 또는 인프라 한계 | 모델 경량화 또는 배치 처리 방식 도입 |

## 성능 최적화 팁

### 대용량 데이터

```python
# 샘플링
sampled_df = df.sample(n=10000, random_state=42)

# 효율적 플롯
plt.hexbin(df['x'], df['y'], gridsize=30)  # 산점도 대신
sns.kdeplot(data=df.sample(10000), x='x', y='y')  # 밀도 플롯

# 데이터 타입 최적화
df['int_col'] = df['int_col'].astype('int32')  # int64 → int32
df['cat_col'] = df['cat_col'].astype('category')  # object → category
```

### 메모리 관리

```python
# 플롯 객체 정리
plt.close('all')  # 모든 플롯 닫기

# 가비지 컬렉션
import gc
gc.collect()
```

## 유용한 단축키 및 팁

### Jupyter Notebook 단축키

| 단축키 | 기능 |
|--------|------|
| `Shift + Enter` | 셀 실행 |
| `Ctrl + Enter` | 셀 실행 후 유지 |
| `Alt + Enter` | 셀 실행 후 새 셀 추가 |
| `M` | 마크다운 셀로 변환 |
| `Y` | 코드 셀로 변환 |
| `A` | 위에 셀 추가 |
| `B` | 아래에 셀 추가 |
| `D, D` | 셀 삭제 |

### 코딩 팁

```python
# 체이닝으로 코드 간결화
df.groupby('category')['value'].mean().sort_values(ascending=False).head()

# 여러 플롯 한 번에 생성
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, col in zip(axes.flat, ['col1', 'col2', 'col3', 'col4']):
    sns.histplot(df[col], ax=ax)

# 함수로 재사용 가능한 플롯
def create_custom_plot(data, x, y, hue=None, title=""):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    return plt.gcf()
```

## 머신러닝 관련 추가 자료

### 공식 문서
- [Seaborn 공식 문서](https://seaborn.pydata.org/)
- [Seaborn API 참조](https://seaborn.pydata.org/api.html)
- [Seaborn 예제 갤러리](https://seaborn.pydata.org/examples/index.html)

### 머신러닝 관련 라이브러리
- [Scikit-learn 문서](https://scikit-learn.org/stable/)
- [Scikit-learn 모델 평가 가이드](https://scikit-learn.org/stable/model_evaluation.html)
- [Pandas 문서](https://pandas.pydata.org/docs/)
- [NumPy 문서](https://numpy.org/doc/stable/)

### 모델 해석 및 설명 가능한 AI
- [SHAP 라이브러리 문서](https://shap.readthedocs.io/)
- [LIME 라이브러리 문서](https://lime-ml.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

### 머신러닝 운영 (MLOps)
- [MLOps.org](https://ml-ops.org/)
- [Kubeflow 문서](https://www.kubeflow.org/docs/)
- [MLflow 문서](https://mlflow.org/docs/latest/index.html)

### 커뮤니티 및 튜토리얼
- [Stack Overflow Seaborn 태그](https://stackoverflow.com/questions/tagged/seaborn)
- [Towards Data Science - Seaborn 튜토리얼](https://towardsdatascience.com/tagged/seaborn)
- [Kaggle Notebooks - 데이터 시각화](https://www.kaggle.com/notebooks)

### 프로젝트 내 관련 자료
- [머신러닝 문제 해결 프레임워크](ml-problem-solution-framework.md)
- [머신러닝 실전 문제 해결 예시](ml-practical-examples.md)
- [문제 해결 중심 기본 플롯](03-basic-plots-problem-solution.md)
- [실용적인 문제 해결 가이드](10-troubleshooting-practical.md)

---

이 참조 요약은 머신러닝 프로젝트에서 Seaborn을 활용하여 문제를 해결하는 데 필요한 핵심 기능들을 빠르게 참조할 수 있도록 구성되었습니다. 자세한 내용은 각 기능의 공식 문서와 프로젝트 내 관련 자료를 참조하세요.