# Seaborn 문제 해결 실용 가이드

이 문서는 Seaborn을 사용하면서 마주할 수 있는 일반적인 문제들과 해결 방안을 정리하며, 특히 머신러닝 프로젝트에서의 실제 문제 해결 사례를 다룹니다.

## 목차

1. [일반적인 문제 해결](#일반적인-문제-해결)
2. [머신러닝 프로젝트에서의 문제 해결](#머신러닝-프로젝트에서의-문제-해결)
3. [디버깅 기법](#디버깅-기법)
4. [문제 해결 체크리스트](#문제-해결-체크리스트)

## 일반적인 문제 해결

### 1. 데이터 관련 오류

#### 결측치로 인한 오류

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 결측치가 포함된 데이터 생성
data_with_na = pd.DataFrame({
    'x': [1, 2, np.nan, 4, 5],
    'y': [2, 3, 4, np.nan, 6],
    'category': ['A', 'B', 'C', 'A', 'B']
})

# 문제 상황
try:
    sns.scatterplot(data=data_with_na, x='x', y='y')
    print("결측치가 있어도 기본적으로 처리됩니다")
except Exception as e:
    print(f"오류 발생: {e}")

# 해결 방안 1: 결측치 제거
data_cleaned = data_with_na.dropna()
print(f"결측치 제거 후: {len(data_with_na)} -> {len(data_cleaned)}")

# 해결 방안 2: 결측치 대체
data_filled = data_with_na.fillna(data_with_na.mean())
print(f"결측치 대체 완료")

# 해결 방안 3: 플롯 함수에서 결측치 처리
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본 데이터
sns.scatterplot(data=data_with_na, x='x', y='y', ax=axes[0])
axes[0].set_title("원본 데이터 (자동 처리)")

# 결측치 제거
sns.scatterplot(data=data_cleaned, x='x', y='y', ax=axes[1])
axes[1].set_title("결측치 제거")

# 결측치 대체
sns.scatterplot(data=data_filled, x='x', y='y', ax=axes[2])
axes[2].set_title("결측치 대체")

plt.tight_layout()
plt.show()
```

#### 데이터 타입 불일치

```python
# 데이터 타입 문제
mixed_type_data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': ['2', '3', '4', '5', '6'],  # 문자열
    'category': [1, 2, 1, 2, 1]      # 정수지만 카테고리
})

# 문제 상황
try:
    sns.scatterplot(data=mixed_type_data, x='x', y='y')
    print("문제 없음: Seaborn이 자동으로 변환")
except Exception as e:
    print(f"오류 발생: {e}")

# 해결 방안: 명시적 타입 변환
mixed_type_data['y'] = pd.to_numeric(mixed_type_data['y'])
mixed_type_data['category'] = mixed_type_data['category'].astype('category')

print("데이터 타입:")
print(mixed_type_data.dtypes)

# 타입 변환 후 플롯
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mixed_type_data, x='x', y='y', hue='category')
plt.title("타입 변환 후 플롯")
plt.show()
```

### 2. 플롯 관련 오류

#### 플롯 크기 및 레이아웃 문제

```python
# 레이아웃 문제 예시
import matplotlib.pyplot as plt

# 문제 1: 플롯이 잘림
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.set_title(f"플롯 {i+1} - 매우 긴 제목으로 인해 레이아웃 문제 발생 가능")
    ax.plot([1, 2, 3], [1, 4, 2])

# 해결 방안 1: tight_layout 사용
plt.tight_layout()
plt.show()

# 문제 2: 범례가 플롯을 가림
fig, ax = plt.subplots(figsize=(6, 4))
categories = ['카테고리A', '카테고리B', '카테고리C', '카테고리D', '카테고리E']
for i, cat in enumerate(categories):
    ax.plot([1, 2, 3], [i+1, i+2, i+1.5], label=cat, linewidth=3)

ax.legend(title='범례')
ax.set_title("범례가 플롯을 가리는 문제")

# 해결 방안 2: 범례 위치 조정
fig, ax = plt.subplots(figsize=(6, 4))
for i, cat in enumerate(categories):
    ax.plot([1, 2, 3], [i+1, i+2, i+1.5], label=cat, linewidth=3)

ax.legend(title='범례', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title("범례 위치 조정")
plt.tight_layout()
plt.show()
```

#### 색상 및 스타일 문제

```python
# 색상 관련 문제
color_problem_data = pd.DataFrame({
    'x': range(20),
    'y': np.random.randn(20),
    'category': ['A']*10 + ['B']*10
})

# 문제 1: 너무 많은 카테고리
many_categories = pd.DataFrame({
    'x': range(100),
    'y': np.random.randn(100),
    'category': [f'카테고리{i}' for i in range(25)]
})

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# 문제 상황: 너무 많은 카테고리로 색상 구분 어려움
sns.scatterplot(data=many_categories, x='x', y='y', hue='category')
plt.title("너무 많은 카테고리 (문제)")

# 해결 방안
plt.subplot(1, 2, 2)
# 상위 카테고리만 표시하거나 색상 팔레트 조정
top_categories = many_categories['category'].value_counts().nlargest(5).index
filtered_data = many_categories[many_categories['category'].isin(top_categories)]
sns.scatterplot(data=filtered_data, x='x', y='y', hue='category', palette='deep')
plt.title("카테고리 필터링 (해결)")

plt.tight_layout()
plt.show()

# 문제 2: 색맹 접근성
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# 문제 상황: 색맹에게 구별 어려운 색상
sns.scatterplot(data=color_problem_data, x='x', y='y', hue='category', 
                palette=['red', 'green'])
plt.title("색맹 접근성 문제")

# 해결 방안
plt.subplot(1, 2, 2)
sns.scatterplot(data=color_problem_data, x='x', y='y', hue='category', 
                palette='colorblind')
plt.title("색맹 친화적 팔레트")

plt.tight_layout()
plt.show()
```

### 3. 성능 관련 문제

#### 대용량 데이터 처리

```python
# 대용량 데이터 성능 문제
large_data = pd.DataFrame({
    'x': np.random.randn(100000),
    'y': np.random.randn(100000),
    'category': np.random.choice(['A', 'B', 'C'], 100000)
})

# 문제 상황: 대용량 데이터로 인한 느린 렌더링
import time

def plot_performance_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 전체 데이터 (느림)
    start_time = time.time()
    sns.scatterplot(data=large_data, x='x', y='y', s=1, alpha=0.3, ax=axes[0])
    full_time = time.time() - start_time
    axes[0].set_title(f"전체 데이터 (n=100,000)\n{full_time:.2f}초")
    
    # 샘플링 (빠름)
    start_time = time.time()
    sampled_data = large_data.sample(10000, random_state=42)
    sns.scatterplot(data=sampled_data, x='x', y='y', s=5, alpha=0.5, ax=axes[1])
    sample_time = time.time() - start_time
    axes[1].set_title(f"샘플링 (n=10,000)\n{sample_time:.2f}초")
    
    # 효율적 플롯 (가장 빠름)
    start_time = time.time()
    axes[2].hexbin(large_data['x'], large_data['y'], gridsize=30, cmap='viridis')
    hexbin_time = time.time() - start_time
    axes[2].set_title(f"육각형 빈 플롯 (n=100,000)\n{hexbin_time:.2f}초")
    
    plt.tight_layout()
    return fig

fig = plot_performance_comparison()
plt.show()

print(f"성능 비교:")
print(f"전체 데이터: 샘플링보다 {10000/full_time:.0f}배 느림")
print(f"육각형 빈 플롯: 전체 데이터보다 {full_time/hexbin_time:.0f}배 빠름")
```

### 4. 통계 관련 문제

#### 통계 계산 오류

```python
# 통계 관련 문제
statistical_data = pd.DataFrame({
    'group': ['A']*5 + ['B']*5,
    'value': [1, 2, 3, 4, 5, 1, 1, 1, 1, 20]  # 이상치 포함
})

# 문제 상황: 이상치로 인한 왜곡된 통계
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 데이터
sns.boxplot(data=statistical_data, x='group', y='value', ax=axes[0])
axes[0].set_title("이상치 포함 (왜곡된 통계)")

# 해결 방안 1: 이상치 제거
filtered_data = statistical_data[statistical_data['value'] < 20]
sns.boxplot(data=filtered_data, x='group', y='value', ax=axes[1])
axes[1].set_title("이상치 제거")

# 해결 방안 2: 강건한 통계 사용
sns.boxplot(data=statistical_data, x='group', y='value', showfliers=False, ax=axes[2])
axes[2].set_title("이상치 표시 안함")

plt.tight_layout()
plt.show()

# 통계 값 비교
print("그룹별 통계:")
print(statistical_data.groupby('group')['value'].describe())
print("\n이상치 제거 후 통계:")
print(filtered_data.groupby('group')['value'].describe())
```

## 머신러닝 프로젝트에서의 문제 해결

### 문제: 머신러닝 모델링 전 데이터 품질 진단

**상황**: 분류 모델링을 위해 데이터를 준비했지만, 데이터 품질이 의심스러워 시각화로 확인해야 함

**해결책**: 다양한 시각화 기법으로 데이터 품질 종합 진단

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# 문제 상황: 의료 진단 분류 모델링 전 데이터 품질 진단
# 데이터 생성 (다양한 데이터 품질 문제 포함)
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, 
                          n_redundant=1, n_clusters_per_class=2, 
                          class_sep=1.0, random_state=42)

# 데이터프레임 생성
feature_names = ['혈압', '콜레스테롤', '혈당', 'BMI', '나이']
df = pd.DataFrame(X, columns=feature_names)
df['진단결과'] = np.where(y == 0, '정상', '질환')

# 인위적인 데이터 품질 문제 추가
# 1. 결측치 추가
missing_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_indices, '혈압'] = np.nan

# 2. 이상치 추가
outlier_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[outlier_indices, '콜레스테롤'] *= 3

# 3. 데이터 타입 불일치
df.loc[df.sample(30).index, 'BMI'] = [f"{val}" for val in df.loc[df.sample(30).index, 'BMI']]

# 데이터 품질 진단 함수
def diagnose_data_quality(df, target_col=None):
    """데이터 품질 종합 진단"""
    print("=" * 50)
    print("데이터 품질 진단 보고서")
    print("=" * 50)
    
    # 기본 정보
    print(f"데이터 크기: {df.shape}")
    print(f"결측치: {df.isnull().sum().sum()}개 ({df.isnull().sum().sum()/df.size:.2%})")
    
    # 결측치 상세 정보
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("\n결측치 상세 정보:")
        for col, count in missing_info[missing_info > 0].items():
            print(f"  - {col}: {count}개 ({count/len(df):.2%})")
    
    # 데이터 타입 정보
    print(f"\n데이터 타입:")
    print(df.dtypes)
    
    # 수치형 변수 기술 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n수치형 변수 기술 통계:")
        print(df[numeric_cols].describe().round(2))
    
    # 범주형 변수 정보
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\n범주형 변수 정보:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  - {col}: {unique_count}개 고유값")
            if unique_count <= 10:
                print(f"    값 분포: {df[col].value_counts().to_dict()}")

# 데이터 품질 진단 실행
diagnose_data_quality(df, target_col='진단결과')

# 시각화를 통한 데이터 품질 진단
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 결측치 시각화
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=axes[0, 0])
axes[0, 0].set_title('결측치 패턴')

# 2. 수치형 변수 분포 (이상치 확인)
numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # 처음 3개
for i, col in enumerate(numeric_cols):
    if i < 3:
        row, col_idx = 0, i+1
        sns.boxplot(data=df, y=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} 분포 (이상치 확인)')

# 3. 타겟 변수 분포
if '진단결과' in df.columns:
    sns.countplot(data=df, x='진단결과', ax=axes[1, 0])
    axes[1, 0].set_title('타겟 변수 분포')
    axes[1, 0].set_ylabel('개수')

# 4. 상관관계 히트맵 (다중공선성 확인)
numeric_data = df.select_dtypes(include=[np.number])
if len(numeric_data.columns) > 1:
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('상관관계 행렬')

# 5. 클래스별 피처 분포
if '진단결과' in df.columns and len(numeric_cols) > 0:
    feature = numeric_cols[0]
    sns.histplot(data=df.dropna(subset=[feature]), x=feature, hue='진단결과', 
                kde=True, alpha=0.7, ax=axes[1, 2])
    axes[1, 2].set_title(f'클래스별 {feature} 분포')

plt.tight_layout()
plt.show()

# 문제 해결 방안 시연
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 해결책 1: 결측치 처리
df_clean = df.copy()
df_clean['혈압'] = df_clean['혈압'].fillna(df_clean['혈압'].median())

sns.boxplot(data=df, y='혈압', ax=axes[0])
axes[0].set_title('결측치 처리 전 (혈압)')

sns.boxplot(data=df_clean, y='혈압', ax=axes[1])
axes[1].set_title('결측치 처리 후 (혈압)')

# 해결책 2: 이상치 처리
df_clean = df_clean.copy()
Q1 = df_clean['콜레스테롤'].quantile(0.25)
Q3 = df_clean['콜레스테롤'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean['콜레스테롤'] = df_clean['콜레스테롤'].clip(lower_bound, upper_bound)

sns.boxplot(data=df_clean, y='콜레스테롤', ax=axes[2])
axes[2].set_title('이상치 처리 후 (콜레스테롤)')

plt.tight_layout()
plt.show()

print("\n데이터 정제 결과:")
print(f"정제 전 결측치: {df.isnull().sum().sum()}개")
print(f"정제 후 결측치: {df_clean.isnull().sum().sum()}개")
```

### 문제: 피처 스케일링 효과 시각화

**상황**: 피처 스케일링이 머신러닝 모델 성능에 미치는 영향을 시각적으로 확인해야 함

**해결책**: 스케일링 전후 분포 비교와 모델 성능 시각화

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 문제 상황: 다양한 스케일링 기법이 모델 성능에 미치는 영향 분석
# 데이터 생성
np.random.seed(42)
n_samples = 500

# 다양한 스케일을 가진 피처 생성
data = {
    '소득': np.random.lognormal(10, 1, n_samples),  # 큰 스케일
    '나이': np.random.normal(40, 12, n_samples),    # 중간 스케일
    '경력년수': np.random.exponential(5, n_samples), # 왜곡된 분포
    '신용점수': np.random.normal(650, 100, n_samples)  # 정규분포
}

# 타겟 변수 생성 (소득과 신용점수에 영향을 받음)
df_scale = pd.DataFrame(data)
df_scale['대출승인'] = ((df_scale['소득'] > np.exp(9.5)) & 
                       (df_scale['신용점수'] > 600)).astype(int)

# 스케일링 기법 비교
scalers = {
    '원본': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

results = []

for scaler_name, scaler in scalers.items():
    # 데이터 준비
    X = df_scale[['소득', '나이', '경력년수', '신용점수']].copy()
    y = df_scale['대출승인']
    
    # 스케일링 적용
    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X_scaled = X.copy()
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # 모델 훈련 및 평가
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    results.append({
        '스케일러': scaler_name,
        '훈련 정확도': train_acc,
        '테스트 정확도': test_acc
    })

# 스케일링 효과 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 원본 데이터 분포
for i, feature in enumerate(['소득', '나이', '경력년수']):
    sns.histplot(df_scale[feature], kde=True, ax=axes[0, i])
    axes[0, i].set_title(f'원본 {feature} 분포')

# 스케일링 후 분포 (StandardScaler)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_scale[['소득', '나이', '경력년수']]), 
                         columns=['소득', '나이', '경력년수'])

for i, feature in enumerate(['소득', '나이', '경력년수']):
    sns.histplot(df_scaled[feature], kde=True, ax=axes[1, i])
    axes[1, i].set_title(f'StandardScaler 후 {feature} 분포')

plt.tight_layout()
plt.show()

# 스케일러별 성능 비교
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='스케일러', y='테스트 정확도', palette='viridis')
plt.title('스케일러별 모델 성능 비교')
plt.ylabel('테스트 정확도')
plt.ylim(0, 1)
plt.show()

print("스케일러별 성능 비교:")
print(results_df.round(3))
```

### 문제: 고차원 데이터 시각화

**상황**: 10개 이상의 피처를 가진 데이터를 효과적으로 시각화해야 함

**해결책**: 차원 축소 기법과 다중 플롯 그리드 활용

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# 문제 상황: 고객 세분화를 위한 고차원 데이터 시각화
# 고차원 데이터 생성
X, y = make_blobs(n_samples=500, centers=4, n_features=10, 
                  cluster_std=1.5, random_state=42)

feature_names = [f'피처_{i+1}' for i in range(10)]
df_high_dim = pd.DataFrame(X, columns=feature_names)
df_high_dim['클러스터'] = y

# 차원 축소를 통한 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_high_dim['PCA1'] = X_pca[:, 0]
df_high_dim['PCA2'] = X_pca[:, 1]

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
df_high_dim['tSNE1'] = X_tsne[:, 0]
df_high_dim['tSNE2'] = X_tsne[:, 1]

# 고차원 데이터 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 상관관계 히트맵 (피처 간 관계)
corr_matrix = df_high_dim[feature_names].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=axes[0, 0])
axes[0, 0].set_title('피처 간 상관관계')

# 2. PCA 시각화
sns.scatterplot(data=df_high_dim, x='PCA1', y='PCA2', 
                hue='클러스터', palette='deep', s=50, alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title(f'PCA 시각화 (설명력: {pca.explained_variance_ratio_.sum():.2f})')

# 3. t-SNE 시각화
sns.scatterplot(data=df_high_dim, x='tSNE1', y='tSNE2', 
                hue='클러스터', palette='deep', s=50, alpha=0.7, ax=axes[0, 2])
axes[0, 2].set_title('t-SNE 시각화')

# 4-6. 주요 피처들의 분포
important_features = ['피처_1', '피처_2', '피처_3']
for i, feature in enumerate(important_features):
    row, col = 1, i
    sns.boxplot(data=df_high_dim, x='클러스터', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'클러스터별 {feature} 분포')

plt.tight_layout()
plt.show()

# 설명된 분산 비율 시각화
plt.figure(figsize=(10, 6))
pca_full = PCA().fit(X)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('주성분 수')
plt.ylabel('누적 설명 분산 비율')
plt.title('PCA 주성분 수에 따른 설명 분산 비율')
plt.grid(True, alpha=0.3)
plt.show()

print("차원 축소 결과:")
print(f"PCA 2개 주성분으로 설명되는 분산: {pca.explained_variance_ratio_.sum():.2f}")
```

### 문제: 대용량 데이터 시각화 성능 최적화

**상황**: 10만 개 이상의 데이터 포인트를 시각화할 때 성능 문제 발생

**해결책**: 샘플링, 효율적 플롯 유형, 데이터 집계 기법 활용

```python
# 문제 상황: 대용량 거래 데이터 시각화 성능 최적화
# 대용량 데이터 생성
np.random.seed(42)
n_large = 100000

large_data = pd.DataFrame({
    '거래금액': np.random.lognormal(8, 1.5, n_large),
    '거래시간': pd.date_range('2023-01-01', periods=n_large, freq='H'),
    '고객ID': np.random.randint(1, 1000, n_large),
    '상품카테고리': np.random.choice(['전자제품', '의류', '식품', '가구'], n_large),
    '결제수단': np.random.choice(['신용카드', '계좌이체', '모바일결제'], n_large)
})

# 시간 패턴 추가
large_data['거래금액'] *= (1 + 0.3 * np.sin(np.arange(n_large) * 2 * np.pi / 24))  # 일일 패턴

# 성능 최적화 기법 비교
import time

def plot_performance_comparison():
    """다양한 시각화 기법의 성능 비교"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 전체 데이터 산점도 (느림)
    start_time = time.time()
    sample_subset = large_data.sample(n=5000, random_state=42)  # 성능을 위해 샘플링
    sns.scatterplot(data=sample_subset, x='거래시간', y='거래금액', 
                    alpha=0.5, s=10, ax=axes[0, 0])
    full_time = time.time() - start_time
    axes[0, 0].set_title(f'전체 데이터 (샘플링): {full_time:.2f}초')
    
    # 2. 집계 데이터 시각화 (빠름)
    start_time = time.time()
    daily_agg = large_data.groupby(large_data['거래시간'].dt.date).agg({
        '거래금액': ['mean', 'count']
    }).reset_index()
    daily_agg.columns = ['날짜', '평균거래금액', '거래건수']
    
    sns.lineplot(data=daily_agg, x='날짜', y='평균거래금액', ax=axes[0, 1])
    agg_time = time.time() - start_time
    axes[0, 1].set_title(f'집계 데이터: {agg_time:.2f}초')
    
    # 3. 2D 밀도 히스토그램 (중간)
    start_time = time.time()
    # 날짜를 숫자로 변환
    large_data_copy = large_data.copy()
    large_data_copy['거래시간_num'] = large_data_copy['거래시간'].astype(int) // 10**9
    
    hb = axes[1, 0].hexbin(large_data_copy['거래시간_num'], 
                           large_data_copy['거래금액'], 
                           gridsize=20, cmap='viridis')
    axes[1, 0].set_title(f'2D 히스토그램: {time.time() - start_time:.2f}초')
    plt.colorbar(hb, ax=axes[1, 0])
    
    # 4. 카테고리별 집계 시각화
    start_time = time.time()
    category_agg = large_data.groupby('상품카테고리')['거래금액'].mean().reset_index()
    sns.barplot(data=category_agg, x='상품카테고리', y='거래금액', ax=axes[1, 1])
    category_time = time.time() - start_time
    axes[1, 1].set_title(f'카테고리별 집계: {category_time:.2f}초')
    
    plt.tight_layout()
    plt.show()
    
    return {
        '샘플링 산점도': full_time,
        '집계 데이터': agg_time,
        '2D 히스토그램': time.time() - start_time,
        '카테고리 집계': category_time
    }

# 성능 비교 실행
performance_results = plot_performance_comparison()

print("시각화 기법별 성능 비교:")
for method, time_taken in performance_results.items():
    print(f"- {method}: {time_taken:.3f}초")

# 메모리 효율적 시각화 팁
print("\n대용량 데이터 시각화 팁:")
print("1. 데이터 샘플링: 전체 데이터의 일부만 시각화")
print("2. 집계: 데이터를 그룹별로 요약하여 시각화")
print("3. 효율적 플롯: hexbin, 2D 밀도 플롯 등 사용")
print("4. 데이터 타입 최적화: 범주형 데이터는 category 타입으로 변환")
print("5. 적절한 해상도: 너무 세밀한 플롯 피하기")
```

### 문제: 분류 모델 성능 종합 평가

**상황**: 단일 정확도 지표만으로는 분류 모델 성능을 충분히 평가할 수 없음

**해결책**: 다양한 성능 지표와 시각화를 통한 종합적 모델 평가

```python
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# 문제 상황: 의료 진단 분류 모델의 성능을 다각적으로 평가
# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_redundant=2, n_clusters_per_class=2, 
                          weights=[0.7, 0.3], random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                  stratify=y, random_state=42)

# 모델 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 종합적인 모델 평가 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['예측 정상', '예측 질환'], 
            yticklabels=['실제 정상', '실제 질환'], ax=axes[0, 0])
axes[0, 0].set_title('혼동 행렬')

# 2. ROC 곡선
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
sns.lineplot(x=fpr, y=tpr, ax=axes[0, 1])
sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', ax=axes[0, 1])
axes[0, 1].set_title(f'ROC 곡선 (AUC = {roc_auc:.3f})')
axes[0, 1].set_xlabel('위양성률 (FPR)')
axes[0, 1].set_ylabel('진양성률 (TPR)')

# 3. 정밀도-재현율 곡선
precision, recall, _ = precision_recall_curve(y_test, y_proba)
sns.lineplot(x=recall, y=precision, ax=axes[0, 2])
axes[0, 2].set_title('정밀도-재현율 곡선')
axes[0, 2].set_xlabel('재현율')
axes[0, 2].set_ylabel('정밀도')

# 4. 특성 중요도
feature_importance = pd.DataFrame({
    'feature': [f'피처_{i+1}' for i in range(X.shape[1])],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=axes[1, 0])
axes[1, 0].set_title('상위 10개 중요 특성')

# 5. 예측 확률 분포
pred_df = pd.DataFrame({
    '실제값': y_test,
    '예측확률': y_proba
})
pred_df['클래스'] = pred_df['실제값'].map({0: '정상', 1: '질환'})

sns.histplot(data=pred_df, x='예측확률', hue='클래스', 
             kde=True, alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title('클래스별 예측 확률 분포')
axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7)

# 6. 교차 검증 성능 분포
cv_pred = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
cv_pred_binary = (cv_pred > 0.5).astype(int)
cv_accuracy = np.mean(cv_pred_binary == y_train)

sns.histplot(cv_pred[y_train == 0], kde=True, alpha=0.5, label='정상', ax=axes[1, 2])
sns.histplot(cv_pred[y_train == 1], kde=True, alpha=0.5, label='질환', ax=axes[1, 2])
axes[1, 2].set_title(f'교차 검증 예측 확률 (정확도: {cv_accuracy:.3f})')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# 성능 지표 출력
print("분류 성능 보고서:")
print(classification_report(y_test, y_pred, target_names=['정상', '질환']))
```

### 문제: 회귀 모델 오차 패턴 분석

**상황**: 회귀 모델의 예측 오차에 패턴이 있는지 분석하여 모델 개선 방향 찾기

**해결책**: 잔차 분석과 오차 패턴 시각화

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

# 문제 상황: 주택 가격 예측 모델의 오차 패턴 분석
# 데이터 생성
np.random.seed(42)
n_houses = 1000

house_data = pd.DataFrame({
    '면적': np.random.normal(120, 30, n_houses),
    '방수': np.random.poisson(3, n_houses) + 1,
    '나이': np.random.exponential(20, n_houses),
    '위치등급': np.random.randint(1, 6, n_houses)
})

# 가격 계산 (비선형 관계 포함)
house_data['가격'] = (
    50000 + 
    house_data['면적'] * 1000 + 
    house_data['방수'] * 20000 - 
    house_data['나이'] * 1000 +
    house_data['위치등급'] * 30000 +
    house_data['면적']**2 * 2  # 비선형 관계
)

# 노이즈 추가
house_data['가격'] += np.random.normal(0, 20000, n_houses)

# 데이터 준비
X = house_data.drop('가격', axis=1)
y = house_data['가격']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 훈련
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)

# 예측
y_train_pred = reg_model.predict(X_train)
y_test_pred = reg_model.predict(X_test)

# 오차 패턴 분석 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 실제값 vs 예측값 (훈련 데이터)
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.5, ax=axes[0, 0])
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
axes[0, 0].set_title('훈련 데이터: 실제값 vs 예측값')
axes[0, 0].set_xlabel('실제 가격')
axes[0, 0].set_ylabel('예측 가격')

# 2. 실제값 vs 예측값 (테스트 데이터)
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5, ax=axes[0, 1])
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0, 1].set_title('테스트 데이터: 실제값 vs 예측값')
axes[0, 1].set_xlabel('실제 가격')
axes[0, 1].set_ylabel('예측 가격')

# 3. 잔차 분석
residuals = y_test - y_test_pred
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5, ax=axes[0, 2])
axes[0, 2].axhline(y=0, color='r', linestyle='--')
axes[0, 2].set_title('잔차 분석')
axes[0, 2].set_xlabel('예측 가격')
axes[0, 2].set_ylabel('잔차 (실제 - 예측)')

# 4. 잔차 분포
sns.histplot(residuals, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('잔차 분포')
axes[1, 0].set_xlabel('잔차')

# 5. 특성별 잔차 패턴
for i, feature in enumerate(['면적', '방수']):
    if i < 2:
        sns.scatterplot(x=X_test[feature], y=residuals, alpha=0.5, ax=axes[1, i+1])
        axes[1, i+1].axhline(y=0, color='r', linestyle='--')
        axes[1, i+1].set_title(f'{feature}별 잔차 패턴')
        axes[1, i+1].set_xlabel(feature)
        axes[1, i+1].set_ylabel('잔차')

plt.tight_layout()
plt.show()

# 성능 지표 계산
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("회귀 모델 성능 평가:")
print(f"훈련 RMSE: ${train_rmse:,.2f}")
print(f"테스트 RMSE: ${test_rmse:,.2f}")
print(f"훈련 R²: {train_r2:.3f}")
print(f"테스트 R²: {test_r2:.3f}")
print(f"과적합 정도: ${test_rmse - train_rmse:,.2f}")

# 잔차 패턴 분석
print("\n잔차 패턴 분석:")
print(f"잔차 평균: {residuals.mean():.2f} (0에 가까워야 함)")
print(f"잔차 표준편차: {residuals.std():.2f}")
print(f"잔차 왜도: {pd.Series(residuals).skew():.3f} (0에 가까워야 함)")

# 잔차와 예측값 간 상관관계
residual_pred_corr = np.corrcoef(residuals, y_test_pred)[0, 1]
print(f"잔차-예측값 상관계수: {residual_pred_corr:.3f} (0에 가까워야 함)")
```

## 디버깅 기법

### 1. 데이터 검증

```python
def validate_data_for_plotting(df, x_col=None, y_col=None, hue_col=None):
    """플롯팅을 위한 데이터 검증"""
    issues = []
    
    # 기본 검증
    if df.empty:
        issues.append("데이터프레임이 비어있습니다")
        return issues
    
    # 열 존재 여부 확인
    for col, name in [(x_col, 'x'), (y_col, 'y'), (hue_col, 'hue')]:
        if col and col not in df.columns:
            issues.append(f"'{col}' 열이 존재하지 않습니다")
    
    # 결측치 확인
    for col, name in [(x_col, 'x'), (y_col, 'y'), (hue_col, 'hue')]:
        if col and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            issues.append(f"'{col}' 열에 {null_count}개의 결측치가 있습니다")
    
    # 데이터 타입 확인
    for col, name in [(x_col, 'x'), (y_col, 'y')]:
        if col and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"'{col}' 열이 수치형이 아닙니다: {df[col].dtype}")
    
    # 카테고리 개수 확인
    if hue_col and df[hue_col].nunique() > 20:
        issues.append(f"'{hue_col}'의 카테고리가 너무 많습니다: {df[hue_col].nunique()}개")
    
    return issues

# 데이터 검증 예시
test_data = pd.DataFrame({
    'x': [1, 2, np.nan, 4],
    'y': ['a', 'b', 'c', 'd'],  # 문자열
    'hue': [1, 2, 3, 4]
})

issues = validate_data_for_plotting(test_data, 'x', 'y', 'hue')
print("데이터 검증 결과:")
for issue in issues:
    print(f"- {issue}")
```

### 2. 플롯 디버깅

```python
def debug_plot_creation(plot_func, *args, **kwargs):
    """플롯 생성 과정 디버깅"""
    try:
        print("플롯 생성 시작...")
        
        # 데이터 검증
        if 'data' in kwargs:
            data = kwargs['data']
            print(f"데이터 크기: {data.shape}")
            print(f"데이터 타입:\n{data.dtypes}")
            print(f"결측치:\n{data.isnull().sum()}")
        
        # 플롯 생성
        fig = plot_func(*args, **kwargs)
        print("플롯 생성 성공")
        return fig
        
    except Exception as e:
        print(f"플롯 생성 오류: {e}")
        
        # 추가 정보 출력
        print("\n디버깅 정보:")
        if 'data' in kwargs:
            data = kwargs['data']
            print(f"데이터 샘플:\n{data.head()}")
            
            # 수치형 열 확인
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            print(f"수치형 열: {list(numeric_cols)}")
            
            # 범주형 열 확인
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            print(f"범주형 열: {list(categorical_cols)}")
        
        return None

# 디버깅 예시
problematic_data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': ['a', 'b', 'c', 'd', 'e']  # 문자열 y값
})

fig = debug_plot_creation(
    sns.scatterplot,
    data=problematic_data,
    x='x',
    y='y'
)

if fig:
    plt.show()
```

### 3. 단계적 문제 해결

```python
def step_by_step_troubleshooting(data, plot_type='scatter', x_col=None, y_col=None, hue_col=None):
    """단계적 문제 해결"""
    
    steps = [
        "1. 데이터 기본 검증",
        "2. 데이터 전처리",
        "3. 간단한 플롯 시도",
        "4. 복잡한 플롯 시도",
        "5. 최종 플롯"
    ]
    
    current_data = data.copy()
    
    for i, step in enumerate(steps):
        print(f"\n{step}")
        print("=" * len(step))
        
        try:
            if i == 0:  # 데이터 기본 검증
                issues = validate_data_for_plotting(current_data, x_col, y_col, hue_col)
                if issues:
                    print("문제 발견:")
                    for issue in issues:
                        print(f"- {issue}")
                    continue
                print("데이터 검증 통과")
                
            elif i == 1:  # 데이터 전처리
                # 결측치 처리
                for col in [x_col, y_col, hue_col]:
                    if col and current_data[col].isnull().any():
                        if col == hue_col:
                            current_data = current_data.dropna(subset=[col])
                        else:
                            current_data[col] = current_data[col].fillna(current_data[col].median())
                
                # 타입 변환
                for col in [x_col, y_col]:
                    if col and not pd.api.types.is_numeric_dtype(current_data[col]):
                        try:
                            current_data[col] = pd.to_numeric(current_data[col])
                        except:
                            print(f"{col} 열을 수치형으로 변환할 수 없습니다")
                            continue
                
                print("데이터 전처리 완료")
                
            elif i == 2:  # 간단한 플롯
                if plot_type == 'scatter':
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=current_data, x=x_col, y=y_col)
                    plt.title("간단한 산점도")
                    plt.show()
                    
            elif i == 3:  # 복잡한 플롯
                if plot_type == 'scatter' and hue_col:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=current_data, x=x_col, y=y_col, hue=hue_col)
                    plt.title("색상 구분 산점도")
                    plt.show()
                    
            elif i == 4:  # 최종 플롯
                plt.figure(figsize=(10, 8))
                if plot_type == 'scatter':
                    sns.scatterplot(data=current_data, x=x_col, y=y_col, hue=hue_col, 
                                   size=current_data[x_col] if x_col else None)
                    plt.title("최종 플롯")
                    plt.show()
                
                print("문제 해결 완료!")
                return True
                
        except Exception as e:
            print(f"단계 {i+1}에서 오류 발생: {e}")
            continue
    
    print("문제 해결 실패")
    return False

# 단계적 문제 해결 예시
problem_data = pd.DataFrame({
    'x': [1, 2, np.nan, 4, 5],
    'y': ['2', '3', '4', '5', '6'],  # 문자열
    'hue': ['A', 'B', 'A', 'B', 'A']
})

success = step_by_step_troubleshooting(
    problem_data,
    plot_type='scatter',
    x_col='x',
    y_col='y',
    hue_col='hue'
)
```

## 문제 해결 체크리스트

### 일반적인 문제 해결 체크리스트

#### 데이터 관련
- [ ] 데이터프레임이 비어있지 않은가?
- [ ] 필요한 열이 모두 존재하는가?
- [ ] 결측치가 적절히 처리되었는가?
- [ ] 데이터 타입이 올바른가?
- [ ] 카테고리 개수가 적절한가?

#### 플롯 관련
- [ ] 플롯 크기가 적절한가?
- [ ] 레이아웃이 잘리지 않았는가?
- [ ] 범례와 제목이 잘 보이는가?
- [ ] 색상 팔레트가 적절한가?
- [ ] 축 범위가 적절한가?

#### 성능 관련
- [ ] 데이터 크기가 너무 크지 않은가?
- [ ] 적절한 샘플링을 사용했는가?
- [ ] 효율적인 플롯 유형을 선택했는가?
- [ ] 불필요한 플롯 요소를 제거했는가?

#### 통계 관련
- [ ] 이상치가 통계를 왜곡하지 않는가?
- [ ] 적절한 통계 방법을 사용했는가?
- [ ] 신뢰 구간이 올바르게 표시되는가?

### 머신러닝 프로젝트 체크리스트

#### 머신러닝 데이터 준비 단계
- [ ] 데이터 품질 진단 완료 (결측치, 이상치, 타입 불일치)
- [ ] 피처 스케일링 필요성 확인 및 적용
- [ ] 클래스 불균형 확인 및 처리 방안 결정
- [ ] 피처 간 상관관계 분석 및 다중공선성 확인

#### 머신러닝 모델링 단계
- [ ] 적절한 평가 지표 선택 (분류/회귀 문제에 따라)
- [ ] 교차 검증을 통한 안정적인 성능 평가
- [ ] 하이퍼파라미터 튜닝 결과 시각화
- [ ] 다양한 모델 성능 비교 및 최적 모델 선택

#### 머신러닝 모델 평가 단계
- [ ] 혼동 행렬을 통한 분류 성능 상세 분석
- [ ] ROC/PR 곡선을 통한 임계값 민감도 분석
- [ ] 잔차 분석을 통한 회귀 모델 가정 검증
- [ ] 특성 중요도 시각화를 통한 모델 해석

#### 시각화 성능 최적화
- [ ] 대용량 데이터 적절한 샘플링 또는 집계
- [ ] 효율적인 플롯 유형 선택 (hexbin, 2D 밀도 등)
- [ ] 고차원 데이터 차원 축소 기법 활용
- [ ] 메모리 사용량 최적화

## 결론

Seaborn은 머신러닝 프로젝트에서 단순한 시각화 도구를 넘어 다음과 같은 역할을 수행합니다:

1. **데이터 탐색**: 데이터의 품질 문제를 시각적으로 발견
2. **피처 분석**: 피처 간 관계와 분포를 통해 모델링 방향 결정
3. **모델 평가**: 다각적인 성능 지표와 오차 패턴 분석
4. **결과 전달**: 비전문가에게 모델 결과를 직관적으로 전달

이러한 문제 해결 접근 방식을 통해 머신러닝 프로젝트의 각 단계에서 발생하는 문제들을 효과적으로 진단하고 해결할 수 있습니다.

## 다음 단계

문제 해결 방법을 익혔다면, [실제 사용 사례](11-examples.md) 문서에서 복잡한 비즈니스 문제를 Seaborn으로 해결하는 방법을 학습해보세요.

## 추가 자료

- [Seaborn 공통 문제 해결](https://seaborn.pydata.org/tutorial.html)
- [matplotlib 문제 해결](https://matplotlib.org/stable/users/troubleshooting.html)
- [pandas 데이터 정제 가이드](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Python 데이터 시각화 디버깅](https://realpython.com/python-data-visualization-troubleshooting/)
- [머신러닝 모델 평가 가이드](https://scikit-learn.org/stable/model_evaluation.html)
- [머신러닝 문제 해결 프레임워크](ml-problem-solution-framework.md)