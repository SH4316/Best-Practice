# 머신러닝 문제 해결을 위한 Seaborn 기본 플롯

이 문서는 머신러닝 프로젝트에서 발생하는 실제 문제들을 Seaborn 기본 플롯을 통해 해결하는 방법을 다룹니다. 각 플롯 유형을 특정 머신러닝 문제 상황과 연결하여 설명합니다.

## 관계형 플롯 (Relational Plots) - 피처 관계 탐색

관계형 플롯은 머신러닝 모델링 전 피처 간의 관계를 이해하고, 클래스 분리 패턴을 발견하는 데 사용됩니다.

### 문제: 분류 모델링 전 클래스 분리 패턴 파악

**상황**: 이진 분류 문제에서 어떤 피처들이 클래스를 잘 분리하는지 파악해야 함

**해결책**: 산점도를 사용하여 피처 간 관계와 클래스 분리 패턴 시각화

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# 문제 상황: 암 진단 데이터에서 양성/악성을 구분하는 피처 패턴 파악
# 데이터 생성
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          class_sep=1.5, random_state=42)

# 데이터프레임 생성
feature_names = ['종양 크기', '세포 밀도']
df = pd.DataFrame(X, columns=feature_names)
df['진단 결과'] = np.where(y == 0, '양성', '악성')

# Seaborn으로 클래스 분리 패턴 시각화
plt.figure(figsize=(12, 5))

# 기본 산점도
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='종양 크기', y='세포 밀도', hue='진단 결과', 
                palette=['blue', 'red'], alpha=0.7)
plt.title('암 진단: 종양 크기 vs 세포 밀도')
plt.legend(title='진단 결과')

# 밀도 추정이 추가된 산점도
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='종양 크기', y='세포 밀도', hue='진단 결과', 
                palette=['blue', 'red'], alpha=0.3)
sns.kdeplot(data=df, x='종양 크기', y='세포 밀도', hue='진단 결과', 
            palette=['blue', 'red'], alpha=0.3)
plt.title('밀도 추정이 추가된 클래스 분리 패턴')
plt.legend(title='진단 결과')

plt.tight_layout()
plt.show()

# 인사이트 도출
print("클래스 분리 패턴 분석:")
print(f"- 양성 종양 평균 크기: {df[df['진단 결과'] == '양성']['종양 크기'].mean():.2f}")
print(f"- 악성 종양 평균 크기: {df[df['진단 결과'] == '악성']['종양 크기'].mean():.2f}")
print(f"- 양성 종양 평균 밀도: {df[df['진단 결과'] == '양성']['세포 밀도'].mean():.2f}")
print(f"- 악성 종양 평균 밀도: {df[df['진단 결과'] == '악성']['세포 밀도'].mean():.2f}")
```

### 문제: 다중 피처 간의 복잡한 관계 파악

**상황**: 여러 피처가 복합적으로 작용하는 패턴을 파악해야 함

**해결책**: 다차원 산점도로 여러 변수를 동시에 시각화

```python
# 문제 상황: 고객 이탈 예측에서 다양한 고객 특성과 이탈 관계 파악
# 데이터 생성
np.random.seed(42)
n_customers = 300

customer_data = pd.DataFrame({
    '서비스 기간(월)': np.random.exponential(scale=24, size=n_customers),
    '월 요금($)': np.random.normal(65, 20, n_customers),
    '고객 만족도': np.random.normal(3.5, 1, n_customers),
    '이탈 여부': np.random.choice(['이탈', '유지'], n_customers, p=[0.3, 0.7])
})

# 이탈 고객 특성 추가
customer_data.loc[customer_data['이탈 여부'] == '이탈', '서비스 기간(월)'] *= 0.5
customer_data.loc[customer_data['이탈 여부'] == '이탈', '월 요금($)'] *= 1.2
customer_data.loc[customer_data['이탈 여부'] == '이탈', '고객 만족도'] *= 0.7

# 값 제한
customer_data['서비스 기간(월)'] = customer_data['서비스 기간(월)'].clip(1, 72)
customer_data['월 요금($)'] = customer_data['월 요금($)'].clip(20, 150)
customer_data['고객 만족도'] = customer_data['고객 만족도'].clip(1, 5)

# 다차원 산점도 시각화
plt.figure(figsize=(15, 5))

# 1. 기본 2차원 산점도
plt.subplot(1, 3, 1)
sns.scatterplot(data=customer_data, x='서비스 기간(월)', y='월 요금($)', 
                hue='이탈 여부', palette=['green', 'red'], alpha=0.7)
plt.title('서비스 기간 vs 월 요금')

# 2. 크기와 스타일을 추가한 3차원 시각화
plt.subplot(1, 3, 2)
sns.scatterplot(data=customer_data, x='서비스 기간(월)', y='월 요금($)', 
                hue='이탈 여부', size='고객 만족도', 
                style='이탈 여부', palette=['green', 'red'], 
                sizes=(20, 200), alpha=0.7)
plt.title('서비스 기간 vs 월 요금 (크기=만족도)')

# 3. relplot을 사용한 다중 플롯
g = sns.relplot(data=customer_data, x='서비스 기간(월)', y='월 요금($)', 
                col='이탈 여부', hue='고객 만족도', size='고객 만족도',
                palette='viridis', sizes=(20, 200), alpha=0.7, height=5)
g.fig.suptitle('이탈 여부별 고객 특성 분포', y=1.02)

plt.tight_layout()
plt.show()

# 인사이트 도출
churned = customer_data[customer_data['이탈 여부'] == '이탈']
retained = customer_data[customer_data['이탈 여부'] == '유지']

print("고객 이탈 패턴 분석:")
print(f"- 이탈 고객 평균 서비스 기간: {churned['서비스 기간(월)'].mean():.1f}개월")
print(f"- 유지 고객 평균 서비스 기간: {retained['서비스 기간(월)'].mean():.1f}개월")
print(f"- 이탈 고객 평균 월 요금: ${churned['월 요금($)'].mean():.2f}")
print(f"- 유지 고객 평균 월 요금: ${retained['월 요금($)'].mean():.2f}")
```

### 문제: 시계열 데이터의 추세와 패턴 파악

**상황**: 시간에 따른 모델 성능 변화나 데이터 추세를 분석해야 함

**해결책**: 선 그래프로 시간에 따른 변화 패턴 시각화

```python
# 문제 상황: 머신러닝 모델의 성능이 시간에 따라 어떻게 변화하는지 모니터링
# 시계열 데이터 생성
dates = pd.date_range(start='2023-01-01', periods=90)
performance_data = pd.DataFrame({
    '날짜': dates,
    '정확도': 0.85 + 0.05 * np.sin(np.arange(90) * 2 * np.pi / 30) + np.random.normal(0, 0.02, 90),
    '정밀도': 0.82 + 0.03 * np.sin(np.arange(90) * 2 * np.pi / 30 + np.pi/4) + np.random.normal(0, 0.015, 90),
    '재현율': 0.88 + 0.04 * np.sin(np.arange(90) * 2 * np.pi / 30 + np.pi/2) + np.random.normal(0, 0.025, 90)
})

# 모델 업데이트 시점 표시
update_dates = ['2023-01-15', '2023-02-15', '2023-03-15']
for date in update_dates:
    idx = performance_data[performance_data['날짜'] == date].index[0]
    performance_data.loc[idx:, '정확도'] += 0.02  # 업데이트 후 성능 향상

# 시계열 성능 모니터링 시각화
plt.figure(figsize=(15, 10))

# 1. 기본 선 그래프
plt.subplot(2, 2, 1)
sns.lineplot(data=performance_data, x='날짜', y='정확도')
plt.title('모델 정확도 시계열')
plt.ylabel('정확도')
plt.grid(True, alpha=0.3)

# 2. 신뢰 구간이 포함된 선 그래프
plt.subplot(2, 2, 2)
sns.lineplot(data=performance_data, x='날짜', y='정확도', ci='sd')
plt.title('신뢰 구간이 포함된 정확도')
plt.ylabel('정확도')

# 3. 여러 지표 비교
plt.subplot(2, 2, 3)
performance_melted = pd.melt(performance_data, id_vars=['날짜'], 
                            value_vars=['정확도', '정밀도', '재현율'],
                            var_name='지표', value_name='값')
sns.lineplot(data=performance_melted, x='날짜', y='값', hue='지표', style='지표', markers=True)
plt.title('다양한 성능 지표 비교')
plt.ylabel('값')
plt.legend(title='성능 지표')

# 4. 모델 업데이트 시점 표시
plt.subplot(2, 2, 4)
sns.lineplot(data=performance_data, x='날짜', y='정확도', marker='o')
for date in update_dates:
    plt.axvline(pd.to_datetime(date), color='red', linestyle='--', alpha=0.7)
plt.title('모델 업데이트 시점과 성능 변화')
plt.ylabel('정확도')

plt.tight_layout()
plt.show()

# 인사이트 도출
print("모델 성능 분석:")
print(f"평균 정확도: {performance_data['정확도'].mean():.3f}")
print(f"최고 정확도: {performance_data['정확도'].max():.3f}")
print(f"최저 정확도: {performance_data['정확도'].min():.3f}")
print(f"성능 표준편차: {performance_data['정확도'].std():.4f}")
```

## 범주형 플롯 (Categorical Plots) - 범주형 피처 분석

범주형 플롯은 범주형 피처의 분포를 비교하고, 클래스별 특성 차이를 파악하는 데 사용됩니다.

### 문제: 클래스별 피처 분포 비교

**상황**: 분류 문제에서 각 클래스에 따른 수치형 피처의 분포 차이를 파악해야 함

**해결책**: 박스 플롯과 바이올린 플롯으로 클래스별 분포 비교

```python
# 문제 상황: 붓꽃 품종별 꽃잎/꽃받침 특성 차이 파악
# 데이터 로드
iris = sns.load_dataset("iris")

# 범주형 피처 분석 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 박스 플롯으로 품종별 꽃받침 길이 비교
sns.boxplot(data=iris, x='species', y='sepal_length', ax=axes[0, 0])
axes[0, 0].set_title('품종별 꽃받침 길이 분포')
axes[0, 0].set_ylabel('꽃받침 길이 (cm)')

# 2. 바이올린 플롯으로 품종별 꽃잎 길이 분포
sns.violinplot(data=iris, x='species', y='petal_length', ax=axes[0, 1])
axes[0, 1].set_title('품종별 꽃잎 길이 분포')
axes[0, 1].set_ylabel('꽃잎 길이 (cm)')

# 3. 박스 플롯으로 품종별 꽃받침 너비 비교
sns.boxplot(data=iris, x='species', y='sepal_width', ax=axes[1, 0])
axes[1, 0].set_title('품종별 꽃받침 너비 분포')
axes[1, 0].set_ylabel('꽃받침 너비 (cm)')

# 4. 바이올린 플롯으로 품종별 꽃잎 너비 분포
sns.violinplot(data=iris, x='species', y='petal_width', ax=axes[1, 1])
axes[1, 1].set_title('품종별 꽃잎 너비 분포')
axes[1, 1].set_ylabel('꽃잎 너비 (cm)')

plt.tight_layout()
plt.show()

# 인사이트 도출
print("품종별 특성 분석:")
for species in iris['species'].unique():
    species_data = iris[iris['species'] == species]
    print(f"\n{species}:")
    print(f"  - 평균 꽃받침 길이: {species_data['sepal_length'].mean():.2f}cm")
    print(f"  - 평균 꽃잎 길이: {species_data['petal_length'].mean():.2f}cm")
    print(f"  - 평균 꽃받침 너비: {species_data['sepal_width'].mean():.2f}cm")
    print(f"  - 평균 꽃잎 너비: {species_data['petal_width'].mean():.2f}cm")
```

### 문제: 다중 범주형 변수의 교효과 분석

**상황**: 두 개 이상의 범주형 변수가 결합될 때의 효과를 파악해야 함

**해결책**: 분할 바이올린 플롯과 다중 범주형 플롯으로 교효과 시각화

```python
# 문제 상황: 성별과 흡연 여부에 따른 의료 비용 차이 분석
# 데이터 생성
np.random.seed(42)
n_patients = 200

medical_data = pd.DataFrame({
    '성별': np.random.choice(['남성', '여성'], n_patients),
    '흡연 여부': np.random.choice(['흡연자', '비흡연자'], n_patients, p=[0.3, 0.7]),
    '나이': np.random.normal(45, 15, n_patients).astype(int).clip(18, 80)
})

# 의료 비용 계산 (성별, 흡연 여부, 나이에 따른 차이)
base_cost = 5000
medical_data['의료 비용($)'] = base_cost

# 성별에 따른 차이
medical_data.loc[medical_data['성별'] == '남성', '의료 비용($)'] *= 1.1

# 흡연 여부에 따른 차이
medical_data.loc[medical_data['흡연 여부'] == '흡연자', '의료 비용($)'] *= 2.0

# 나이에 따른 차이
medical_data['의료 비용($)'] += medical_data['나이'] * 50

# 노이즈 추가
medical_data['의료 비용($)'] += np.random.normal(0, 1000, n_patients)

# 다중 범주형 변수 분석 시각화
plt.figure(figsize=(15, 10))

# 1. 성별별 의료 비용 분포
plt.subplot(2, 2, 1)
sns.boxplot(data=medical_data, x='성별', y='의료 비용($)')
plt.title('성별별 의료 비용 분포')

# 2. 흡연 여부별 의료 비용 분포
plt.subplot(2, 2, 2)
sns.boxplot(data=medical_data, x='흡연 여부', y='의료 비용($)')
plt.title('흡연 여부별 의료 비용 분포')

# 3. 성별과 흡연 여부의 교효과
plt.subplot(2, 2, 3)
sns.boxplot(data=medical_data, x='성별', y='의료 비용($)', hue='흡연 여부')
plt.title('성별과 흡연 여부에 따른 의료 비용')

# 4. 분할 바이올린 플롯으로 교효과 시각화
plt.subplot(2, 2, 4)
sns.violinplot(data=medical_data, x='성별', y='의료 비용($)', hue='흡연 여부', split=True)
plt.title('성별과 흡연 여부에 따른 의료 비용 (바이올린)')

plt.tight_layout()
plt.show()

# 인사이트 도출
print("의료 비용 분석:")
for gender in medical_data['성별'].unique():
    for smoker in medical_data['흡연 여부'].unique():
        subset = medical_data[(medical_data['성별'] == gender) & 
                            (medical_data['흡연 여부'] == smoker)]
        avg_cost = subset['의료 비용($)'].mean()
        print(f"{gender} {smoker}: 평균 ${avg_cost:,.0f}")
```

### 문제: 범주별 통계적 요약 비교

**상황**: 범주형 그룹별 평균, 중앙값 등 통계적 요약을 비교해야 함

**해결책**: 막대 그래프와 포인트 플롯으로 범주별 통계 비교

```python
# 문제 상황: 다양한 머신러닝 알고리즘의 성능 비교
# 데이터 생성
algorithms = ['로지스틱 회귀', '의사결정나무', '랜덤 포레스트', 'SVM', '신경망']
datasets = ['데이터셋A', '데이터셋B', '데이터셋C']

results = []
for algo in algorithms:
    for dataset in datasets:
        # 알고리즘과 데이터셋에 따른 성능 차이
        base_acc = {'로지스틱 회귀': 0.80, '의사결정나무': 0.75, '랜덤 포레스트': 0.85, 
                   'SVM': 0.82, '신경망': 0.88}[algo]
        dataset_factor = {'데이터셋A': 1.0, '데이터셋B': 0.95, '데이터셋C': 1.05}[dataset]
        
        accuracy = base_acc * dataset_factor + np.random.normal(0, 0.02)
        results.append({
            '알고리즘': algo,
            '데이터셋': dataset,
            '정확도': accuracy
        })

results_df = pd.DataFrame(results)

# 범주별 통계 비교 시각화
plt.figure(figsize=(15, 10))

# 1. 알고리즘별 평균 정확도
plt.subplot(2, 2, 1)
algo_accuracy = results_df.groupby('알고리즘')['정확도'].mean().reset_index()
sns.barplot(data=algo_accuracy, x='알고리즘', y='정확도', palette='viridis')
plt.title('알고리즘별 평균 정확도')
plt.xticks(rotation=45)

# 2. 데이터셋별 평균 정확도
plt.subplot(2, 2, 2)
dataset_accuracy = results_df.groupby('데이터셋')['정확도'].mean().reset_index()
sns.barplot(data=dataset_accuracy, x='데이터셋', y='정확도', palette='plasma')
plt.title('데이터셋별 평균 정확도')

# 3. 알고리즘과 데이터셋의 교효과
plt.subplot(2, 2, 3)
sns.barplot(data=results_df, x='알고리즘', y='정확도', hue='데이터셋')
plt.title('알고리즘과 데이터셋에 따른 정확도')
plt.xticks(rotation=45)

# 4. 포인트 플롯으로 정확도 추세 비교
plt.subplot(2, 2, 4)
sns.pointplot(data=results_df, x='데이터셋', y='정확도', hue='알고리즘', 
              markers=['o', 's', 'D', '^', 'v'], linestyles=['-', '--', '-.', ':', (0, (3, 1, 1, 1))])
plt.title('데이터셋별 알고리즘 성능 추세')
plt.ylim(0.7, 1.0)

plt.tight_layout()
plt.show()

# 인사이트 도출
print("알고리즘 성능 분석:")
for algo in algorithms:
    algo_data = results_df[results_df['알고리즘'] == algo]
    avg_acc = algo_data['정확도'].mean()
    std_acc = algo_data['정확도'].std()
    print(f"{algo}: 평균 정확도 {avg_acc:.3f} ± {std_acc:.3f}")
```

## 분포 플롯 (Distribution Plots) - 데이터 분포 이해

분포 플롯은 피처의 분포 형태를 파악하고, 이상치를 발견하며, 정규성을 검증하는 데 사용됩니다.

### 문제: 피처 분포의 왜도와 첨도 파악

**상황**: 머신러닝 모델의 가정(예: 정규분포)을 만족하는지 확인해야 함

**해결책**: 히스토그램과 밀도 플롯으로 분포 형태 분석

```python
# 문제 상황: 회귀 모델링 전 타겟 변수와 피처의 분포 확인
# 데이터 생성
np.random.seed(42)
n_samples = 1000

# 다양한 분포를 가진 피처 생성
data = {
    '정규분포 피처': np.random.normal(50, 10, n_samples),
    '왜곡분포 피처': np.random.exponential(scale=20, size=n_samples),
    '이봉분포 피처': np.concatenate([
        np.random.normal(30, 5, n_samples//2),
        np.random.normal(70, 5, n_samples//2)
    ]),
    '균등분포 피처': np.random.uniform(0, 100, n_samples)
}

df_dist = pd.DataFrame(data)

# 분포 분석 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 정규분포 피처
sns.histplot(data=df_dist, x='정규분포 피처', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('정규분포 피처')
axes[0, 0].set_xlabel('값')

# 2. 왜곡분포 피처
sns.histplot(data=df_dist, x='왜곡분포 피처', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('왜곡분포 피처')
axes[0, 1].set_xlabel('값')

# 3. 이봉분포 피처
sns.histplot(data=df_dist, x='이봉분포 피처', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('이봉분포 피처')
axes[1, 0].set_xlabel('값')

# 4. 균등분포 피처
sns.histplot(data=df_dist, x='균등분포 피처', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('균등분포 피처')
axes[1, 1].set_xlabel('값')

plt.tight_layout()
plt.show()

# 분포 통계 계산
print("분포 통계 분석:")
for feature in df_dist.columns:
    data = df_dist[feature]
    skewness = data.skew()
    kurtosis = data.kurtosis()
    print(f"\n{feature}:")
    print(f"  - 왜도 (Skewness): {skewness:.3f}")
    print(f"  - 첨도 (Kurtosis): {kurtosis:.3f}")
    
    if abs(skewness) < 0.5:
        print("  - 분포 형태: 거의 대칭")
    elif skewness > 0.5:
        print("  - 분포 형태: 오른쪽으로 왜곡")
    else:
        print("  - 분포 형태: 왼쪽으로 왜곡")
```

### 문제: 클래스별 분포 차이 분석

**상황**: 분류 문제에서 각 클래스에 따른 피처 분포의 차이를 파악해야 함

**해결책**: 클래스별 밀도 플롯과 누적 분포 함수로 분포 차이 비교

```python
# 문제 상황: 스팸 메일 분류에서 스팸/정상 메일의 특성 분포 차이 파악
# 데이터 생성
np.random.seed(42)
n_emails = 500

# 정상 메일 특성
normal_emails = pd.DataFrame({
    '단어 수': np.random.normal(100, 30, n_emails//2),
    '특수문자 비율': np.random.normal(0.05, 0.02, n_emails//2),
    '대문자 비율': np.random.normal(0.1, 0.05, n_emails//2),
    '메일 유형': ['정상'] * (n_emails//2)
})

# 스팸 메일 특성
spam_emails = pd.DataFrame({
    '단어 수': np.random.normal(200, 50, n_emails//2),
    '특수문자 비율': np.random.normal(0.15, 0.05, n_emails//2),
    '대문자 비율': np.random.normal(0.3, 0.1, n_emails//2),
    '메일 유형': ['스팸'] * (n_emails//2)
})

# 데이터 결합
email_data = pd.concat([normal_emails, spam_emails], ignore_index=True)

# 값 제한
email_data['단어 수'] = email_data['단어 수'].clip(10, 400)
email_data['특수문자 비율'] = email_data['특수문자 비율'].clip(0, 0.5)
email_data['대문자 비율'] = email_data['대문자 비율'].clip(0, 1)

# 클래스별 분포 비교 시각화
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

features = ['단어 수', '특수문자 비율', '대문자 비율']
for i, feature in enumerate(features):
    # 히스토그램으로 분포 비교
    sns.histplot(data=email_data, x=feature, hue='메일 유형', 
                kde=True, alpha=0.5, ax=axes[i, 0])
    axes[i, 0].set_title(f'{feature} 분포 (히스토그램)')
    
    # 밀도 플롯으로 분포 비교
    sns.kdeplot(data=email_data, x=feature, hue='메일 유형', 
                fill=True, alpha=0.5, ax=axes[i, 1])
    axes[i, 1].set_title(f'{feature} 분포 (밀도 플롯)')

plt.tight_layout()
plt.show()

# 누적 분포 함수(ECDF)로 분포 차이 분석
plt.figure(figsize=(15, 5))

for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.ecdfplot(data=email_data, x=feature, hue='메일 유형')
    plt.title(f'{feature} 누적 분포')

plt.tight_layout()
plt.show()

# 인사이트 도출
print("메일 유형별 특성 분석:")
for email_type in ['정상', '스팸']:
    subset = email_data[email_data['메일 유형'] == email_type]
    print(f"\n{email_type} 메일:")
    for feature in features:
        mean_val = subset[feature].mean()
        std_val = subset[feature].std()
        print(f"  - 평균 {feature}: {mean_val:.2f} ± {std_val:.2f}")
```

### 문제: 다변량 분포와 상관관계 파악

**상황**: 여러 피처 간의 상관관계와 다변량 분포를 파악해야 함

**해결책**: 2D 밀도 플롯과 조인트 플롯으로 다변량 관계 시각화

```python
# 문제 상황: 주택 가격 예측에서 주요 피처 간의 상관관계 파악
# 데이터 생성
np.random.seed(42)
n_houses = 300

housing_data = pd.DataFrame({
    '면적(m²)': np.random.normal(120, 30, n_houses),
    '방 수': np.random.poisson(3, n_houses) + 1,
    '나이(년)': np.random.exponential(scale=20, size=n_houses),
    '가격($)': 0  # 계산 필요
})

# 가격 계산 (다른 피처와의 관계 기반)
housing_data['가격($)'] = (
    50000 +  # 기본 가격
    housing_data['면적(m²)'] * 1000 +  # 면적에 따른 가격
    housing_data['방 수'] * 20000 -  # 방 수에 따른 가격
    housing_data['나이(년)'] * 1000  # 나이에 따른 가격 감소
)

# 노이즈 추가
housing_data['가격($)'] += np.random.normal(0, 20000, n_houses)

# 값 제한
housing_data['면적(m²)'] = housing_data['면적(m²)'].clip(50, 250)
housing_data['나이(년)'] = housing_data['나이(년)'].clip(0, 50)
housing_data['가격($)'] = housing_data['가격($)'].clip(50000, 500000)

# 다변량 관계 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 면적과 가격 관계 (2D 밀도 플롯)
sns.kdeplot(data=housing_data, x='면적(m²)', y='가격($)', 
            fill=True, cmap='viridis', ax=axes[0, 0])
axes[0, 0].set_title('면적과 가격의 2D 밀도')

# 2. 방 수와 가격 관계
sns.kdeplot(data=housing_data, x='방 수', y='가격($)', 
            fill=True, cmap='plasma', ax=axes[0, 1])
axes[0, 1].set_title('방 수와 가격의 2D 밀도')

# 3. 나이와 가격 관계
sns.kdeplot(data=housing_data, x='나이(년)', y='가격($)', 
            fill=True, cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('나이와 가격의 2D 밀도')

# 4. 상관관계 히트맵
corr_matrix = housing_data[['면적(m²)', '방 수', '나이(년)', '가격($)']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('피처 간 상관관계')

plt.tight_layout()
plt.show()

# 조인트 플롯으로 상세한 관계 분석
sns.jointplot(data=housing_data, x='면적(m²)', y='가격($)', 
              kind='reg', height=8, space=0.2)
plt.suptitle('면적과 가격 관계 (조인트 플롯)', y=1.02)
plt.show()

# 인사이트 도출
print("주택 가격 상관관계 분석:")
print(f"면적-가격 상관계수: {housing_data['면적(m²)'].corr(housing_data['가격($)']):.3f}")
print(f"방 수-가격 상관계수: {housing_data['방 수'].corr(housing_data['가격($)']):.3f}")
print(f"나이-가격 상관계수: {housing_data['나이(년)'].corr(housing_data['가격($)']):.3f}")
```

## 결론

머신러닝 프로젝트에서 Seaborn의 기본 플롯들은 다음과 같은 문제 해결에 활용될 수 있습니다:

1. **관계형 플롯**: 피처 간 관계와 클래스 분리 패턴 파악
2. **범주형 플롯**: 범주형 피처의 분포 비교와 교효과 분석
3. **분포 플롯**: 데이터 분포 형태 파악과 정규성 검증

이러한 시각화 도구들을 통해 머신러닝 모델링의 각 단계에서 데이터를 더 깊이 이해하고 더 나은 모델을 구축할 수 있습니다.

## 다음 단계

기본 플롯의 문제 해결 활용법을 익혔다면, [고급 플롯 유형](04-advanced-plots.md) 문서에서 더 복잡한 시각화 기법을 학습해보세요.

## 추가 자료

- [Seaborn 관계형 플롯 문서](https://seaborn.pydata.org/tutorial/relational.html)
- [Seaborn 범주형 플롯 문서](https://seaborn.pydata.org/tutorial/categorical.html)
- [Seaborn 분포 플롯 문서](https://seaborn.pydata.org/tutorial/distributions.html)
- [머신러닝 문제 해결 프레임워크](ml-problem-solution-framework.md)