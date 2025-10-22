# 머신러닝 실전 문제 해결 예시

이 문서는 실제 머신러닝 프로젝트에서 발생하는 구체적인 문제들을 Seaborn을 통해 해결하는 실용적인 예시들을 다룹니다. 각 예시는 문제 정의부터 해결 과정, 결과 해석까지 포함합니다.

## 예시 1: 이상치 탐지 및 처리

### 문제 상황
제조 공정의 센서 데이터를 분석하여 이상치를 탐지하고 처리해야 함

### 해결 과정

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 문제: 제조 공정 센서 데이터에서 이상치 탐지
# 데이터 생성 (정상 데이터와 이상치 포함)
np.random.seed(42)
n_samples = 500

# 정상 데이터 생성
normal_data = pd.DataFrame({
    '온도': np.random.normal(25, 2, n_samples),
    '압력': np.random.normal(100, 10, n_samples),
    '진동': np.random.normal(0.5, 0.1, n_samples),
    '전류': np.random.normal(5, 0.5, n_samples)
})

# 이상치 추가 (5%)
n_outliers = int(n_samples * 0.05)
outlier_indices = np.random.choice(normal_data.index, n_outliers, replace=False)

# 다양한 유형의 이상치 추가
normal_data.loc[outlier_indices[:n_outliers//4], '온도'] *= 2  # 극단값
normal_data.loc[outlier_indices[n_outliers//4:n_outliers//2], '압력'] += 50  # 편향
normal_data.loc[outlier_indices[n_outliers//2:3*n_outliers//4], '진동'] = 2  # 고정값
normal_data.loc[outlier_indices[3*n_outliers//4:], '전류'] = np.random.uniform(0, 10, n_outliers//4)  # 무작위

# 이상치 탐지 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 각 센서의 분포 (이상치 확인)
for i, sensor in enumerate(['온도', '압력', '진동']):
    sns.boxplot(data=normal_data, y=sensor, ax=axes[0, i])
    axes[0, i].set_title(f'{sensor} 분포 (이상치 확인)')

# 2. 센서 간 관계 (이상치 패턴)
sns.scatterplot(data=normal_data, x='온도', y='압력', ax=axes[1, 0])
axes[1, 0].set_title('온도 vs 압력')

sns.scatterplot(data=normal_data, x='진동', y='전류', ax=axes[1, 1])
axes[1, 1].set_title('진동 vs 전류')

# 3. 상관관계 히트맵
corr_matrix = normal_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
axes[1, 2].set_title('센서 간 상관관계')

plt.tight_layout()
plt.show()

# Isolation Forest를 이용한 이상치 탐지
scaler = StandardScaler()
scaled_data = scaler.fit_transform(normal_data)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(scaled_data)

normal_data['이상치_여부'] = np.where(outlier_labels == -1, '이상치', '정상')

# 이상치 탐지 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 이상치 분포
sns.countplot(data=normal_data, x='이상치_여부', ax=axes[0, 0])
axes[0, 0].set_title('이상치/정상 데이터 분포')

# 2. 이상치 강조 산점도
palette = {'정상': 'blue', '이상치': 'red'}
sns.scatterplot(data=normal_data, x='온도', y='압력', 
                hue='이상치_여부', palette=palette, s=50, alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title('온도 vs 압력 (이상치 강조)')

# 3. 이상치와 정상 데이터의 통계적 특성 비교
sensor_stats = normal_data.groupby('이상치_여부')[['온도', '압력', '진동', '전류']].mean()
sns.heatmap(sensor_stats.T, annot=True, fmt='.2f', cmap='viridis', ax=axes[1, 0])
axes[1, 0].set_title('이상치/정상 데이터 평균값 비교')

# 4. 시계열 패턴 (가상 시간 정보 추가)
normal_data['시간'] = range(len(normal_data))
sns.lineplot(data=normal_data, x='시간', y='온도', hue='이상치_여부', 
             palette=palette, alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title('시간에 따른 온도 변화 (이상치 표시)')

plt.tight_layout()
plt.show()

# 인사이트 도출
outlier_data = normal_data[normal_data['이상치_여부'] == '이상치']
normal_only = normal_data[normal_data['이상치_여부'] == '정상']

print("이상치 탐지 결과:")
print(f"전체 데이터: {len(normal_data)}개")
print(f"정상 데이터: {len(normal_only)}개 ({len(normal_only)/len(normal_data):.1%})")
print(f"이상치: {len(outlier_data)}개 ({len(outlier_data)/len(normal_data):.1%})")

print("\n센서별 평균값 비교:")
for sensor in ['온도', '압력', '진동', '전류']:
    normal_mean = normal_only[sensor].mean()
    outlier_mean = outlier_data[sensor].mean()
    print(f"{sensor}: 정상 {normal_mean:.2f}, 이상치 {outlier_mean:.2f} (차이: {abs(outlier_mean-normal_mean):.2f})")
```

### 해결책
1. **탐지**: Isolation Forest 알고리즘으로 다변량 이상치 탐지
2. **시각화**: 박스 플롯, 산점도, 히트맵으로 이상치 패턴 분석
3. **처리**: 이상치 제거 또는 보정을 통한 데이터 정제

## 예시 2: 피처 선택 및 중요도 분석

### 문제 상황
고객 이탈 예측 모델에서 어떤 피처가 중요한지 파악하고 최적의 피처 조합을 찾아야 함

### 해결 과정

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# 문제: 고객 이탈 예측에서 중요 피처 식별
# 데이터 생성
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    '가입기간': np.random.exponential(scale=24, size=n_customers),
    '월요금': np.random.normal(65, 20, n_customers),
    '나이': np.random.normal(40, 12, n_customers),
    '사용량': np.random.normal(20, 5, n_customers),
    '고객지원횟수': np.random.poisson(2, n_customers),
    '지연결제횟수': np.random.poisson(0.5, n_customers),
    '추천인수': np.random.poisson(1, n_customers),
    '서비스만족도': np.random.normal(3.5, 1, n_customers)
})

# 이탈 여부 생성 (다양한 피처 영향)
churn_prob = (
    0.1 +  # 기본 이탈 확률
    0.3 * (customer_data['월요금'] > 80) +  # 높은 요금
    0.2 * (customer_data['가입기간'] < 6) +  # 짧은 가입 기간
    0.15 * (customer_data['지연결제횟수'] > 0) +  # 지연 결제
    0.1 * (customer_data['서비스만족도'] < 3)  # 낮은 만족도
)

customer_data['이탈여부'] = (np.random.random(n_customers) < churn_prob).astype(int)

# 값 제한
customer_data['가입기간'] = customer_data['가입기간'].clip(1, 72)
customer_data['월요금'] = customer_data['월요금'].clip(20, 150)
customer_data['나이'] = customer_data['나이'].clip(18, 80)
customer_data['사용량'] = customer_data['사용량'].clip(5, 40)
customer_data['서비스만족도'] = customer_data['서비스만족도'].clip(1, 5)

# 피처 중요도 분석
X = customer_data.drop('이탈여부', axis=1)
y = customer_data['이탈여부']

# 랜덤 포레스트로 피처 중요도 계산
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importance = pd.DataFrame({
    '피처': X.columns,
    '중요도': rf_model.feature_importances_
}).sort_values('중요도', ascending=False)

# 통계적 피처 선택 (SelectKBest)
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# 재귀적 피처 제거 (RFE)
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=5)
rfe.fit(X, y)
rfe_selected = X.columns[rfe.support_]

# 피처 선택 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 랜덤 포레스트 피처 중요도
sns.barplot(data=feature_importance, x='중요도', y='피처', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('랜덤 포레스트 피처 중요도')

# 2. SelectKBest 점수
kbest_scores = pd.DataFrame({
    '피처': X.columns,
    'F-점수': selector.scores_
}).sort_values('F-점수', ascending=False)

sns.barplot(data=kbest_scores, x='F-점수', y='피처', palette='plasma', ax=axes[0, 1])
axes[0, 1].set_title('SelectKBest F-점수')

# 3. 피처 선택 방법 비교
selection_comparison = pd.DataFrame({
    '피처': X.columns,
    '랜덤 포레스트': X.columns.isin(feature_importance['피처'].head(5)),
    'SelectKBest': X.columns.isin(selected_features),
    'RFE': X.columns.isin(rfe_selected)
})

selection_comparison['총 선택 횟수'] = selection_comparison[['랜덤 포레스트', 'SelectKBest', 'RFE']].sum(axis=1)
selection_comparison_sorted = selection_comparison.sort_values('총 선택 횟수', ascending=False)

sns.barplot(data=selection_comparison_sorted, x='총 선택 횟수', y='피처', palette='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('피처 선택 방법별 일치 횟수')

# 4. 피처 수에 따른 모델 성능
feature_counts = range(1, len(X.columns) + 1)
cv_scores = []

for count in feature_counts:
    top_features = feature_importance['피처'].head(count)
    X_subset = X[top_features]
    scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), 
                           X_subset, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

sns.lineplot(x=feature_counts, y=cv_scores, marker='o', ax=axes[1, 1])
axes[1, 1].set_title('피처 수에 따른 교차 검증 정확도')
axes[1, 1].set_xlabel('피처 수')
axes[1, 1].set_ylabel('교차 검증 정확도')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 인사이트 도출
print("피처 중요도 분석 결과:")
print(feature_importance.round(3))

print("\n피처 선택 방법 비교:")
print(f"SelectKBest 선택 피처: {list(selected_features)}")
print(f"RFE 선택 피처: {list(rfe_selected)}")

print(f"\n최적 피처 수: {feature_counts[np.argmax(cv_scores)]}개")
print(f"최고 교차 검증 정확도: {max(cv_scores):.3f}")

# 공통으로 선택된 피처
common_features = set(selected_features) & set(rfe_selected) & set(feature_importance['피처'].head(5).values)
print(f"\n모든 방법에서 공통으로 선택된 피처: {list(common_features)}")
```

### 해결책
1. **중요도 분석**: 랜덤 포레스트, 통계적 검정, 재귀적 제거로 피처 중요도 평가
2. **최적 조합**: 다양한 피처 선택 방법 비교로 최적의 피처 조합 탐색
3. **성능 검증**: 교차 검증으로 피처 수에 따른 모델 성능 변화 분석

## 예시 3: 모델 해석 및 설명

### 문제 상황
복잡한 머신러닝 모델의 예측 결과를 비전문가에게 설명해야 함

### 해결 과정

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import shap  # SHAP 라이브러리 (설치 필요: pip install shap)

# 문제: 신용 평가 모델의 예측 결과 해석
# 데이터 생성
np.random.seed(42)
n_applicants = 1000

credit_data = pd.DataFrame({
    '소득': np.random.lognormal(10, 0.5, n_applicants),
    '부채비율': np.random.beta(2, 5, n_applicants),
    '신용기간': np.random.exponential(scale=60, n_applicants),
    '연체횟수': np.random.poisson(0.5, n_applicants),
    '자산': np.random.lognormal(9, 1, n_applicants),
    '나이': np.random.normal(40, 10, n_applicants),
    '고용형태': np.random.choice(['정규직', '계약직', '자영업'], n_applicants, p=[0.6, 0.3, 0.1])
})

# 신용 점수 계산 (복잡한 비선형 관계)
credit_score = (
    300 +  # 기본 점수
    200 * np.tanh((credit_data['소득'] - 50000) / 30000) +  # 소득의 비선형 효과
    150 * (1 - credit_data['부채비율']) +  # 부채비율
    100 * np.tanh(credit_data['신용기간'] / 120) +  # 신용 기간
    100 * np.exp(-credit_data['연체횟수']) -  # 연체 횟수
    50 * np.tanh((credit_data['자산'] - 10000) / 20000)  # 자산
)

# 고용형태에 따른 조정
employment_adjustment = {'정규직': 50, '계약직': 0, '자영업': -30}
credit_score += credit_data['고용형태'].map(employment_adjustment)

# 나이에 따른 조정
credit_score += 20 * np.sin((credit_data['나이'] - 25) * np.pi / 30)

# 노이즈 추가
credit_score += np.random.normal(0, 30, n_applicants)

# 신용 등급 (A, B, C, D)
credit_data['신용등급'] = pd.cut(credit_score, 
                                 bins=[0, 500, 600, 700, 850], 
                                 labels=['D', 'C', 'B', 'A'])

# 데이터 전처리
credit_data['나이'] = credit_data['나이'].clip(20, 70)
credit_data['신용기간'] = credit_data['신용기간'].clip(0, 300)
credit_data['연체횟수'] = credit_data['연체횟수'].clip(0, 10)

# 범주형 변수 인코딩
credit_encoded = pd.get_dummies(credit_data, columns=['고용형태'], drop_first=True)

# 모델 훈련
X_credit = credit_encoded.drop('신용등급', axis=1)
y_credit = credit_encoded['신용등급']

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_credit, y_credit)

# 모델 해석 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 특성 중요도
feature_imp = pd.DataFrame({
    '특성': X_credit.columns,
    '중요도': gb_model.feature_importances_
}).sort_values('중요도', ascending=False)

sns.barplot(data=feature_imp.head(10), x='중요도', y='특성', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('신용 평가 모델 특성 중요도')

# 2. 순열 중요도
perm_importance = permutation_importance(gb_model, X_credit, y_credit, 
                                       n_repeats=10, random_state=42)
perm_imp_df = pd.DataFrame({
    '특성': X_credit.columns,
    '순열 중요도': perm_importance.importances_mean
}).sort_values('순열 중요도', ascending=False)

sns.barplot(data=perm_imp_df.head(10), x='순열 중요도', y='특성', palette='plasma', ax=axes[0, 1])
axes[0, 1].set_title('순열 중요도 (Permutation Importance)')

# 3. 신용 등급별 특성 분포
for i, feature in enumerate(['소득', '부채비율', '신용기간']):
    if i < 3:
        row, col = 0, i+2
        sns.boxplot(data=credit_data, x='신용등급', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'신용등급별 {feature} 분포')

# 4. 예측 확률 분석
y_proba = gb_model.predict_proba(X_credit)
proba_df = pd.DataFrame(y_proba, columns=['D', 'C', 'B', 'A'])
proba_df['실제등급'] = y_credit.values
proba_melted = pd.melt(proba_df, id_vars=['실제등급'], 
                      value_vars=['D', 'C', 'B', 'A'],
                      var_name='예측등급', value_name='확률')

# 실제 등급과 예측 확률 관계
for i, grade in enumerate(['A', 'B', 'C', 'D']):
    if i < 2:
        row, col = 1, i
        grade_data = proba_melted[proba_melted['실제등급'] == grade]
        sns.boxplot(data=grade_data, x='예측등급', y='확률', ax=axes[row, col])
        axes[row, col].set_title(f'실제 {grade}등급의 예측 확률')
        axes[row, col].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 개별 예측 해석 (SHAP 값 사용 예시)
# 실제 SHAP 분석을 위해서는 shap 라이브러리 설치 필요
try:
    import shap
    
    # SHAP 값 계산
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_credit)
    
    # SHAP 요약 플롯
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_credit, plot_type="bar")
    plt.title('SHAP 값으로 본 특성 중요도')
    
    # 개별 예측 해석
    sample_idx = 0  # 첫 번째 샘플
    plt.figure(figsize=(10, 6))
    shap.force_plot(explainer.expected_value[0], shap_values[0][sample_idx], 
                   X_credit.iloc[sample_idx], matplotlib=True)
    plt.title(f'샘플 {sample_idx}의 예측 해석 (SHAP)')
    
except ImportError:
    print("SHAP 라이브러리가 설치되지 않았습니다. 설치 명령: pip install shap")

# 인사이트 도출
print("신용 평가 모델 해석 결과:")
print("\n상위 5개 중요 특성:")
print(feature_imp.head().round(3))

print("\n신용 등급별 평균 특성값:")
for grade in ['A', 'B', 'C', 'D']:
    grade_data = credit_data[credit_data['신용등급'] == grade]
    print(f"\n{grade}등급:")
    for feature in ['소득', '부채비율', '신용기간', '연체횟수']:
        mean_val = grade_data[feature].mean()
        print(f"  - 평균 {feature}: {mean_val:.2f}")
```

### 해결책
1. **전역적 해석**: 특성 중요도, 순열 중요도로 모델 전체 동작 설명
2. **국소적 해석**: SHAP 값으로 개별 예측의 이유 설명
3. **시각적 설명**: 박스 플롯, 히트맵으로 복잡한 관계 직관적 표현

## 예시 4: 모델 성능 모니터링

### 문제 상황
배포된 머신러닝 모델의 성능이 시간에 따라 어떻게 변화하는지 모니터링해야 함

### 해결 과정

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime

# 문제: 추천 시스템 모델 성능 모니터링
# 시계열 성능 데이터 생성
np.random.seed(42)
days = 90
dates = pd.date_range(end=datetime.datetime.now(), periods=days)

# 기본 성능 지표
base_accuracy = 0.85
base_precision = 0.82
base_recall = 0.88
base_f1 = 0.85

performance_data = []

for i, date in enumerate(dates):
    # 시간에 따른 성능 변화 (모델 드리프트)
    drift_factor = 1.0 - (i / days) * 0.1  # 점진적 성능 저하
    
    # 주간 패턴
    weekly_pattern = 0.02 * np.sin(i * 2 * np.pi / 7)
    
    # 랜덤 변동
    random_noise = np.random.normal(0, 0.01)
    
    # 이벤트 기반 성능 변화
    event_effect = 0
    if i in [30, 31]:  # 모델 업데이트
        event_effect = 0.03
    elif i in [60, 61]:  # 데이터 분포 변화
        event_effect = -0.02
    
    # 최종 성능 지표
    accuracy = base_accuracy * drift_factor + weekly_pattern + random_noise + event_effect
    precision = base_precision * drift_factor + weekly_pattern * 0.8 + random_noise + event_effect
    recall = base_recall * drift_factor + weekly_pattern * 1.2 + random_noise + event_effect
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # 사용자 피드백
    user_satisfaction = np.random.normal(4.0, 0.5) * drift_factor
    user_satisfaction = max(1.0, min(5.0, user_satisfaction))
    
    # 시스템 지표
    response_time = 200 + i * 2 + np.random.normal(0, 20)  # 점차 느려짐
    daily_requests = np.random.poisson(1000) * (1 + np.random.normal(0, 0.1))
    
    performance_data.append({
        '날짜': date,
        '정확도': accuracy,
        '정밀도': precision,
        '재현율': recall,
        'F1점수': f1,
        '사용자만족도': user_satisfaction,
        '응답시간(ms)': response_time,
        '일일요청수': daily_requests
    })

perf_df = pd.DataFrame(performance_data)

# 성능 모니터링 시각화
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. 정확도 시계열
sns.lineplot(data=perf_df, x='날짜', y='정확도', ax=axes[0, 0])
axes[0, 0].set_title('모델 정확도 추이')
axes[0, 0].set_ylabel('정확도')

# 모델 업데이트 시점 표시
for date in [perf_df['날짜'][30], perf_df['날짜'][60]]:
    axes[0, 0].axvline(date, color='red', linestyle='--', alpha=0.7)

# 2. 정밀도-재현율 시계열
sns.lineplot(data=perf_df, x='날짜', y='정밀도', label='정밀도', ax=axes[0, 1])
sns.lineplot(data=perf_df, x='날짜', y='재현율', label='재현율', ax=axes[0, 1])
axes[0, 1].set_title('정밀도와 재현율 추이')
axes[0, 1].legend()

# 3. F1 점수 시계열
sns.lineplot(data=perf_df, x='날짜', y='F1점수', ax=axes[0, 2])
axes[0, 2].set_title('F1 점수 추이')

# 4. 사용자 만족도 시계열
sns.lineplot(data=perf_df, x='날짜', y='사용자만족도', ax=axes[1, 0])
axes[1, 0].set_title('사용자 만족도 추이')
axes[1, 0].set_ylabel('만족도 (1-5)')
axes[1, 0].set_ylim(1, 5)

# 5. 응답 시간 시계열
sns.lineplot(data=perf_df, x='날짜', y='응답시간(ms)', ax=axes[1, 1])
axes[1, 1].set_title('평균 응답 시간 추이')
axes[1, 1].set_ylabel('응답 시간 (ms)')

# 6. 일일 요청 수 시계열
sns.lineplot(data=perf_df, x='날짜', y='일일요청수', ax=axes[1, 2])
axes[1, 2].set_title('일일 요청 수 추이')

# 7. 성능 지표 간 상관관계
perf_metrics = ['정확도', '정밀도', '재현율', 'F1점수', '사용자만족도']
corr_matrix = perf_df[perf_metrics].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2, 0])
axes[2, 0].set_title('성능 지표 간 상관관계')

# 8. 성능 분포 (최근 30일 vs 이전 30일)
recent_30 = perf_df.tail(30)
previous_30 = perf_df.iloc[-60:-30]

perf_comparison = pd.DataFrame({
    '기간': ['최근 30일'] * len(recent_30) + ['이전 30일'] * len(previous_30),
    '정확도': list(recent_30['정확도']) + list(previous_30['정확도'])
})

sns.boxplot(data=perf_comparison, x='기간', y='정확도', ax=axes[2, 1])
axes[2, 1].set_title('기간별 정확도 분포 비교')

# 9. 이상 감지 (통계적 과정 관리)
mean_acc = perf_df['정확도'].mean()
std_acc = perf_df['정확도'].std()
upper_control = mean_acc + 2 * std_acc
lower_control = mean_acc - 2 * std_acc

sns.lineplot(data=perf_df, x='날짜', y='정확도', ax=axes[2, 2])
axes[2, 2].axhline(mean_acc, color='green', linestyle='-', alpha=0.7, label='평균')
axes[2, 2].axhline(upper_control, color='red', linestyle='--', alpha=0.7, label='관리 상한')
axes[2, 2].axhline(lower_control, color='red', linestyle='--', alpha=0.7, label='관리 하한')
axes[2, 2].set_title('통계적 과정 관리 (SPC)')
axes[2, 2].legend()

plt.tight_layout()
plt.show()

# 성능 저하 경고 시스템
def detect_performance_degradation(df, metric='정확도', window=7, threshold=0.05):
    """성능 저하 감지"""
    df_copy = df.copy()
    df_copy[f'{metric}_이동평균'] = df_copy[metric].rolling(window=window).mean()
    
    # 최근 성능과 이전 성능 비교
    recent_avg = df_copy[f'{metric}_이동평균'].iloc[-1]
    previous_avg = df_copy[f'{metric}_이동평균'].iloc[-window*2]
    
    degradation = (previous_avg - recent_avg) / previous_avg
    
    return degradation, recent_avg, previous_avg

degradation, recent_avg, previous_avg = detect_performance_degradation(perf_df)

print("모델 성능 모니터링 결과:")
print(f"모니터링 기간: {perf_df['날짜'].min().date()} ~ {perf_df['날짜'].max().date()}")
print(f"평균 정확도: {perf_df['정확도'].mean():.3f}")
print(f"평균 사용자 만족도: {perf_df['사용자만족도'].mean():.2f}/5")
print(f"평균 응답 시간: {perf_df['응답시간(ms)'].mean():.1f}ms")

print(f"\n성능 저하 분석:")
print(f"최근 7일 평균 정확도: {recent_avg:.3f}")
print(f"이전 7일 평균 정확도: {previous_avg:.3f}")
print(f"성능 저하율: {degradation:.1%}")

if degradation > threshold:
    print("⚠️ 경고: 모델 성능이 5% 이상 저하되었습니다. 재학습을 고려하세요.")
else:
    print("✅ 모델 성능이 안정적인 범위 내에 있습니다.")
```

### 해결책
1. **시계열 모니터링**: 주요 성능 지표의 시간에 따른 변화 추적
2. **이상 감지**: 통계적 과정 관리로 비정상적인 성능 변화 탐지
3. **경고 시스템**: 성능 저하 시 자동 알림 및 재학습 권장

## 결론

이러한 실용적인 예시들은 머신러닝 프로젝트에서 Seaborn을 활용하여 복잡한 문제들을 해결하는 방법을 보여줍니다:

1. **데이터 품질 관리**: 이상치 탐지 및 처리
2. **피처 엔지니어링**: 중요 피처 식별 및 선택
3. **모델 해석**: 복잡한 모델의 동작 설명
4. **운영 모니터링**: 배포된 모델의 성능 추적

각 예시는 실제 비즈니스 문제를 중심으로 구성되어 있어, 실제 머신러닝 프로젝트에 직접 적용할 수 있는 실용적인 접근 방식을 제공합니다.

## 다음 단계

실용적인 문제 해결 예시를 익혔다면, [참조 요약 및 치트 시트](12-reference.md) 문서에서 Seaborn의 핵심 기능들을 빠르게 참조할 수 있는 자료를 확인해보세요.

## 추가 자료

- [Scikit-learn 모델 평가 가이드](https://scikit-learn.org/stable/model_evaluation.html)
- [SHAP 라이브러리 문서](https://shap.readthedocs.io/)
- [머신러닝 운영 (MLOps) 가이드](https://ml-ops.org/)
- [머신러닝 문제 해결 프레임워크](ml-problem-solution-framework.md)