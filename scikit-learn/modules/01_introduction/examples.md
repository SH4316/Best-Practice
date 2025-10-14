# 모듈 1: 실용적인 코드 예제

## 예제 1: scikit-learn 설치 확인

```python
# scikit-learn 및 버전 확인
import sklearn
import numpy as np
import pandas as pd

print(f"scikit-learn 버전: {sklearn.__version__}")
print(f"NumPy 버전: {np.__version__}")
print(f"pandas 버전: {pd.__version__}")

# 모든 주요 모듈이 사용 가능한지 확인
from sklearn import datasets, model_selection, preprocessing, metrics
print("모든 주요 모듈이 성공적으로 임포트되었습니다!")
```

**설명**: 이 코드는 scikit-learn과 의존성 패키지가 제대로 설치되었는지 확인합니다. 다른 환경에서 작업할 때 재현성을 위해 버전을 확인하는 것이 좋은 방법입니다.

## 예제 2: 첫 번째 데이터셋 로드 및 탐색

```python
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# iris 데이터셋 로드
iris = load_iris()

# 더 쉬운 탐색을 위해 DataFrame 생성
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 기본 정보 표시
print("데이터셋 형태:", iris_df.shape)
print("\n처음 5개 행:")
print(iris_df.head())

print("\n데이터셋 정보:")
iris_df.info()

print("\n통계 요약:")
print(iris_df.describe())

print("\n클래스 분포:")
print(iris_df['species'].value_counts())

# 데이터 시각화
plt.figure(figsize=(12, 6))

# 꽃받침 길이 vs 꽃받침 너비 산점도
plt.subplot(1, 2, 1)
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', 
                hue='species', style='species', s=100)
plt.title('꽃받침 길이 vs 꽃받침 너비')

# 꽃잎 길이 vs 꽃잎 너비 산점도
plt.subplot(1, 2, 2)
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', 
                hue='species', style='species', s=100)
plt.title('꽃잎 길이 vs 꽃잎 너비')

plt.tight_layout()
plt.show()
```

**설명**: 이 예제는 내장 데이터셋을 로드하는 방법, pandas DataFrame으로 변환하여 쉽게 조작하는 방법, 그리고 기본적인 탐색적 데이터 분석을 수행하는 방법을 보여줍니다. iris 데이터셋은 머신러닝 분류 작업을 학습하기에 완벽한 고전적인 데이터셋입니다.

## 예제 3: 첫 번째 머신러닝 모델 - 분류

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 재현성을 위한 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터를 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"훈련 세트 형태: {X_train.shape}")
print(f"테스트 세트 형태: {X_test.shape}")
print(f"훈련 세트 클래스 분포: {np.bincount(y_train)}")
print(f"테스트 세트 클래스 분포: {np.bincount(y_test)}")

# K-최근접 이웃 분류기 생성 및 훈련
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 테스트 세트에서 예측 수행
y_pred = knn.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델 정확도: {accuracy:.3f}")

# 상세 분류 보고서 표시
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 혼동 행렬 생성
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('예측된 레이블')
plt.ylabel('실제 레이블')
plt.title('혼동 행렬')
plt.show()

# 새로운 데이터에 대한 예측
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # setosa일 가능성이 높음
    [6.7, 3.0, 5.2, 2.3],  # virginica일 가능성이 높음
    [5.9, 3.0, 4.2, 1.5]   # versicolor일 가능성이 높음
])

predictions = knn.predict(new_flowers)
predicted_species = [iris.target_names[p] for p in predictions]

print("\n새로운 꽃에 대한 예측:")
for i, species in enumerate(predicted_species):
    print(f"꽃 {i+1}: {species}")
```

**설명**: 이 예제는 완전한 머신러닝 워크플로우를 보여줍니다:
1. 데이터 로드
2. 훈련/테스트 세트로 분할 (클래스 분포 유지를 위한 층화 추출)
3. K-최근접 이웃 분류기 훈련
4. 예측 수행
5. 정확도, 분류 보고서, 혼동 행렬을 사용한 성능 평가
6. 새로운 보이지 않는 데이터에 대한 예측

## 예제 4: 첫 번째 머신러닝 모델 - 회귀

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위한 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 합성 회귀 데이터 생성
X, y = make_regression(
    n_samples=100, 
    n_features=1, 
    noise=20, 
    random_state=RANDOM_STATE
)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"훈련 세트 형태: {X_train.shape}")
print(f"테스트 세트 형태: {X_test.shape}")

# 선형 회귀 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 수행
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 모델 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n훈련 MSE: {train_mse:.2f}")
print(f"테스트 MSE: {test_mse:.2f}")
print(f"훈련 R²: {train_r2:.3f}")
print(f"테스트 R²: {test_r2:.3f}")

# 모델 파라미터 표시
print(f"\n모델 계수: {model.coef_[0]:.3f}")
print(f"모델 절편: {model.intercept_:.3f}")

# 결과 시각화
plt.figure(figsize=(12, 5))

# 훈련 데이터
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.7, label='실제값')
plt.plot(X_train, y_train_pred, 'r-', label='예측값')
plt.xlabel('특성')
plt.ylabel('타겟')
plt.title('훈련 데이터')
plt.legend()
plt.grid(True)

# 테스트 데이터
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.7, label='실제값')
plt.plot(X_test, y_test_pred, 'r-', label='예측값')
plt.xlabel('특성')
plt.ylabel('타겟')
plt.title('테스트 데이터')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 새로운 데이터에 대한 예측
new_points = np.array([[-2], [0], [2]])
new_predictions = model.predict(new_points)

print("\n새로운 점에 대한 예측:")
for i, pred in enumerate(new_predictions):
    print(f"점 {new_points[i][0]:.1f}: {pred:.2f}")
```

**설명**: 이 예제는 합성 데이터를 사용한 완전한 회귀 워크플로우를 보여줍니다. 다음을 수행하는 방법을 보여줍니다:
1. 회귀 데이터 생성
2. 선형 회귀 모델 훈련
3. MSE와 R²를 사용한 성능 평가
4. 모델 적합도 시각화
5. 새로운 데이터에 대한 예측

## 예제 5: scikit-learn API 일관성 이해하기

```python
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import numpy as np

# 재현성을 위한 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 분류 데이터 생성
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=10, n_informative=5, 
    n_redundant=2, random_state=RANDOM_STATE
)

# 회귀 데이터 생성
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, n_informative=5, 
    noise=0.1, random_state=RANDOM_STATE
)

# 두 데이터셋 분할
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

# 일관된 인터페이스로 모델 초기화
classifiers = [
    ('로지스틱 회귀', LogisticRegression(random_state=RANDOM_STATE)),
    ('K-최근접 이웃', KNeighborsClassifier(n_neighbors=5)),
    ('서포트 벡터 머신', SVC(random_state=RANDOM_STATE))
]

regressors = [
    ('선형 회귀', LinearRegression()),
    ('K-최근접 이웃', KNeighborsRegressor(n_neighbors=5)),
    ('서포트 벡터 회귀', SVR())
]

# 모델 훈련 및 평가 함수
def evaluate_models(models, X_train, X_test, y_train, y_test, task_type='classification'):
    print(f"\n{task_type.title()} 결과:")
    print("-" * 50)
    
    for name, model in models:
        # 모든 모델이 동일한 API를 따름
        model.fit(X_train, y_train)  # 모델 훈련
        score = model.score(X_test, y_test)  # 모델 평가
        
        print(f"{name}: {score:.3f}")

# 분류기 평가
evaluate_models(classifiers, X_clf_train, X_clf_test, y_clf_train, y_clf_test)

# 회귀기 평가
evaluate_models(regressors, X_reg_train, X_reg_test, y_reg_train, y_reg_test)

# 변환기 API 시연
scaler = StandardScaler()

# 훈련 데이터에 맞추고 변환
X_clf_train_scaled = scaler.fit_transform(X_clf_train)

# 테스트 데이터만 변환 (훈련 데이터에서 학습된 파라미터 사용)
X_clf_test_scaled = scaler.transform(X_clf_test)

# 스케일링된 데이터로 모델 훈련
model = LogisticRegression(random_state=RANDOM_STATE)
model.fit(X_clf_train_scaled, y_clf_train)

# 성능 비교
original_score = model.score(X_clf_test, y_clf_test)
scaled_score = model.score(X_clf_test_scaled, y_clf_test)

print(f"\n로지스틱 회귀 성능:")
print(f"원본 데이터: {original_score:.3f}")
print(f"스케일링된 데이터: {scaled_score:.3f}")
```

**설명**: 이 예제는 다른 알고리즘 간의 scikit-learn API 일관성을 보여줍니다. 모든 추정기는 동일한 패턴을 따릅니다:
1. 파라미터로 초기화
2. `fit()` 메서드로 훈련
3. `predict()` 또는 `score()` 메서드로 예측/평가

이 일관성으로 인해 다른 알고리즘 간에 쉽게 전환하고 성능을 비교할 수 있습니다.

## 예제 6: 재현 가능한 머신러닝을 위한 모범 사례

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# 재현성을 위한 설정
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'model_params': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    }
}

# 랜덤 시드 설정
np.random.seed(CONFIG['random_state'])

# 데이터 로드 및 준비
data = load_breast_cancer()
X, y = data.data, data.target

# 더 나은 해석을 위해 특성 이름 생성
feature_names = data.feature_names.tolist()

# 층화 추출로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=CONFIG['test_size'], 
    random_state=CONFIG['random_state'],
    stratify=y  # 클래스 분포 유지
)

# 문서화를 위한 데이터 정보 저장
data_info = {
    'n_samples': X.shape[0],
    'n_features': X.shape[1],
    'n_classes': len(np.unique(y)),
    'class_distribution': dict(zip(np.unique(y, return_counts=True)[0], 
                                  np.unique(y, return_counts=True)[1])),
    'feature_names': feature_names
}

print("데이터 정보:")
for key, value in data_info.items():
    print(f"{key}: {value}")

# 데이터 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 문서화된 파라미터로 모델 훈련
model = LogisticRegression(**CONFIG['model_params'])
model.fit(X_train_scaled, y_train)

# 예측 수행
y_pred = model.predict(X_test_scaled)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델 정확도: {accuracy:.3f}")

# 특성 중요도 표시 (계수)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\n중요한 특성 상위 5개:")
print(feature_importance.head())

# 모델 설정 및 결과 저장
results = {
    'config': CONFIG,
    'data_info': data_info,
    'model_accuracy': accuracy,
    'feature_importance': feature_importance.head().to_dict()
}

print("\n실험이 성공적으로 완료되었습니다!")
print("재현성을 위해 모든 파라미터와 결과가 문서화되었습니다.")
```

**설명**: 이 예제는 재현 가능한 머신러닝을 위한 모범 사례를 보여줍니다:
1. 파라미터 저장을 위한 설정 딕셔너리 사용
2. 재현성을 위한 랜덤 시드 설정
3. 데이터 특성 문서화
4. 층화 샘플링 사용
5. 적절한 전처리 워크플로우
6. 모든 파라미터와 결과 기록

이 예제들은 scikit-learn API와 모범 사례를 이해하는 데 견고한 기반을 제공합니다. 분류 및 회귀 작업 모두에 대한 필수 워크플로우를 다루면서 scikit-learn 라이브러리의 일관성과 강력함을 보여줍니다.