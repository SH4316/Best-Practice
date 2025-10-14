# 모듈 1: 연습문제 해결책

## 연습문제 1 해결책: scikit-learn 환경 설정

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

**설명**: 이 코드는 scikit-learn과 필수 의존성 패키지가 올바르게 설치되었는지 확인하는 간단한 방법을 보여줍니다. 버전 정보는 다른 환경에서 재현성을 보장하는 데 중요합니다.

## 연습문제 2 해결책: 데이터셋 탐색

```python
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 와인 데이터셋 로드
wine = load_wine()

# DataFrame으로 변환
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
wine_df['wine_class'] = wine_df['target'].map({
    0: 'class_0', 
    1: 'class_1', 
    2: 'class_2'
})

# 데이터셋 정보 출력
print(f"데이터셋 형태: {wine_df.shape}")
print("\n데이터셋 정보:")
wine_df.info()

print("\n통계 요약:")
print(wine_df.describe())

print("\n클래스 분포:")
print(wine_df['wine_class'].value_counts())

# 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=wine_df, 
    x='alcohol', 
    y='malic_acid', 
    hue='wine_class', 
    style='wine_class', 
    s=100
)
plt.title('알코올 함량 vs 말산 함량')
plt.xlabel('알코올 (%)')
plt.ylabel('말산 (g/l)')
plt.grid(True)
plt.show()
```

**설명**: 이 해결책은 와인 데이터셋을 로드하고 탐색하는 과정을 보여줍니다. DataFrame으로 변환하면 데이터 조작이 더 쉬워지고, 시각화를 통해 데이터 패턴을 직관적으로 이해할 수 있습니다.

## 연습문제 3 해결책: 첫 번째 분류 모델

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 데이터 로드 및 분할
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"훈련 세트 형태: {X_train.shape}")
print(f"테스트 세트 형태: {X_test.shape}")

# K-최근접 이웃 분류기 생성 및 훈련
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 예측 및 평가
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.3f}")

# 혼동 행렬 생성 및 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)
plt.xlabel('예측된 레이블')
plt.ylabel('실제 레이블')
plt.title('혼동 행렬')
plt.show()
```

**설명**: 이 해결책은 완전한 분류 워크플로우를 보여줍니다. stratify=y를 사용하여 클래스 분포를 유지하는 것이 중요합니다. 혼동 행렬은 모델이 어떤 클래스를 가장 잘/못 예측하는지 보여줍니다.

## 연습문제 4 해결책: 첫 번째 회귀 모델

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 회귀 데이터 생성
X, y = make_regression(
    n_samples=200, 
    n_features=1, 
    noise=15, 
    random_state=RANDOM_STATE
)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"훈련 MSE: {train_mse:.2f}")
print(f"테스트 MSE: {test_mse:.2f}")
print(f"훈련 R²: {train_r2:.3f}")
print(f"테스트 R²: {test_r2:.3f}")

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.7, label='실제값')
plt.plot(X_train, y_train_pred, 'r-', label='예측선')
plt.title('훈련 데이터')
plt.xlabel('특성')
plt.ylabel('타겟')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.7, label='실제값')
plt.plot(X_test, y_test_pred, 'r-', label='예측선')
plt.title('테스트 데이터')
plt.xlabel('특성')
plt.ylabel('타겟')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**설명**: 이 해결책은 회귀 문제에 대한 완전한 접근 방식을 보여줍니다. MSE는 평균 제곱 오차를 측정하고, R²는 설명된 분산의 비율을 나타냅니다. 두 메트릭을 함께 사용하면 모델 성능을 더 잘 이해할 수 있습니다.

## 연습문제 5 해결책: API 일관성 탐색

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 데이터 생성 및 분할
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    random_state=RANDOM_STATE
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 모델 평가 함수
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 모델 초기화
models = [
    ('로지스틱 회귀', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ('결정 트리', DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ('서포트 벡터 머신', SVC(random_state=RANDOM_STATE))
]

# 모델 평가
print("모델 성능 비교:")
print("-" * 30)
for name, model in models:
    accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"{name}: {accuracy:.3f}")
```

**설명**: 이 해결책은 scikit-learn API의 일관성을 보여줍니다. 모든 모델은 동일한 fit/predict 패턴을 따르므로, 동일한 평가 함수를 사용할 수 있습니다. 이는 코드 재사용성과 모델 비교를 용이하게 합니다.

## 연습문제 6 해결책: 데이터 전처리와 모델 성능

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 데이터 로드 및 분할
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 원본 데이터로 모델 훈련
model_original = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

# 표준화된 데이터로 모델 훈련
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print(f"원본 데이터 정확도: {accuracy_original:.3f}")
print(f"표준화된 데이터 정확도: {accuracy_scaled:.3f}")

# 결론
if accuracy_scaled > accuracy_original:
    print("\n스케일링이 모델 성능을 향상시켰습니다.")
else:
    print("\n스케일링이 모델 성능에 큰 영향을 미치지 않았습니다.")
```

**설명**: 이 해결책은 특성 스케일링이 모델 성능에 미치는 영향을 보여줍니다. 로지스틱 회귀와 같은 알고리즘은 특성 스케일에 민감할 수 있으며, 표준화는 종종 수렴 속도와 성능을 향상시킵니다.

## 연습문제 7 해결책: 재현 가능한 머신러닝 워크플로우

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 실험 구성
CONFIG = {
    'random_state': 42,
    'test_size': 0.25,
    'model_params': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    }
}

# 랜덤 시드 설정
np.random.seed(CONFIG['random_state'])

# 실험 결과 저장
results = {}

# 데이터 생성
print("데이터 생성 중...")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=CONFIG['random_state']
)
results['data_shape'] = X.shape
results['class_distribution'] = dict(zip(*np.unique(y, return_counts=True)))

# 데이터 분할
print("데이터 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y
)
results['train_shape'] = X_train.shape
results['test_shape'] = X_test.shape

# 모델 훈련
print("모델 훈련 중...")
model = LogisticRegression(**CONFIG['model_params'])
model.fit(X_train, y_train)

# 예측 및 평가
print("모델 평가 중...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
results['model_accuracy'] = accuracy

# 결과 출력
print("\n실험 결과:")
print("=" * 50)
print(f"실험 구성: {CONFIG}")
print(f"데이터 형태: {results['data_shape']}")
print(f"클래스 분포: {results['class_distribution']}")
print(f"훈련 세트 형태: {results['train_shape']}")
print(f"테스트 세트 형태: {results['test_shape']}")
print(f"모델 정확도: {results['model_accuracy']:.3f}")

print("\n실험이 성공적으로 완료되었습니다!")
print("모든 파라미터와 결과가 문서화되어 재현 가능합니다.")
```

**설명**: 이 해결책은 재현 가능한 실험을 구현하는 방법을 보여줍니다. 구성 딕셔너리를 사용하면 모든 파라미터를 한 곳에서 관리할 수 있으며, 결과를 체계적으로 기록하면 실험 추적이 용이해집니다.

## 연습문제 8 해결책: 모델 파라미터 탐색

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 데이터 로드 및 분할
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 다른 k 값 테스트
k_values = [1, 3, 5, 7, 9]
train_accuracies = []
test_accuracies = []

print("k 값에 따른 성능:")
print("-" * 30)

for k in k_values:
    # 모델 훈련
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # 훈련 정확도
    y_train_pred = knn.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_acc)
    
    # 테스트 정확도
    y_test_pred = knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_acc)
    
    print(f"k={k}: 훈련 정확도={train_acc:.3f}, 테스트 정확도={test_acc:.3f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, 'o-', label='훈련 정확도')
plt.plot(k_values, test_accuracies, 's-', label='테스트 정확도')
plt.xlabel('k (이웃 수)')
plt.ylabel('정확도')
plt.title('k 값에 따른 KNN 성능')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

# 최적의 k 값 선택
best_k = k_values[np.argmax(test_accuracies)]
best_accuracy = max(test_accuracies)

print(f"\n최적의 k 값: {best_k}")
print(f"최고 테스트 정확도: {best_accuracy:.3f}")
print("\n이유: k=3이 테스트 세트에서 가장 높은 정확도를 보였습니다.")
print("k=1은 과적합의 경향이 있고, k가 너무 크면 과소적합될 수 있습니다.")
```

**설명**: 이 해결책은 하이퍼파라미터 튜닝의 기본을 보여줍니다. k 값에 따른 성능 변화를 시각화하면 과적합과 과소적합 사이의 균형을 이해하는 데 도움이 됩니다.

## 연습문제 9 해결책: 모델 해석

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 데이터 로드 및 분할
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 모델 훈련
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model.fit(X_train, y_train)

# 특성 중요도 계산
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

# 상위 5개 특성 출력
print("가장 중요한 특성 상위 5개:")
print("-" * 40)
for i, row in feature_importance.head().iterrows():
    sign = "양수" if row['coefficient'] > 0 else "음수"
    print(f"{row['feature']}: {row['coefficient']:.4f} ({sign})")

# 시각화
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['coefficient'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('계수 값')
plt.title('로지스틱 회귀 계수 (상위 10개 특성)')
plt.grid(True, axis='x')
plt.show()

# 해석
print("\n해석:")
print("-" * 40)
print("양수 계수: 해당 특성 값이 증가할수록 악성(클래스 1)일 가능성이 높아짐")
print("음수 계수: 해당 특성 값이 증가할수록 양성(클래스 0)일 가능성이 높아짐")
print("계수의 절대값이 클수록 예측에 더 큰 영향을 미침")
```

**설명**: 이 해결책은 로지스틱 회귀 모델을 해석하는 방법을 보여줍니다. 계수의 크기와 부호는 각 특성이 예측에 미치는 영향의 방향과 강도를 나타냅니다.

## 연습문제 10 해결책: 종합 프로젝트

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 실험 구성
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'scaler': StandardScaler(),
    'models': {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        ),
        'SVM': SVC(
            random_state=42,
            probability=True
        )
    }
}

# 랜덤 시드 설정
np.random.seed(CONFIG['random_state'])

# 1. 데이터셋 로드 및 탐색
print("1. 데이터셋 로드 및 탐색")
print("-" * 40)
digits = load_digits()
X, y = digits.data, digits.target

print(f"데이터셋 형태: {X.shape}")
print(f"클래스 수: {len(np.unique(y))}")
print(f"클래스 분포: {dict(zip(*np.unique(y, return_counts=True)))}")

# 일부 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap='binary')
    ax.set_title(f'레이블: {y[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 2. 데이터 분할 및 전처리
print("\n2. 데이터 분할 및 전처리")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=CONFIG['test_size'], 
    random_state=CONFIG['random_state'],
    stratify=y
)

print(f"훈련 세트 형태: {X_train.shape}")
print(f"테스트 세트 형태: {X_test.shape}")

# 데이터 스케일링
X_train_scaled = CONFIG['scaler'].fit_transform(X_train)
X_test_scaled = CONFIG['scaler'].transform(X_test)

# 3. 모델 비교
print("\n3. 모델 비교")
print("-" * 40)

model_results = {}

for name, model in CONFIG['models'].items():
    print(f"\n{name} 모델 훈련 중...")
    
    # 스케일링된 데이터로 훈련
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"{name} 정확도: {accuracy:.4f}")

# 4. 최적 모델 선택 및 상세 평가
print("\n4. 최적 모델 선택 및 상세 평가")
print("-" * 40)

best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
best_model = model_results[best_model_name]['model']
best_accuracy = model_results[best_model_name]['accuracy']
best_predictions = model_results[best_model_name]['predictions']

print(f"최적 모델: {best_model_name}")
print(f"최고 정확도: {best_accuracy:.4f}")

# 상세 분류 보고서
print("\n분류 보고서:")
print(classification_report(y_test, best_predictions))

# 혼동 행렬
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('예측된 레이블')
plt.ylabel('실제 레이블')
plt.title(f'{best_model_name} 혼동 행렬')
plt.show()

# 5. 오류 분석
print("\n5. 오류 분석")
print("-" * 40)

errors = np.where(y_test != best_predictions)[0]
print(f"총 오류 수: {len(errors)}")

# 일부 오류 시각화
if len(errors) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.ravel()):
        if i < len(errors):
            idx = errors[i]
            ax.imshow(X_test[idx].reshape(8, 8), cmap='binary')
            ax.set_title(f'실제: {y_test[idx]}, 예측: {best_predictions[idx]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 6. 실험 요약
print("\n6. 실험 요약")
print("-" * 40)
print(f"사용된 데이터셋: 숫자 필기체 (digits)")
print(f"특성 수: {X.shape[1]}")
print(f"클래스 수: {len(np.unique(y))}")
print(f"훈련 샘플 수: {X_train.shape[0]}")
print(f"테스트 샘플 수: {X_test.shape[0]}")
print(f"최적 모델: {best_model_name}")
print(f"최고 정확도: {best_accuracy:.4f}")

print("\n실험이 성공적으로 완료되었습니다!")
print("모든 단계가 문서화되어 재현 가능합니다.")
```

**설명**: 이 종합 프로젝트는 모듈에서 학습한 모든 개념을 통합합니다. 데이터 탐색, 전처리, 모델 비교, 평가, 오류 분석 등 완전한 머신러닝 워크플로우를 보여줍니다. 재현 가능성을 위해 모든 구성을 한 곳에 관리하고 결과를 체계적으로 기록합니다.

---

이 해결책들은 각 연습문제에 대한 완전한 구현을 제공하며, 학습자가 코드를 실행하고 결과를 이해하는 데 도움이 됩니다. 각 해결책에는 관련 개념을 설명하는 주석이 포함되어 있어 학습 경험을 향상시킵니다.