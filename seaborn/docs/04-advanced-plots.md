# Seaborn 고급 플롯 유형

Seaborn은 기본 플롯 외에도 복잡한 데이터 관계와 패턴을 시각화하는 다양한 고급 플롯을 제공합니다. 이 문서에서는 다음 고급 플롯 유형을 다룹니다:

1. **행렬 플롯 (Matrix Plots)**: 데이터 행렬을 시각화합니다
2. **회귀 플롯 (Regression Plots)**: 회귀 분석 결과를 시각화합니다
3. **다중 플롯 (Multi-plot Grids)**: 여러 플롯을 조합하여 복잡한 관계를 보여줍니다

## 행렬 플롯 (Matrix Plots)

행렬 플롯은 2차원 데이터를 색상으로 인코딩하여 시각화합니다.

### heatmap() - 히트맵

히트맵은 행렬 형태의 데이터를 색상으로 표현하여 패턴과 상관관계를 쉽게 파악할 수 있습니다.

#### 기본 히트맵

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 상관관계 행렬 데이터 생성
data = np.random.randn(10, 12)
corr = np.corrcoef(data)

plt.figure(figsize=(10, 8))
sns.heatmap(corr)
plt.title("상관관계 히트맵")
plt.show()
```

#### 주석과 색상 막대가 포함된 히트맵

```python
# 샘플 데이터 생성
flights = sns.load_dataset("flights")
flights_pivot = flights.pivot("month", "year", "passengers")

plt.figure(figsize=(12, 8))
sns.heatmap(
    flights_pivot,
    annot=True,           # 셀에 값 표시
    fmt="d",             # 정수 형식
    cmap="YlGnBu",       # 색상 팔레트
    linewidths=.5        # 셀 경계선
)
plt.title("연도 및 월별 항공편 승객 수")
plt.show()
```

#### 중심값이 있는 히트맵

```python
# 중심값을 지정하여 발산 색상 팔레트 사용
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    center=0,            # 중심값
    cmap="RdBu_r",       # 발산 색상 팔레트
    annot=True,          # 주석 표시
    fmt=".2f",           # 소수점 2자리
    square=True          # 정사각형 셀
)
plt.title("중심값이 있는 상관관계 히트맵")
plt.show()
```

### clustermap() - 클러스터링된 히트맵

클러스터링된 히트맵은 데이터의 계층적 클러스터링을 수행하여 유사한 행과 열을 그룹화합니다.

#### 기본 클러스터링

```python
# 샘플 데이터 생성
data = np.random.randn(20, 15)
data += np.arange(20).reshape(-1, 1)  # 행별 패턴 추가
data += np.arange(15).reshape(1, -1)  # 열별 패턴 추가

sns.clustermap(data)
plt.title("클러스터링된 히트맵")
plt.show()
```

#### 사용자 정의 클러스터링

```python
# iris 데이터셋으로 클러스터링
iris = sns.load_dataset("iris")
species = iris.pop("species")

sns.clustermap(
    iris,
    cmap="vlag",
    row_cluster=True,
    col_cluster=False,
    figsize=(10, 12),
    standard_scale=1,    # 열별 표준화
    row_colors=species.map({
        "setosa": "blue",
        "versicolor": "green",
        "virginica": "red"
    })
)
plt.title("Iris 데이터 클러스터링")
plt.show()
```

## 회귀 플롯 (Regression Plots)

회귀 플롯은 변수 간의 관계를 회귀선과 함께 보여주어 통계적 관계를 파악하는 데 도움을 줍니다.

### regplot() - 회귀 플롯

`regplot()`은 산점도와 회귀선을 결합하여 변수 간의 선형 관계를 보여줍니다.

#### 기본 회귀 플롯

```python
tips = sns.load_dataset("tips")

plt.figure(figsize=(10, 6))
sns.regplot(data=tips, x="total_bill", y="tip")
plt.title("총 청구액과 팁의 회귀 관계")
plt.show()
```

#### 신뢰 구간과 이상치 표시

```python
plt.figure(figsize=(10, 6))
sns.regplot(
    data=tips,
    x="total_bill",
    y="tip",
    ci=95,              # 95% 신뢰 구간
    scatter_kws={'alpha':0.5},  # 산점도 투명도
    line_kws={'color':'red'}     # 회귀선 색상
)
plt.title("신뢰 구간이 포함된 회귀 플롯")
plt.show()
```

#### 다항 회귀

```python
plt.figure(figsize=(10, 6))
sns.regplot(
    data=tips,
    x="total_bill",
    y="tip",
    order=2,            # 2차 다항 회귀
    scatter_kws={'alpha':0.6},
    line_kws={'color':'green'}
)
plt.title("2차 다항 회귀")
plt.show()
```

### lmplot() - 선형 모델 플롯

`lmplot()`은 `regplot()`의 상위 인터페이스로, FacetGrid를 사용하여 다중 플롯을 생성할 수 있습니다.

#### 범주별 회귀 플롯

```python
sns.lmplot(
    data=tips,
    x="total_bill",
    y="tip",
    col="day",          # 열로 요일 구분
    row="time",         # 행으로 시간 구분
    height=4,
    aspect=1.2,
    ci=95
)
plt.suptitle("요일 및 시간별 회귀 관계", y=1.02)
plt.tight_layout()
plt.show()
```

#### 로지스틱 회귀

```python
# 이진 데이터 생성
tips['big_tip'] = (tips['tip'] / tips['total_bill']) > 0.2

sns.lmplot(
    data=tips,
    x="total_bill",
    y="big_tip",
    logistic=True,      # 로지스틱 회귀
    y_jitter=.03        # y축 지터 추가
)
plt.title("로지스틱 회귀: 큰 팁과 총 청구액의 관계")
plt.show()
```

### residplot() - 잔차 플롯

잔차 플롯은 회귀 모델의 잔차를 시각화하여 모델의 적합성을 평가하는 데 사용됩니다.

```python
plt.figure(figsize=(10, 6))
sns.residplot(
    data=tips,
    x="total_bill",
    y="tip",
    lowess=True,        # LOWESS 평활화
    line_kws={'color':'red'}
)
plt.title("회귀 모델 잔차 플롯")
plt.xlabel("총 청구액")
plt.ylabel("잔차")
plt.axhline(y=0, color='gray', linestyle='--')
plt.show()
```

## 다중 플롯 그리드 (Multi-plot Grids)

다중 플롯 그리드는 여러 플롯을 조합하여 복잡한 데이터 관계를 효과적으로 시각화합니다.

### PairGrid - 쌍 그리드

`PairGrid`는 데이터셋의 모든 변수 쌍에 대한 플롯을 생성하는 그리드입니다.

#### 기본 PairGrid

```python
iris = sns.load_dataset("iris")

g = sns.PairGrid(iris, hue="species")
g.map_upper(sns.scatterplot)    # 위쪽 삼각형: 산점도
g.map_diag(sns.histplot)        # 대각선: 히스토그램
g.map_lower(sns.kdeplot)        # 아래쪽 삼각형: 밀도 플롯
g.add_legend()
plt.suptitle("Iris 데이터 PairGrid", y=1.02)
plt.show()
```

#### 사용자 정의 PairGrid

```python
g = sns.PairGrid(iris, hue="species", 
                 vars=["sepal_length", "sepal_width", "petal_length"])
g.map_upper(sns.regplot, scatter_kws={'alpha':0.3})
g.map_diag(sns.histplot, kde=True)
g.map_lower(sns.kdeplot, levels=4, color=".2")
g.add_legend()
plt.suptitle("선택적 변수로 구성된 PairGrid", y=1.02)
plt.show()
```

### FacetGrid - 퍼셋 그리드

`FacetGrid`는 데이터의 하위 집합에 대한 플롯을 생성하는 그리드입니다.

#### 기본 FacetGrid

```python
tips = sns.load_dataset("tips")

g = sns.FacetGrid(tips, col="day", row="time", height=4, aspect=1.2)
g.map(sns.histplot, "total_bill", bins=15)
g.add_legend()
plt.suptitle("요일 및 시간별 청구액 분포", y=1.02)
plt.tight_layout()
plt.show()
```

#### 복합 플롯 FacetGrid

```python
g = sns.FacetGrid(tips, col="day", height=4, aspect=1.2)
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.map(sns.regplot, "total_bill", "tip", scatter=False, color='red')
g.add_legend()
plt.suptitle("요일별 청구액과 팁의 관계", y=1.02)
plt.tight_layout()
plt.show()
```

### JointGrid - 결합 그리드

`JointGrid`는 두 변수의 관계와 각 변수의 분포를 함께 보여줍니다.

#### 기본 JointGrid

```python
g = sns.JointGrid(data=tips, x="total_bill", y="tip")
g.plot(sns.scatterplot, sns.histplot)
plt.suptitle("총 청구액과 팁의 관계 및 분포", y=1.02)
plt.show()
```

#### 고급 JointGrid

```python
g = sns.JointGrid(data=tips, x="total_bill", y="tip", height=8)
g.plot_joint(sns.regplot, scatter_kws={'alpha':0.6})
g.plot_marginals(sns.histplot, kde=True, bins=20)
g.ax_joint.set_xlabel("총 청구액 ($)")
g.ax_joint.set_ylabel("팁 ($)")
plt.suptitle("고급 JointGrid 예제", y=1.02)
plt.show()
```

## 고급 시각화 기법

### 1. 다중 축 플롯

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

# 첫 번째 y축
sns.lineplot(data=flights, x="year", y="passengers", ax=ax1, color="blue")
ax1.set_ylabel("승객 수", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

# 두 번째 y축
ax2 = ax1.twinx()
sns.lineplot(data=flights, x="year", y="passengers", ax=ax2, color="red", 
             estimator='std', errorbar=None)
ax2.set_ylabel("표준편차", color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title("연도별 승객 수와 표준편차")
plt.show()
```

### 2. 애니메이션 플롯

```python
# 시간에 따른 변화를 보여주는 애니메이션 (概念 예시)
# 실제 구현에는 matplotlib.animation 필요
from matplotlib.animation import FuncAnimation

# 애니메이션을 위한 데이터 준비
years = flights['year'].unique()
fig, ax = plt.subplots(figsize=(10, 6))

def update(year):
    ax.clear()
    year_data = flights[flights['year'] == year]
    sns.barplot(data=year_data, x="month", y="passengers", ax=ax)
    ax.set_title(f"{year}년 월별 승객 수")
    ax.set_ylim(0, 600)

# 애니메이션 생성 (실제 실행 환경에서만 작동)
# anim = FuncAnimation(fig, update, frames=years, interval=1000)
# plt.show()
```

## 모범 사례

### 1. 적절한 고급 플롯 선택

| 데이터 특성 | 목적 | 추천 플롯 |
|-------------|------|-----------|
| 상관관계 행렬 | 변수 간 상관관계 파악 | `heatmap()` |
| 계층적 구조 | 데이터 클러스터링 | `clustermap()` |
| 선형 관계 | 회귀 분석 결과 | `regplot()`, `lmplot()` |
| 다변량 관계 | 모든 변수 쌍 관계 | `PairGrid` |
| 하위 집합 비교 | 그룹별 데이터 비교 | `FacetGrid` |

### 2. 성능 최적화

```python
# 대용량 데이터의 경우 샘플링 사용
sample_data = data.sample(n=1000, random_state=42)

# 플롯 크기 조정으로 메모리 사용 최적화
plt.figure(figsize=(8, 6))  # 적절한 크기

# 복잡한 플롯은 단계적으로 생성
g = sns.FacetGrid(data, col="category", col_wrap=3)
g.map(sns.scatterplot, "x", "y")
```

### 3. 상호작용성 추가

```python
# Plotly와 연동하여 상호작용성 추가 (概念 예시)
import plotly.express as px

fig = px.scatter(data, x="total_bill", y="tip", color="day", 
                 size="party_size", hover_data=["time"])
fig.show()
```

## 다음 단계

고급 플롯 유형을 익혔다면, [스타일링 및 사용자 정의](05-styling.md) 문서에서 플롯의 미적 요소를 개선하는 방법을 학습해보세요.

## 추가 자료

- [Seaborn 행렬 플롯 문서](https://seaborn.pydata.org/tutorial/axis_grids.html#matrix-plots)
- [Seaborn 회귀 플롯 문서](https://seaborn.pydata.org/tutorial/regression.html)
- [Seaborn 다중 플롯 그리드 문서](https://seaborn.pydata.org/tutorial/axis_grids.html)