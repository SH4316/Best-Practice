# Seaborn 기본 플롯 유형

Seaborn은 데이터의 관계와 분포를 시각화하는 데 특화된 다양한 기본 플롯 유형을 제공합니다. 이 문서에서는 세 가지 주요 카테고리의 기본 플롯을 다룹니다:

1. **관계형 플롯 (Relational Plots)**: 변수 간의 관계를 보여줍니다
2. **범주형 플롯 (Categorical Plots)**: 범주형 데이터의 분포를 보여줍니다
3. **분포 플롯 (Distribution Plots)**: 데이터의 분포를 보여줍니다

## 관계형 플롯 (Relational Plots)

관계형 플롯은 두 변수 간의 관계를 시각화하는 데 사용됩니다.

### scatterplot() - 산점도

산점도는 두 연속형 변수 간의 관계를 보여주는 가장 기본적인 플롯입니다.

#### 기본 사용법

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 샘플 데이터 로드
tips = sns.load_dataset("tips")

# 기본 산점도
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("총 청구액과 팁의 관계")
plt.xlabel("총 청구액 ($)")
plt.ylabel("팁 ($)")
plt.show()
```

#### 범주별 색상 구분

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.title("요일별 총 청구액과 팁의 관계")
plt.show()
```

#### 크기와 스타일로 추가 변수 표현

```python
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=tips, 
    x="total_bill", 
    y="tip", 
    hue="day",           # 색상으로 요일 구분
    size="party_size",   # 크기로 파티 크기 구분
    style="time",        # 스타일로 시간 구분
    sizes=(20, 200),     # 크기 범위 설정
    alpha=0.7            # 투명도 설정
)
plt.title("다차원 산점도")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### lineplot() - 선 그래프

선 그래프는 시간에 따른 변화나 연속적인 변수의 추세를 보여줄 때 유용합니다.

#### 기본 사용법

```python
# 시계열 데이터 생성
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'value': np.cumsum(np.random.randn(100)) + 100,
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='value')
plt.title("시간에 따른 값의 변화")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### 신뢰 구간과 범주별 선 그래프

```python
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df, 
    x='date', 
    y='value', 
    hue='category',
    ci='sd',              # 표준편차 신뢰 구간
    style='category',     # 선 스타일 구분
    markers=True,         # 마커 표시
    dashes=False          # 실선으로 표시
)
plt.title("카테고리별 시계열 추세")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### relplot() - 관계형 플롯 인터페이스

`relplot()`은 `scatterplot()`과 `lineplot()`의 상위 인터페이스로, FacetGrid를 사용하여 다중 플롯을 쉽게 생성할 수 있습니다.

#### 다중 플롯 생성

```python
# relplot으로 다중 플롯 생성
sns.relplot(
    data=tips,
    x="total_bill",
    y="tip",
    col="day",        # 열로 요일 구분
    row="time",       # 행으로 시간 구분
    hue="smoker",     # 색상으로 흡연 여부 구분
    height=4,         # 플롯 높이
    aspect=1.2        # 플롯 종횡비
)
plt.suptitle("요일 및 시간별 팁과 청구액의 관계", y=1.02)
plt.tight_layout()
plt.show()
```

## 범주형 플롯 (Categorical Plots)

범주형 플롯은 범주형 데이터의 분포와 통계적 특성을 보여줍니다.

### catplot() - 범주형 플롯 인터페이스

`catplot()`은 다양한 범주형 플롯을 생성할 수 있는 상위 인터페이스입니다.

#### 기본 카테고리 플롯

```python
# kind 매개변수로 플롯 유형 지정
sns.catplot(
    data=tips, 
    x="day", 
    y="total_bill", 
    kind="strip"      # 기본값: strip (점으로 분포 표시)
)
plt.title("요일별 총 청구액 분포")
plt.show()
```

### boxplot() - 박스 플롯

박스 플롯은 데이터의 분포와 통계적 요약(사분위수, 중앙값, 이상치)을 보여줍니다.

#### 기본 박스 플롯

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("요일별 총 청구액 분포")
plt.show()
```

#### 범주별 박스 플롯

```python
plt.figure(figsize=(12, 7))
sns.boxplot(
    data=tips,
    x="day",
    y="total_bill",
    hue="smoker",
    palette="Set2",
    width=0.8
)
plt.title("요일 및 흡연 여부별 총 청구액 분포")
plt.legend(title="흡연 여부")
plt.show()
```

### violinplot() - 바이올린 플롯

바이올린 플롯은 박스 플롯과 커널 밀도 추정을 결합하여 데이터의 분포를 더 자세히 보여줍니다.

#### 기본 바이올린 플롯

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x="day", y="total_bill")
plt.title("요일별 총 청구액 분포")
plt.show()
```

#### 분할 바이올린 플롯

```python
plt.figure(figsize=(12, 7))
sns.violinplot(
    data=tips,
    x="day",
    y="total_bill",
    hue="smoker",
    split=True,        # 양쪽으로 분할
    inner="quartile",  # 내부에 사분위수 표시
    palette="muted"
)
plt.title("요일 및 흡연 여부별 총 청구액 분포")
plt.legend(title="흡연 여부")
plt.show()
```

### barplot() - 막대 그래프

막대 그래프는 범주별 통계적 요약(기본값: 평균)을 보여줍니다.

#### 기본 막대 그래프

```python
plt.figure(figsize=(10, 6))
sns.barplot(data=tips, x="day", y="total_bill")
plt.title("요일별 평균 청구액")
plt.show()
```

#### 신뢰 구간이 포함된 막대 그래프

```python
plt.figure(figsize=(12, 7))
sns.barplot(
    data=tips,
    x="day",
    y="total_bill",
    hue="time",
    ci=95,           # 95% 신뢰 구간
    palette="deep",
    capsize=0.1      # 신뢰 구간 막대 크기
)
plt.title("요일 및 시간별 평균 청구액")
plt.legend(title="시간")
plt.show()
```

### countplot() - 카운트 플롯

카운트 플롯은 각 범주의 관측치 수를 보여줍니다.

#### 기본 카운트 플롯

```python
plt.figure(figsize=(10, 6))
sns.countplot(data=tips, x="day")
plt.title("요일별 방문자 수")
plt.show()
```

#### 수평 카운트 플롯

```python
plt.figure(figsize=(10, 7))
sns.countplot(data=tips, y="day", hue="time")
plt.title("요일 및 시간별 방문자 수")
plt.legend(title="시간")
plt.show()
```

## 분포 플롯 (Distribution Plots)

분포 플롯은 데이터의 분포와 밀도를 보여줍니다.

### histplot() - 히스토그램

히스토그램은 데이터의 분포를 막대로 보여줍니다.

#### 기본 히스토그램

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x="total_bill", bins=20)
plt.title("총 청구액 분포")
plt.xlabel("총 청구액 ($)")
plt.ylabel("빈도")
plt.show()
```

#### 커널 밀도 추정이 포함된 히스토그램

```python
plt.figure(figsize=(10, 6))
sns.histplot(
    data=tips, 
    x="total_bill", 
    bins=20,
    kde=True,        # 커널 밀도 추정曲线 추가
    stat="density"   # 밀도로 표시
)
plt.title("총 청구액 분포와 밀도")
plt.xlabel("총 청구액 ($)")
plt.show()
```

### kdeplot() - 커널 밀도 추정

커널 밀도 추정은 데이터의 확률 밀도 함수를 부드러운 곡선으로 보여줍니다.

#### 기본 밀도 플롯

```python
plt.figure(figsize=(10, 6))
sns.kdeplot(data=tips, x="total_bill")
plt.title("총 청구액 밀도")
plt.xlabel("총 청구액 ($)")
plt.show()
```

#### 2D 밀도 플롯

```python
plt.figure(figsize=(10, 8))
sns.kdeplot(data=tips, x="total_bill", y="tip", shade=True, cmap="Blues")
plt.title("총 청구액과 팁의 2D 밀도")
plt.xlabel("총 청구액 ($)")
plt.ylabel("팁 ($)")
plt.show()
```

### ecdfplot() - 경험적 누적 분포 함수

ECDF 플롯은 데이터의 누적 분포를 보여줍니다.

```python
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=tips, x="total_bill")
plt.title("총 청구액의 누적 분포")
plt.xlabel("총 청구액 ($)")
plt.ylabel("누적 확률")
plt.grid(True, alpha=0.3)
plt.show()
```

### displot() - 분포 플롯 인터페이스

`displot()`은 다양한 분포 플롯을 생성할 수 있는 상위 인터페이스입니다.

#### 다중 분포 플롯

```python
# 히스토그램과 밀도 플롯 결합
sns.displot(
    data=tips,
    x="total_bill",
    col="day",      # 열로 요일 구분
    kde=True,
    bins=15,
    height=4,
    aspect=1.2
)
plt.suptitle("요일별 총 청구액 분포", y=1.02)
plt.tight_layout()
plt.show()
```

## 모범 사례

### 1. 적절한 플롯 선택

| 데이터 유형 | 목적 | 추천 플롯 |
|-------------|------|-----------|
| 두 연속형 변수의 관계 | 관계 파악 | `scatterplot()`, `lineplot()` |
| 범주별 수치형 데이터 | 분포 비교 | `boxplot()`, `violinplot()` |
| 범주별 빈도 | 빈도 비교 | `countplot()` |
| 데이터 분포 | 분포 형태 파악 | `histplot()`, `kdeplot()` |

### 2. 색상 팔레트 활용

```python
# 정량적 데이터: 순차적 팔레트
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="size", palette="viridis")

# 정성적 데이터: 범주형 팔레트
sns.boxplot(data=tips, x="day", y="total_bill", hue="time", palette="Set2")

# 발산 데이터: 발산 팔레트
sns.heatmap(correlation_matrix, cmap="RdBu_r", center=0)
```

### 3. 플롯 크기와 레이아웃

```python
# 적절한 크기 설정
plt.figure(figsize=(12, 8))

# 다중 플롯 간격 조정
plt.tight_layout()

# 제목과 레이블 추가
plt.title("명확한 제목")
plt.xlabel("x축 레이블")
plt.ylabel("y축 레이블")
```

## 다음 단계

기본 플롯 유형을 익혔다면, [고급 플롯 유형](04-advanced-plots.md) 문서에서 더 복잡한 시각화 기법을 학습해보세요.

## 추가 자료

- [Seaborn 관계형 플롯 문서](https://seaborn.pydata.org/tutorial/relational.html)
- [Seaborn 범주형 플롯 문서](https://seaborn.pydata.org/tutorial/categorical.html)
- [Seaborn 분포 플롯 문서](https://seaborn.pydata.org/tutorial/distributions.html)