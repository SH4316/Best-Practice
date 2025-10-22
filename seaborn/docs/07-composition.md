# Seaborn 플롯 구성 및 레이아웃

효과적인 데이터 시각화는 개별 플롯의 품질뿐만 아니라 여러 플롯을 어떻게 구성하고 배치하는지에 따라 결정됩니다. 이 문서에서는 복잡한 시각화를 효과적으로 구성하는 다양한 방법을 다룹니다.

## 다중 플롯 기본

### matplotlib을 이용한 기본 다중 플롯

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 샘플 데이터
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# 2x2 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 각 서브플롯에 다른 플롯 그리기
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[0, 0])
axes[0, 0].set_title("산점도")

sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 1])
axes[0, 1].set_title("박스 플롯")

sns.histplot(tips["total_bill"], ax=axes[1, 0])
axes[1, 0].set_title("히스토그램")

sns.violinplot(data=tips, x="day", y="tip", ax=axes[1, 1])
axes[1, 1].set_title("바이올린 플롯")

plt.tight_layout()
plt.show()
```

### 불규칙한 레이아웃

```python
# 불규칙한 서브플롯 레이아웃
fig = plt.figure(figsize=(15, 10))

# 그리드 생성
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 상단 전체 플롯
ax1 = fig.add_subplot(gs[0, :])
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax1)
ax1.set_title("요일별 청구액 분포")

# 중간 부분 플롯들
ax2 = fig.add_subplot(gs[1, 0])
sns.histplot(tips["total_bill"], ax=ax2)
ax2.set_title("청구액 히스토그램")

ax3 = fig.add_subplot(gs[1, 1])
sns.histplot(tips["tip"], ax=ax3)
ax3.set_title("팁 히스토그램")

ax4 = fig.add_subplot(gs[1, 2])
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax4)
ax4.set_title("청구액 vs 팁")

# 하단 전체 플롯
ax5 = fig.add_subplot(gs[2, :])
sns.violinplot(data=tips, x="day", y="total_bill", hue="time", ax=ax5)
ax5.set_title("요일 및 시간별 청구액 분포")

plt.suptitle("복합 레이아웃 예제", fontsize=16, y=0.98)
plt.show()
```

## Seaborn Grid 시스템

### FacetGrid를 이용한 조건부 플롯

```python
# FacetGrid를 이용한 다중 플롯
g = sns.FacetGrid(
    tips, 
    col="day",           # 열로 요일 구분
    row="time",          # 행으로 시간 구분
    height=4,            # 각 플롯의 높이
    aspect=1.2,          # 종횡비
    palette="muted"
)

# 각 퍼셋에 플롯 그리기
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)

# 회귀선 추가
g.map(sns.regplot, "total_bill", "tip", scatter=False, color='red')

# 축 레이블 설정
g.set_axis_labels("총 청구액 ($)", "팁 ($)")
g.set_titles(col_template="{col_name}요일", row_template="{row_name}")

plt.suptitle("요일 및 시간별 청구액과 팁의 관계", y=1.02)
plt.tight_layout()
plt.show()
```

### PairGrid를 이용한 다변량 관계 시각화

```python
# PairGrid를 이용한 모든 변수 쌍 관계
g = sns.PairGrid(
    iris, 
    hue="species",
    palette="deep",
    height=2.5,
    aspect=1.2
)

# 다른 위치에 다른 플롯 유형
g.map_upper(sns.scatterplot, alpha=0.7)       # 위쪽: 산점도
g.map_diag(sns.histplot, kde=True)           # 대각선: 히스토그램
g.map_lower(sns.kdeplot, levels=4, alpha=0.8) # 아래쪽: 밀도 플롯

g.add_legend(title="품종")
plt.suptitle("Iris 데이터셋 다변량 관계", y=1.02)
plt.tight_layout()
plt.show()
```

### JointGrid를 이용한 결합 플롯

```python
# JointGrid를 이용한 결합 플롯
g = sns.JointGrid(
    data=tips, 
    x="total_bill", 
    y="tip",
    height=8,
    space=0.2
)

# 메인 플롯
g.plot_joint(sns.scatterplot, alpha=0.6, hue="day", palette="deep")

# 주변 분포 플롯
g.plot_marginals(sns.histplot, kde=True, bins=20)

# 회귀선 추가
g.plot_joint(sns.regplot, scatter=False, color='red')

# 축 레이블
g.ax_joint.set_xlabel("총 청구액 ($)")
g.ax_joint.set_ylabel("팁 ($)")

plt.suptitle("결합 플롯: 청구액과 팁의 관계 및 분포", y=1.02)
plt.show()
```

## 고급 레이아웃 기법

### 복합 시각화 구성

```python
# 복합 시각화 예제
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 1. 메인 히트맵 (상단 2x2)
ax1 = fig.add_subplot(gs[0:2, 0:2])
correlation_data = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(
    correlation_data, 
    annot=True, 
    cmap="RdBu_r", 
    center=0,
    ax=ax1,
    cbar_kws={"shrink": 0.8}
)
ax1.set_title("상관관계 히트맵")

# 2. 시계열 플롯 (상단 우측 2x2)
ax2 = fig.add_subplot(gs[0, 2])
# 시계열 데이터 생성
time_data = tips.groupby('day')['total_bill'].mean().reset_index()
sns.lineplot(data=time_data, x='day', y='total_bill', ax=ax2, marker='o')
ax2.set_title("요일별 평균 청구액")

# 3. 분포 비교 (상단 우측 하단)
ax3 = fig.add_subplot(gs[1, 2])
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax3)
ax3.set_title("요일별 청구액 분포")

# 4. 산점도 (상단 우측 최하단)
ax4 = fig.add_subplot(gs[0, 2])
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax4)
ax4.set_title("청구액 vs 팁")

# 5. 히스토그램 (중간 전체)
ax5 = fig.add_subplot(gs[2, :])
sns.histplot(
    data=tips, 
    x="total_bill", 
    hue="day", 
    multiple="stack",
    bins=20,
    ax=ax5
)
ax5.set_title("요일별 청구액 히스토그램")

plt.suptitle("복합 데이터 시각화 대시보드", fontsize=18, y=0.95)
plt.show()
```

### 인셋 플롯 (Inset Plots)

```python
# 메인 플롯과 인셋 플롯
fig, ax = plt.subplots(figsize=(10, 6))

# 메인 플롯
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax)
ax.set_title("청구액과 팁의 관계")

# 인셋 플롯 위치 설정
from matplotlib.gridspec import GridSpec

gs = GridSpec(4, 4, figure=fig)
ax_inset = fig.add_subplot(gs[1, 3])  # 오른쪽 상단 위치

# 인셋 플롯 내용
sns.histplot(tips["total_bill"], ax=ax_inset, bins=15)
ax_inset.set_title("청구액 분포", fontsize=10)
ax_inset.set_xlabel("", fontsize=8)
ax_inset.set_ylabel("", fontsize=8)
ax_inset.tick_params(labelsize=8)

# 인셋 플롯을 메인 플롯에 연결하는 영역 표시
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# 특정 영역 확대 (개념 예시)
# mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.show()
```

## 플롯 크기 및 비율 최적화

### 다양한 화면 크기에 맞는 플롯

```python
# 화면 크기에 따른 플롯 크기 최적화
def create_responsive_plot(data, plot_type="dashboard"):
    """
    화면 크기에 따른 반응형 플롯 생성
    """
    if plot_type == "dashboard":
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 대시보드 구성
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        ax5 = fig.add_subplot(gs[2, :])
        
        axes = [ax1, ax2, ax3, ax4, ax5]
        
    elif plot_type == "presentation":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("발표용 플롯", fontsize=16)
        axes = axes.flatten()
        
    elif plot_type == "publication":
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle("논문용 플롯", fontsize=12)
        axes = axes.flatten()
        
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    
    return fig, axes

# 사용 예시
fig, axes = create_responsive_plot(tips, "dashboard")

# 각 축에 플롯 그리기
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0])
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[1])
sns.histplot(tips["total_bill"], ax=axes[2])
sns.violinplot(data=tips, x="day", y="tip", ax=axes[3])
sns.countplot(data=tips, x="day", ax=axes[4])

plt.tight_layout()
plt.show()
```

### 종횡비 최적화

```python
# 데이터 특성에 맞는 종횡비 설정
def optimal_aspect_ratio(data, x_col, y_col):
    """
    데이터 분포에 따른 최적 종횡비 계산
    """
    x_range = data[x_col].max() - data[x_col].min()
    y_range = data[y_col].max() - data[y_col].min()
    
    # 황금 비율을 고려한 종횡비
    aspect = (x_range / y_range) * (1 / 1.618)  # 황금 비율
    
    return aspect

# 최적 종횡비 적용
aspect = optimal_aspect_ratio(tips, "total_bill", "tip")

plt.figure(figsize=(10 * aspect, 10))
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title(f"최적 종횡비 (1:{aspect:.2f})")
plt.show()
```

## 상호작용적 레이아웃

### 플롯 간 정보 공유

```python
# 여러 플롯 간의 정보 공유
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 공유 y축으로 비교 용이
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0])
axes[0].set_title("요일별 청구액")

sns.boxplot(data=tips, x="day", y="tip", ax=axes[1])
axes[1].set_title("요일별 팁")

sns.boxplot(data=tips, x="day", y="size", ax=axes[2])
axes[2].set_title("요일별 파티 크기")

plt.suptitle("공유 y축을 이용한 비교 플롯")
plt.tight_layout()
plt.show()
```

### 연결된 축 스케일링

```python
# 연결된 축 스케일링
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 첫 번째 플롯
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax1)
ax1.set_title("원본 스케일")

# 로그 스케일 플롯
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax2)
ax2.set_xscale("log")
ax2.set_title("로그 스케일")

# 동일한 데이터 포인트 강조
highlight_points = tips[(tips['total_bill'] > 40) & (tips['tip'] > 5)]
for idx, row in highlight_points.iterrows():
    ax1.scatter(row['total_bill'], row['tip'], color='red', s=100, alpha=0.7)
    ax2.scatter(row['total_bill'], row['tip'], color='red', s=100, alpha=0.7)

plt.tight_layout()
plt.show()
```

## 모범 사례

### 1. 정보 계층 구조

```python
# 정보의 중요도에 따른 플롯 크기 배분
def hierarchical_layout(data):
    """
    정보 중요도에 따른 계층적 레이아웃
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1차 정보 (가장 중요) - 상단 전체
    ax_primary = fig.add_subplot(gs[0, :])
    sns.boxplot(data=data, x="day", y="total_bill", ax=ax_primary)
    ax_primary.set_title("1차: 요일별 청구액 분포", fontsize=14, fontweight='bold')
    
    # 2차 정보 (중요) - 중간 좌측
    ax_secondary1 = fig.add_subplot(gs[1:3, 0:2])
    sns.scatterplot(data=data, x="total_bill", y="tip", hue="day", ax=ax_secondary1)
    ax_secondary1.set_title("2차: 청구액과 팁의 관계", fontsize=12)
    
    # 2차 정보 (중요) - 중간 우측
    ax_secondary2 = fig.add_subplot(gs[1:3, 2:])
    sns.histplot(data=data, x="total_bill", hue="day", multiple="stack", ax=ax_secondary2)
    ax_secondary2.set_title("2차: 청구액 분포", fontsize=12)
    
    # 3차 정보 (보조) - 하단 전체
    ax_tertiary = fig.add_subplot(gs[3, :])
    sns.countplot(data=data, x="day", ax=ax_tertiary)
    ax_tertiary.set_title("3차: 요일별 방문자 수", fontsize=10)
    
    plt.suptitle("계층적 정보 구조", fontsize=16, y=0.98)
    return fig

fig = hierarchical_layout(tips)
plt.show()
```

### 2. 시선 흐름 고려

```python
# 시선 흐름을 고려한 레이아웃 (Z자형 패턴)
def z_pattern_layout(data):
    """
    시선 흐름(Z자형)을 고려한 레이아웃
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Z자형 시선 흐름:
    # 1. 왼쪽 상단 -> 2. 오른쪽 상단 -> 3. 왼쪽 하단 -> 4. 오른쪽 하단
    
    # 1. 왼쪽 상단 (시작점)
    ax1 = plt.subplot(2, 2, 1)
    sns.boxplot(data=data, x="day", y="total_bill", ax=ax1)
    ax1.set_title("1. 주요 지표")
    
    # 2. 오른쪽 상단
    ax2 = plt.subplot(2, 2, 2)
    sns.scatterplot(data=data, x="total_bill", y="tip", ax=ax2)
    ax2.set_title("2. 관계 분석")
    
    # 3. 왼쪽 하단
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(data=data, x="total_bill", ax=ax3)
    ax3.set_title("3. 분포 확인")
    
    # 4. 오른쪽 하단 (마지막)
    ax4 = plt.subplot(2, 2, 4)
    sns.countplot(data=data, x="day", ax=ax4)
    ax4.set_title("4. 요약 정보")
    
    plt.suptitle("시선 흐름을 고려한 레이아웃", fontsize=16)
    plt.tight_layout()
    return fig

fig = z_pattern_layout(tips)
plt.show()
```

### 3. 밀도와 균형

```python
# 정보 밀도와 시각적 균형
def balanced_layout(data):
    """
    정보 밀도와 시각적 균형을 고려한 레이아웃
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 고밀도 정보 (중앙)
    ax_center = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=data, x="total_bill", y="tip", hue="day", ax=ax_center)
    ax_center.set_title("중심 분석")
    
    # 저밀도 정보 (주변)
    ax_top = fig.add_subplot(gs[0, :])
    sns.boxplot(data=data, x="day", y="total_bill", ax=ax_top)
    ax_top.set_title("상위 요약")
    
    ax_bottom = fig.add_subplot(gs[2, :])
    sns.histplot(data=data, x="total_bill", hue="day", multiple="stack", ax=ax_bottom)
    ax_bottom.set_title("하위 요약")
    
    ax_left = fig.add_subplot(gs[1, 0])
    sns.violinplot(y=data["total_bill"], ax=ax_left)
    ax_left.set_title("좌측 분포")
    
    ax_right = fig.add_subplot(gs[1, 2])
    sns.violinplot(y=data["tip"], ax=ax_right)
    ax_right.set_title("우측 분포")
    
    plt.suptitle("균형 잡힌 정보 밀도", fontsize=16)
    return fig

fig = balanced_layout(tips)
plt.show()
```

## 다음 단계

플롯 구성 및 레이아웃을 익혔다면, [pandas 및 matplotlib 연동](08-integration.md) 문서에서 다른 라이브러리와의 통합 방법을 학습해보세요.

## 추가 자료

- [Matplotlib 레이아웃 가이드](https://matplotlib.org/stable/tutorials/intermediate/gridspec.html)
- [Seaborn Grid 시스템](https://seaborn.pydata.org/tutorial/axis_grids.html)
- [데이터 시각화 디자인 원칙](https://www.storytellingwithdata.com/)
- [정보 시각화 레이아웃 원칙](https:// InteractionDesign.org/)