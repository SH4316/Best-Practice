# Seaborn 스타일링 및 사용자 정의

Seaborn은 전문적인 모양의 기본 스타일을 제공하면서도, 세밀한 사용자 정의가 가능합니다. 이 문서에서는 Seaborn 플롯의 미적 요소를 개선하는 다양한 방법을 다룹니다.

## 테마 설정

Seaborn은 다섯 가지 기본 테마를 제공하며, 전역 또는 로컬로 적용할 수 있습니다.

### 기본 테마

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 샘플 데이터
tips = sns.load_dataset("tips")

# 사용 가능한 테마
themes = ["darkgrid", "whitegrid", "dark", "white", "ticks"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, theme in enumerate(themes):
    with sns.axes_style(theme):
        sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[i])
        axes[i].set_title(f'theme="{theme}"')

# 마지막 축은 숨기기
axes[-1].set_visible(False)
plt.tight_layout()
plt.show()
```

### 전역 테마 설정

```python
# 전역 테마 설정
sns.set_theme(style="whitegrid", palette="muted")

# 모든 후속 플롯에 적용
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("전역 테마가 적용된 박스 플롯")
plt.show()

# 기본 설정으로 복원
sns.reset_defaults()
```

### 임시 테마 적용

```python
# 특정 플롯에만 테마 적용
with sns.axes_style("darkgrid"):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=tips, x="day", y="total_bill", hue="time")
    plt.title("임시 테마가 적용된 바이올린 플롯")
plt.show()

# 원래 테마로 복귀
plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x="day", y="total_bill", hue="time")
plt.title("원래 테마로 복귀")
plt.show()
```

## 컨텍스트 설정

컨텍스트는 플롯의 크기와 요소들의 스케일을 조정하여 다른 환경에 최적화합니다.

### 컨텍스트 유형

```python
contexts = ["paper", "notebook", "talk", "poster"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, context in enumerate(contexts):
    sns.set_context(context)
    sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[i])
    axes[i].set_title(f'context="{context}"')

plt.tight_layout()
plt.show()

# 기본 컨텍스트로 복원
sns.set_context("notebook")
```

### 사용자 정의 컨텍스트

```python
# 사용자 정의 컨텍스트 생성
sns.set_context("notebook", 
                font_scale=1.5,      # 폰트 크기 배율
                rc={"lines.linewidth": 2.5})  # 라인 두께

plt.figure(figsize=(10, 6))
sns.lineplot(data=tips, x="total_bill", y="tip")
plt.title("사용자 정의 컨텍스트 적용")
plt.show()
```

## 스타일 요소 직접 제어

### rcParams를 통한 세밀한 제어

```python
# matplotlib의 rcParams를 직접 수정
custom_params = {
    "axes.spines.right": False,      # 오른쪽 축 숨기기
    "axes.spines.top": False,        # 위쪽 축 숨기기
    "axes.grid": True,               # 그리드 표시
    "grid.color": ".8",              # 그리드 색상
    "grid.linestyle": "--",          # 그리드 스타일
    "font.family": "serif",          # 폰트 패밀리
    "font.serif": ["Times New Roman"],  # 세리프 폰트
    "axes.titlesize": 16,            # 제목 크기
    "axes.labelsize": 12,            # 레이블 크기
    "xtick.labelsize": 10,           # x축 눈금 레이블 크기
    "ytick.labelsize": 10,           # y축 눈금 레이블 크기
    "legend.fontsize": 11,           # 범례 폰트 크기
    "figure.titlesize": 18           # 그림 제목 크기
}

sns.set_theme(style="white", rc=custom_params)

plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time")
plt.title("사용자 정의 스타일 매개변수")
plt.show()
```

### 개별 플롯 요소 제어

```python
plt.figure(figsize=(10, 6))

# 기본 플롯 생성
ax = sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")

# 축 스타일 직접 제어
ax.spines['left'].set_color('blue')      # 왼쪽 축 색상
ax.spines['left'].set_linewidth(2)       # 왼쪽 축 두께
ax.spines['bottom'].set_color('red')     # 아래쪽 축 색상
ax.spines['bottom'].set_linewidth(2)     # 아래쪽 축 두께

# 그리드 제어
ax.grid(True, linestyle=':', alpha=0.5)  # 점선 그리드

# 제목과 레이블 스타일
ax.set_title("개별 요소 스타일링", fontsize=16, fontweight='bold')
ax.set_xlabel("총 청구액 ($)", fontsize=12, color='darkblue')
ax.set_ylabel("팁 ($)", fontsize=12, color='darkred')

plt.show()
```

## 플롯 크기 및 레이아웃

### 플롯 크기 조정

```python
# 다양한 크기의 플롯
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 작은 플롯
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[0])
axes[0].set_title("작은 플롯")

# 중간 플롯
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[1])
axes[1].set_title("중간 플롯")

# 큰 플롯
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[2])
axes[2].set_title("큰 플롯")

plt.tight_layout()
plt.show()
```

### 다중 플롯 레이아웃

```python
# 복잡한 레이아웃
fig = plt.figure(figsize=(15, 10))

# 그리드 생성
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 메인 플롯 (상단 전체)
ax_main = fig.add_subplot(gs[0, :])
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax_main)
ax_main.set_title("메인 플롯")

# 보조 플롯들
ax1 = fig.add_subplot(gs[1, 0])
sns.histplot(tips["total_bill"], ax=ax1)
ax1.set_title("청구액 분포")

ax2 = fig.add_subplot(gs[1, 1])
sns.histplot(tips["tip"], ax=ax2)
ax2.set_title("팁 분포")

ax3 = fig.add_subplot(gs[1, 2])
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax3)
ax3.set_title("산점도")

# 하단 전체 플롯
ax_bottom = fig.add_subplot(gs[2, :])
sns.violinplot(data=tips, x="day", y="total_bill", ax=ax_bottom)
ax_bottom.set_title("바이올린 플롯")

plt.suptitle("복합 레이아웃 예제", fontsize=16, y=0.98)
plt.show()
```

## 제목과 레이블

### 제목 스타일링

```python
plt.figure(figsize=(10, 6))

# 플롯 생성
ax = sns.boxplot(data=tips, x="day", y="total_bill", hue="time")

# 다양한 제목 스타일
plt.suptitle("상위 제목", fontsize=18, fontweight='bold', y=0.98)
ax.set_title(
    "요일 및 시간별 청구액 분포", 
    fontsize=14, 
    fontweight='normal', 
    style='italic',
    color='darkblue',
    pad=20  # 제목과 플롯 사이 간격
)

# 축 레이블 스타일
ax.set_xlabel("요일", fontsize=12, fontweight='bold')
ax.set_ylabel("청구액 ($)", fontsize=12, fontweight='bold')

# 눈금 레이블 스타일
ax.tick_params(axis='x', labelsize=10, rotation=45)
ax.tick_params(axis='y', labelsize=10)

plt.show()
```

### 주석 추가

```python
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=tips, x="total_bill", y="tip")

# 주석 추가
ax.annotate(
    "가장 큰 팁", 
    xy=(50.81, 10),  # 화살표 끝점
    xytext=(40, 8),  # 텍스트 위치
    arrowprops=dict(arrowstyle="->", color='red'),
    fontsize=12,
    color='red',
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7)
)

ax.annotate(
    "평균 이하", 
    xy=(10, 1.5),
    xytext=(25, 3),
    arrowprops=dict(arrowstyle="->", color='blue'),
    fontsize=12,
    color='blue',
    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=1, alpha=0.7)
)

plt.title("주석이 추가된 산점도")
plt.show()
```

## 범례 스타일링

### 범례 위치와 모양

```python
plt.figure(figsize=(10, 6))

# 플롯 생성
ax = sns.scatterplot(
    data=tips, 
    x="total_bill", 
    y="tip", 
    hue="day", 
    style="time",
    size="party_size",
    sizes=(50, 200)
)

# 범례 스타일링
legend = ax.legend(
    title="범례",
    title_fontsize=12,
    fontsize=10,
    bbox_to_anchor=(1.05, 1),  # 플롯 외부 위치
    loc='upper left',
    borderaxespad=0.1,
    frameon=True,              # 테두리 표시
    fancybox=True,             # 둥근 모서리
    shadow=True,               # 그림자
    framealpha=0.9             # 테두리 투명도
)

# 범례 제목 스타일
legend.get_title().set_fontweight('bold')

plt.title("범례 스타일링 예제")
plt.tight_layout()
plt.show()
```

### 범례 항목 사용자 정의

```python
plt.figure(figsize=(10, 6))

# 플롯 생성
ax = sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")

# 범례 항목 이름 변경
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, 
    ['금요일', '토요일', '일요일', '목요일'],  # 순서에 맞게 레이블 변경
    title="요일",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.title("사용자 정의 범례 레이블")
plt.tight_layout()
plt.show()
```

## 스타일 저장 및 재사용

### 스타일 설정 저장

```python
import json

# 스타일 설정 저장 함수
def save_style(name, **kwargs):
    style_dict = {
        'style': kwargs.get('style', 'whitegrid'),
        'palette': kwargs.get('palette', 'deep'),
        'context': kwargs.get('context', 'notebook'),
        'font_scale': kwargs.get('font_scale', 1.0),
        'rc': kwargs.get('rc', {})
    }
    
    with open(f'{name}.json', 'w') as f:
        json.dump(style_dict, f, indent=2)

# 스타일 설정 로드 함수
def load_style(name):
    with open(f'{name}.json', 'r') as f:
        style_dict = json.load(f)
    return style_dict

# 사용자 정의 스타일 저장
my_style = {
    'style': 'whitegrid',
    'palette': 'muted',
    'context': 'notebook',
    'font_scale': 1.2,
    'rc': {
        'axes.spines.right': False,
        'axes.spines.top': False,
        'grid.linestyle': ':',
        'font.family': 'serif'
    }
}

save_style('my_custom_style', **my_style)
```

### 저장된 스타일 적용

```python
# 저장된 스타일 로드 및 적용
loaded_style = load_style('my_custom_style')
sns.set_theme(
    style=loaded_style['style'],
    palette=loaded_style['palette'],
    rc=loaded_style['rc']
)
sns.set_context(
    loaded_style['context'],
    font_scale=loaded_style['font_scale']
)

# 스타일이 적용된 플롯
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time")
plt.title("저장된 스타일이 적용된 플롯")
plt.show()
```

## 모범 사례

### 1. 일관된 스타일 유지

```python
# 프로젝트 전체에 일관된 스타일 적용
def set_project_style():
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        font_scale=1.1,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.3
        }
    )

# 프로젝트 시작 시 스타일 설정
set_project_style()
```

### 2. 플롯 크기 가이드라인

```python
# 목적에 맞는 플롯 크기 가이드
plot_sizes = {
    'single_plot': (10, 6),
    'multi_plot': (15, 10),
    'presentation': (12, 8),
    'publication': (8, 6),
    'wide_plot': (15, 5),
    'tall_plot': (6, 10)
}

# 사용 예시
plt.figure(figsize=plot_sizes['single_plot'])
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()
```

### 3. 색맹 접근성 고려

```python
# 색맹 친화적 팔레트 사용
colorblind_palette = sns.color_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time", palette=colorblind_palette)
plt.title("색맹 친화적 팔레트")
plt.show()
```

## 다음 단계

스타일링 및 사용자 정의를 익혔다면, [색상 팔레트 및 테마](06-colors.md) 문서에서 색상 활용에 대한 고급 기법을 학습해보세요.

## 추가 자료

- [Seaborn 스타일링 문서](https://seaborn.pydata.org/tutorial/aesthetics.html)
- [Matplotlib 사용자 정의 가이드](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
- [색맹 친화적 시각화 가이드](https://cran.r-project.org/web/packages/viridis/vignettes/intro.html)