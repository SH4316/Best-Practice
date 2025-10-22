# Seaborn 색상 팔레트 및 테마

색상은 데이터 시각화에서 정보를 효과적으로 전달하는 중요한 요소입니다. Seaborn은 다양한 색상 팔레트와 테마를 제공하여 데이터의 특성과 목적에 맞는 색상을 선택할 수 있도록 도와줍니다.

## 색상 팔레트 유형

Seaborn은 세 가지 주요 색상 팔레트 유형을 제공합니다:

1. **순차적 팔레트 (Sequential)**: 낮은 값에서 높은 값으로의 순서를 표현
2. **발산적 팔레트 (Diverging)**: 중심값을 기준으로 양방향으로 발산
3. **범주형 팔레트 (Categorical)**: 구별되는 범주를 표현

## 순차적 팔레트 (Sequential Palettes)

순차적 팔레트는 낮은 값에서 높은 값으로의 순서를 표현할 때 사용합니다.

### 기본 순차적 팔레트

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 기본 순차적 팔레트 시각화
sequential_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

fig, axes = plt.subplots(1, len(sequential_palettes), figsize=(20, 3))

for i, palette in enumerate(sequential_palettes):
    sns.palplot(sns.color_palette(palette, 10), ax=axes[i])
    axes[i].set_title(f'"{palette}"')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

### 사용자 정의 순차적 팔레트

```python
# 단일 색상 기반 순차적 팔레트
single_color_palettes = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']

fig, axes = plt.subplots(1, len(single_color_palettes), figsize=(20, 3))

for i, palette in enumerate(single_color_palettes):
    sns.palplot(sns.color_palette(palette, 10), ax=axes[i])
    axes[i].set_title(f'"{palette}"')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

### 순차적 팔레트 적용 예제

```python
# 데이터 생성
data = np.random.randn(10, 12)
data = np.cumsum(data, axis=0)

# 순차적 팔레트 적용
plt.figure(figsize=(12, 8))
sns.heatmap(data, cmap="viridis", annot=True, fmt=".1f")
plt.title("순차적 팔레트 히트맵")
plt.show()
```

## 발산적 팔레트 (Diverging Palettes)

발산적 팔레트는 중심값을 기준으로 양방향으로 발산하는 데이터를 표현할 때 사용합니다.

### 기본 발산적 팔레트

```python
# 기본 발산적 팔레트
diverging_palettes = ['RdBu', 'RdGy', 'PRGn', 'PiYG', 'BrBG', 'PuOr']

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, palette in enumerate(diverging_palettes):
    sns.palplot(sns.color_palette(palette, 10), ax=axes[i])
    axes[i].set_title(f'"{palette}"')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

### 발산적 팔레트 적용 예제

```python
# 상관관계 데이터 생성
corr = np.corrcoef(np.random.randn(10, 15))

# 발산적 팔레트 적용
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr, 
    cmap="RdBu_r",  # _r은 색상 순서 반전
    center=0,       # 중심값
    annot=True, 
    fmt=".2f",
    vmin=-1, vmax=1
)
plt.title("발산적 팔레트 상관관계 히트맵")
plt.show()
```

## 범주형 팔레트 (Categorical Palettes)

범주형 팔레트는 구별되는 범주를 표현할 때 사용합니다.

### 기본 범주형 팔레트

```python
# 기본 범주형 팔레트
categorical_palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

fig, axes = plt.subplots(2, 3, figsize=(18, 6))
axes = axes.flatten()

for i, palette in enumerate(categorical_palettes):
    sns.palplot(sns.color_palette(palette, 8), ax=axes[i])
    axes[i].set_title(f'"{palette}"')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

### 범주형 팔레트 적용 예제

```python
# 샘플 데이터
tips = sns.load_dataset("tips")

# 범주형 팔레트 적용
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time", palette="muted")
plt.title("범주형 팔레트 박스 플롯")
plt.show()
```

## 사용자 정의 색상 팔레트

### 색상 직접 지정

```python
# 사용자 정의 색상 목록
custom_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

plt.figure(figsize=(10, 6))
sns.countplot(data=tips, x="day", palette=custom_colors)
plt.title("사용자 정의 색상 팔레트")
plt.show()
```

### HLS 색상 공간 사용

```python
# HLS 색상 공간에서 팔레트 생성
hls_palette = sns.hls_palette(8, h=0.5, l=0.6, s=0.8)

plt.figure(figsize=(12, 3))
sns.palplot(hls_palette)
plt.title("HLS 색상 팔레트")
plt.xticks(range(8), [f"색상 {i+1}" for i in range(8)])
plt.show()

# HLS 팔레트 적용
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=tips, 
    x="total_bill", 
    y="tip", 
    hue="day", 
    palette=hls_palette[:4]  # 처음 4개 색상만 사용
)
plt.title("HLS 팔레트 적용 산점도")
plt.show()
```

### Cubehelix 팔레트

```python
# Cubehelix 팔레트 생성
cubehelix_palettes = [
    sns.cubehelix_palette(8, start=0.5, rot=-0.75),
    sns.cubehelix_palette(8, start=2, rot=0.5, dark=0.2, light=0.9),
    sns.cubehelix_palette(8, start=0.8, rot=1.5, reverse=True)
]

fig, axes = plt.subplots(1, 3, figsize=(18, 3))
titles = ["기본 Cubehelix", "밝게 조정", "반전된 Cubehelix"]

for i, palette in enumerate(cubehelix_palettes):
    sns.palplot(palette, ax=axes[i])
    axes[i].set_title(titles[i])
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

## 색상 팔레트 고급 기법

### 색상 팔레트 동적 선택

```python
# 데이터 값에 따라 동적 색상 선택
def dynamic_color_palette(data, palette_type="sequential"):
    """
    데이터 특성에 따라 적절한 색상 팔레트 선택
    """
    if palette_type == "sequential":
        if data.min() >= 0:
            return "viridis"
        else:
            return "RdBu_r"
    elif palette_type == "categorical":
        n_categories = len(data.unique())
        if n_categories <= 6:
            return "deep"
        elif n_categories <= 10:
            return "tab10"
        else:
            return "husl"
    return "default"

# 동적 팔레트 적용
plt.figure(figsize=(10, 6))
palette = dynamic_color_palette(tips["day"], "categorical")
sns.countplot(data=tips, x="day", palette=palette)
plt.title(f"동적 선택된 팔레트: {palette}")
plt.show()
```

### 색상 팔레트 상호작용

```python
# 색상 팔레트 상호작용 도구
def interactive_palette_viewer(palette_name, n_colors=10):
    """
    색상 팔레트를 상호작용적으로 탐색
    """
    try:
        palette = sns.color_palette(palette_name, n_colors)
        
        plt.figure(figsize=(12, 4))
        
        # 팔레트 표시
        plt.subplot(1, 2, 1)
        sns.palplot(palette)
        plt.title(f'"{palette_name}" 팔레트')
        plt.xticks(range(n_colors), [f"{i+1}" for i in range(n_colors)])
        
        # 색상 정보 표시
        plt.subplot(1, 2, 2)
        for i, color in enumerate(palette):
            plt.text(0.1, 0.9 - i*0.1, f"색상 {i+1}: {color}", 
                     transform=plt.gca().transAxes, fontsize=10)
            plt.axhspan(0.9 - i*0.1 - 0.05, 0.9 - i*0.1, 
                        xmin=0.5, xmax=0.8, color=color, transform=plt.gca().transAxes)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("색상 정보")
        
        plt.tight_layout()
        plt.show()
        
    except ValueError as e:
        print(f"오류: {e}")

# 예제 실행
interactive_palette_viewer("viridis", 8)
```

## 색맹 접근성

### 색맹 친화적 팔레트

```python
# 색맹 친화적 팔레트
colorblind_palettes = ['colorblind', 'viridis', 'plasma', 'cividis']

fig, axes = plt.subplots(1, len(colorblind_palettes), figsize=(20, 3))

for i, palette in enumerate(colorblind_palettes):
    sns.palplot(sns.color_palette(palette, 8), ax=axes[i])
    axes[i].set_title(f'색맹 친화적: "{palette}"')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()
```

### 색맹 시뮬레이션

```python
# 색맹 시뮬레이션 함수 (개념 예시)
def simulate_colorblind_vision(palette, blindness_type="deuteranopia"):
    """
    색맹 시뮬레이션 (개념적 예시)
    실제 구현에는 daltonlens 라이브러리 등 필요
    """
    if blindness_type == "deuteranopia":
        # 녹색맹 시뮬레이션 (단순화된 예시)
        simulated_palette = []
        for color in palette:
            r, g, b = color
            # 녹색 채널 감소
            simulated_color = (r, g * 0.5, b)
            simulated_palette.append(simulated_color)
        return simulated_palette
    return palette

# 색맹 시뮬레이션 예시
original_palette = sns.color_palette("Set1", 6)
simulated_palette = simulate_colorblind_vision(original_palette)

fig, axes = plt.subplots(2, 1, figsize=(12, 4))

sns.palplot(original_palette, ax=axes[0])
axes[0].set_title("원래 색상")
axes[0].set_xticks([])
axes[0].set_yticks([])

sns.palplot(simulated_palette, ax=axes[1])
axes[1].set_title("녹색맹 시뮬레이션")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()
```

## 모범 사례

### 1. 데이터 유형에 맞는 팔레트 선택

```python
# 데이터 유형별 팔레트 선택 가이드
palette_guide = {
    "순차적 데이터": {
        "양수만": "viridis, plasma, inferno",
        "비율 데이터": "Blues, Reds, Greens",
        "온도 데이터": "coolwarm, RdBu_r"
    },
    "발산적 데이터": {
        "상관관계": "RdBu_r, PRGn, BrBG",
        "편차 데이터": "RdGy, PuOr, PiYG",
        "중심값 중요": "center 매개변수 사용"
    },
    "범주형 데이터": {
        "6개 이하": "deep, muted, pastel",
        "6-10개": "Set1, Set2, tab10",
        "10개 이상": "husl, Paired"
    }
}

# 가이드 출력
for data_type, palettes in palette_guide.items():
    print(f"\n{data_type}:")
    for subtype, recommendation in palettes.items():
        print(f"  {subtype}: {recommendation}")
```

### 2. 색상 일관성 유지

```python
# 프로젝트 전체 색상 일관성
project_colors = {
    "primary": "#4C72B0",
    "secondary": "#55A868", 
    "accent": "#C44E52",
    "warning": "#CCB974",
    "info": "#64B5CD",
    "categorical": sns.color_palette("deep", 8)
}

# 프로젝트 색상 적용 함수
def apply_project_colors():
    sns.set_palette(project_colors["categorical"])
    return project_colors

# 사용 예제
colors = apply_project_colors()

plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill", hue="time")
plt.title("프로젝트 색상 일관성")
plt.show()
```

### 3. 색상 대비 최적화

```python
# 색상 대비 확인 함수
def check_color_contrast(color1, color2):
    """
    두 색상 간의 대비 확인 (개념적 예시)
    실제 구현에는 colorsys 라이브러리 등 필요
    """
    # 간단한 밝기 차이 계산
    def brightness(rgb):
        return 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    
    return abs(brightness(color1) - brightness(color2))

# 대비가 좋은 색상 조합
good_contrast_pairs = [
    ("#000000", "#FFFFFF"),  # 검정-흰색
    ("#4C72B0", "#FFFFFF"),  # 파랑-흰색
    ("#55A868", "#FFFFFF"),  # 초록-흰색
]

print("색상 대비 확인:")
for color1, color2 in good_contrast_pairs:
    rgb1 = sns.color_palette([color1])[0]
    rgb2 = sns.color_palette([color2])[0]
    contrast = check_color_contrast(rgb1, rgb2)
    print(f"{color1} vs {color2}: 대비값 {contrast:.2f}")
```

## 다음 단계

색상 팔레트 및 테마를 익혔다면, [플롯 구성 및 레이아웃](07-composition.md) 문서에서 복잡한 시각화를 효과적으로 구성하는 방법을 학습해보세요.

## 추가 자료

- [Seaborn 색상 팔레트 문서](https://seaborn.pydata.org/tutorial/color_palettes.html)
- [Matplotlib 색상 맵](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
- [색맹 친화적 시각화 가이드](https://www.colorblindawareness.org/color-blind-friendly-palettes/)
- [색상 이론 및 접근성](https://webaim.org/techniques/colour/)