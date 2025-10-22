# Seaborn 설치 및 설정 가이드

## 시스템 요구사항

Seaborn을 사용하기 위한 최소 요구사항은 다음과 같습니다:

- Python 3.7 이상
- pandas 1.0 이상
- matplotlib 3.0 이상
- numpy 1.15 이상

## 설치 방법

### 1. pip를 사용한 설치 (권장)

```bash
pip install seaborn
```

### 2. conda를 사용한 설치

```bash
conda install -c anaconda seaborn
```

### 3. 개발 버전 설치

최신 기능을 사용하려면 개발 버전을 설치할 수 있습니다:

```bash
pip install git+https://github.com/mwaskom/seaborn.git
```

### 4. 프로젝트 의존성 설치

이 저장소의 모든 예제를 실행하려면:

```bash
pip install -r requirements.txt
```

## 설치 확인

설치가 올바르게 되었는지 확인하려면:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 버전 확인
print(f"Seaborn version: {sns.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

# 샘플 플롯으로 테스트
sns.set_theme()
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("설치 테스트 플롯")
plt.show()
```

## 개발 환경 설정

### Jupyter Notebook 설정

Jupyter Notebook에서 Seaborn을 사용하려면:

```bash
# Jupyter 설치
pip install jupyter

# 노트북 시작
jupyter notebook
```

Jupyter Notebook에서 Seaborn 플롯이 잘 보이도록 설정:

```python
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

# 노트북에서 플롯 크기 설정
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_theme(style="whitegrid")
```

### VS Code 설정

VS Code에서 Python 데이터 시각화 환경을 설정하려면:

1. Python 확장 프로그램 설치
2. Jupyter 확장 프로그램 설치
3. 설정에서 플롯 형식 지정:

```json
{
    "python.terminal.activateEnvironment": true,
    "jupyter.askForKernelRestart": false,
    "python.defaultInterpreterPath": "your_python_path"
}
```

## 기본 설정

### 테마 설정

Seaborn은 다양한 기본 테마를 제공합니다:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 사용 가능한 테마
themes = ["darkgrid", "whitegrid", "dark", "white", "ticks"]

# 테마 적용 예시
sns.set_theme(style="whitegrid")

# 임시로 테마 변경
with sns.axes_style("dark"):
    plt.subplot(121)
    sns.scatterplot(data=tips, x="total_bill", y="tip")

plt.subplot(122)
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()
```

### 컨텍스트 설정

플롯의 크기와 요소들을 컨텍스트에 따라 조정할 수 있습니다:

```python
# 컨텍스트 설정 (paper, notebook, talk, poster)
sns.set_context("talk")  # 발표용 큰 플롯
sns.set_context("paper")  # 논문용 작은 플롯

# 폰트 크기 조정
sns.set_context("notebook", font_scale=1.2)
```

### 색상 팔레트 설정

```python
# 기본 색상 팔레트 설정
sns.set_palette("viridis")

# 사용자 정의 색상 팔레트
custom_palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
sns.set_palette(custom_palette)
```

## 고급 설정

### rcParams를 사용한 전역 설정

matplotlib의 rcParams를 직접 수정하여 세밀한 설정이 가능합니다:

```python
import matplotlib.pyplot as plt

# 전역 설정
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})
```

### 스타일 시트 저장 및 로드

자주 사용하는 스타일을 저장하고 재사용할 수 있습니다:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import json

# 스타일 설정 저장
def save_style(style_name, **kwargs):
    with open(f"{style_name}.json", "w") as f:
        json.dump(kwargs, f)

# 스타일 설정 로드
def load_style(style_name):
    with open(f"{style_name}.json", "r") as f:
        style_dict = json.load(f)
    return style_dict

# 사용 예시
my_style = {
    "style": "whitegrid",
    "palette": "deep",
    "context": "notebook",
    "font_scale": 1.2
}

save_style("my_custom_style", **my_style)
loaded_style = load_style("my_custom_style")
sns.set_theme(**loaded_style)
```

## 문제 해결

### 일반적인 설치 문제

#### 1. 버전 호환성 오류

```
ERROR: seaborn 0.11.0 has requirement matplotlib>=3.0, but you have matplotlib 2.2.3
```

**해결책:**
```bash
pip install --upgrade matplotlib
```

#### 2. 권한 오류

```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**해결책:**
```bash
pip install --user seaborn
# 또는
sudo pip install seaborn
```

#### 3. 가상환경 문제

가상환경을 사용하여 시스템 전체와 격리된 환경에서 작업하는 것이 좋습니다:

```bash
# 가상환경 생성
python -m venv seaborn_env

# 가상환경 활성화
# Windows
seaborn_env\Scripts\activate
# macOS/Linux
source seaborn_env/bin/activate

# Seaborn 설치
pip install seaborn
```

### 폰트 관련 문제

한글 폰트가 제대로 표시되지 않을 때:

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 사용 가능한 한글 폰트 확인
for font in fm.fontManager.ttflist:
    if 'Malgun' in font.name or 'Nanum' in font.name:
        print(font.name, font.fname)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
```

### Jupyter Notebook에서 플롯이 보이지 않을 때

```python
# 인라인 플롯 활성화
%matplotlib inline

# 또는
%matplotlib notebook

# 필요한 경우 백엔드 설정
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드
```

## 다음 단계

설치가 완료되었다면, [기본 플롯 유형](03-basic-plots.md) 문서에서 Seaborn의 다양한 플롯 기능을 살펴보세요.

## 추가 자료

- [Python 가상환경 가이드](https://docs.python.org/3/library/venv.html)
- [Jupyter 설치 가이드](https://jupyter.org/install)
- [Matplotlib 설정 가이드](https://matplotlib.org/stable/tutorials/introductory/customizing.html)