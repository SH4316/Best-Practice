# Seaborn 성능 최적화

데이터 시각화에서 성능은 대용량 데이터를 다룰 때 특히 중요합니다. 이 문서에서는 Seaborn 플롯의 성능을 최적화하는 다양한 기법을 다룹니다.

## 대용량 데이터 처리

### 데이터 샘플링

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# 대용량 데이터 생성
def generate_large_dataset(n_samples=100000):
    """대용량 샘플 데이터 생성"""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(n_samples),
        'y': np.random.randn(n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'group': np.random.choice(['Group1', 'Group2', 'Group3'], n_samples),
        'value': np.random.exponential(scale=2, size=n_samples)
    })

# 성능 측정 함수
def measure_time(func, *args, **kwargs):
    """함수 실행 시간 측정"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# 대용량 데이터 생성
large_df = generate_large_dataset(100000)
print(f"데이터 크기: {large_df.shape}")

# 샘플링 전후 성능 비교
def plot_full_data(df):
    """전체 데이터 플롯"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', alpha=0.5)
    plt.title(f"전체 데이터 (n={len(df)})")
    return plt.gcf()

def plot_sampled_data(df, sample_size=10000):
    """샘플링된 데이터 플롯"""
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
    else:
        df_sampled = df
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sampled, x='x', y='y', alpha=0.5)
    plt.title(f"샘플링된 데이터 (n={len(df_sampled)})")
    return plt.gcf()

# 성능 비교
full_fig, full_time = measure_time(plot_full_data, large_df)
plt.close(full_fig)

sampled_fig, sampled_time = measure_time(plot_sampled_data, large_df)
plt.close(sampled_fig)

print(f"전체 데이터 플롯 시간: {full_time:.3f}초")
print(f"샘플링된 데이터 플롯 시간: {sampled_time:.3f}초")
print(f"성능 향상: {full_time/sampled_time:.1f}배")

# 시각적 비교
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 전체 데이터
sns.scatterplot(data=large_df.sample(n=10000, random_state=42), x='x', y='y', 
                alpha=0.3, ax=axes[0])
axes[0].set_title("전체 데이터 (샘플 10,000개 표시)")

# 샘플링된 데이터
sampled_df = large_df.sample(n=10000, random_state=42)
sns.scatterplot(data=sampled_df, x='x', y='y', alpha=0.5, ax=axes[1])
axes[1].set_title("샘플링된 데이터 (10,000개)")

plt.tight_layout()
plt.show()
```

### 스마트 샘플링 기법

```python
def stratified_sampling(df, stratify_col, sample_size=10000):
    """층화 샘플링"""
    if len(df) <= sample_size:
        return df
    
    # 각 그룹에서 비례적으로 샘플링
    groups = df[stratify_col].unique()
    sampled_dfs = []
    
    total_size = len(df)
    for group in groups:
        group_data = df[df[stratify_col] == group]
        group_size = len(group_data)
        group_sample_size = int((group_size / total_size) * sample_size)
        
        if group_size <= group_sample_size:
            sampled_dfs.append(group_data)
        else:
            sampled_dfs.append(group_data.sample(n=group_sample_size, random_state=42))
    
    return pd.concat(sampled_dfs)

def density_based_sampling(df, x_col, y_col, sample_size=10000, bins=50):
    """밀도 기반 샘플링"""
    if len(df) <= sample_size:
        return df
    
    # 2D 히스토그램으로 밀도 계산
    hist, xedges, yedges = np.histogram2d(df[x_col], df[y_col], bins=bins)
    
    # 각 데이터 포인트가 속한 bin 찾기
    x_indices = np.digitize(df[x_col], xedges) - 1
    y_indices = np.digitize(df[y_col], yedges) - 1
    
    # 밀도에 반비례하여 샘플링 확률 계산
    densities = hist[x_indices, y_indices]
    max_density = np.max(densities)
    
    # 낮은 밀도 영역에서 더 많이 샘플링
    probabilities = (max_density - densities + 1) / (max_density + 1)
    probabilities = probabilities / probabilities.sum()
    
    sampled_indices = np.random.choice(
        len(df), 
        size=min(sample_size, len(df)), 
        replace=False, 
        p=probabilities
    )
    
    return df.iloc[sampled_indices]

# 샘플링 기법 비교
sample_size = 5000
stratified_df = stratified_sampling(large_df, 'category', sample_size)
density_df = density_based_sampling(large_df, 'x', 'y', sample_size)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 무작위 샘플링
random_df = large_df.sample(n=sample_size, random_state=42)
sns.scatterplot(data=random_df, x='x', y='y', hue='category', alpha=0.6, ax=axes[0])
axes[0].set_title("무작위 샘플링")

# 층화 샘플링
sns.scatterplot(data=stratified_df, x='x', y='y', hue='category', alpha=0.6, ax=axes[1])
axes[1].set_title("층화 샘플링")

# 밀도 기반 샘플링
sns.scatterplot(data=density_df, x='x', y='y', hue='category', alpha=0.6, ax=axes[2])
axes[2].set_title("밀도 기반 샘플링")

plt.tight_layout()
plt.show()
```

## 플롯 최적화

### 효율적인 플롯 유형 선택

```python
# 플롯 유형별 성능 비교
def compare_plot_performance(df, sample_size=10000):
    """다양한 플롯 유형의 성능 비교"""
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    plot_types = {
        'scatterplot': lambda: sns.scatterplot(data=df, x='x', y='y'),
        'histplot': lambda: sns.histplot(df['x']),
        'kdeplot': lambda: sns.kdeplot(data=df, x='x', y='y'),
        'boxplot': lambda: sns.boxplot(data=df, x='category', y='value'),
        'violinplot': lambda: sns.violinplot(data=df, x='category', y='value'),
        'heatmap': lambda: sns.heatmap(df.select_dtypes(include=[np.number]).corr())
    }
    
    performance = {}
    
    for plot_type, plot_func in plot_types.items():
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            start_time = time.time()
            plot_func()
            end_time = time.time()
            plt.close(fig)
            
            performance[plot_type] = end_time - start_time
            print(f"{plot_type}: {performance[plot_type]:.3f}초")
        except Exception as e:
            print(f"{plot_type}: 오류 - {e}")
    
    return performance

# 성능 비교 실행
performance = compare_plot_performance(large_df)

# 성능 순위 정렬
sorted_performance = sorted(performance.items(), key=lambda x: x[1])
print("\n플롯 유형별 성능 순위 (빠른 순):")
for i, (plot_type, exec_time) in enumerate(sorted_performance, 1):
    print(f"{i}. {plot_type}: {exec_time:.3f}초")
```

### 플롯 렌더링 최적화

```python
def optimized_scatterplot(df, x_col, y_col, color_col=None, max_points=10000):
    """최적화된 산점도"""
    # 데이터 크기 확인
    n_points = len(df)
    
    if n_points > max_points:
        # 샘플링 또는 집계 선택
        if color_col and df[color_col].nunique() < 20:
            # 카테고리별로 샘플링
            df = stratified_sampling(df, color_col, max_points)
        else:
            # 무작위 샘플링
            df = df.sample(n=max_points, random_state=42)
    
    # 플롯 생성
    plt.figure(figsize=(10, 8))
    
    if color_col and df[color_col].nunique() < 10:
        # 카테고리가 적을 경우 색상 구분
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, 
                       alpha=0.6, s=20, linewidth=0)
    else:
        # 단색 산점도
        sns.scatterplot(data=df, x=x_col, y=y_col, 
                       alpha=0.4, s=10, linewidth=0, color='blue')
    
    plt.title(f"최적화된 산점도 (n={len(df)})")
    return plt.gcf()

def optimized_hexbin(df, x_col, y_col, gridsize=50):
    """최적화된 육각형 빈 플롯 (대용량 데이터용)"""
    plt.figure(figsize=(10, 8))
    
    # matplotlib의 hexbin 사용
    hb = plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap='viridis')
    plt.colorbar(hb, label='데이터 밀도')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"육각형 빈 플롯 (n={len(df)})")
    
    return plt.gcf()

# 최적화된 플롯 비교
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 최적화된 산점도
df_sample = large_df.sample(n=20000, random_state=42)
sns.scatterplot(data=df_sample, x='x', y='y', alpha=0.3, s=5, ax=axes[0])
axes[0].set_title("최적화된 산점도 (20,000개)")

# 육각형 빈 플롯
axes[1].hexbin(large_df['x'], large_df['y'], gridsize=30, cmap='viridis')
axes[1].set_title("육각형 빈 플롯 (100,000개)")

plt.tight_layout()
plt.show()
```

## 메모리 최적화

### 데이터 타입 최적화

```python
def optimize_dtypes(df):
    """데이터 타입 최적화"""
    optimized_df = df.copy()
    
    # 정수형 최적화
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                optimized_df[col] = optimized_df[col].astype('uint8')
            elif col_max < 65535:
                optimized_df[col] = optimized_df[col].astype('uint16')
            elif col_max < 4294967295:
                optimized_df[col] = optimized_df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
    
    # 실수형 최적화
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    
    # 범주형 최적화
    for col in optimized_df.select_dtypes(include=['object']).columns:
        if optimized_df[col].nunique() / len(optimized_df) < 0.5:
            optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df

# 메모리 사용량 비교
original_memory = large_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
optimized_df = optimize_dtypes(large_df)
optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

print(f"원본 데이터 메모리 사용량: {original_memory:.2f} MB")
print(f"최적화된 데이터 메모리 사용량: {optimized_memory:.2f} MB")
print(f"메모리 절감: {original_memory - optimized_memory:.2f} MB ({(1 - optimized_memory/original_memory)*100:.1f}%)")
```

### 청크 처리

```python
def process_large_dataset_in_chunks(df, chunk_size=10000, plot_func=sns.scatterplot):
    """대용량 데이터를 청크로 처리"""
    n_chunks = len(df) // chunk_size + 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_chunks))
    
    for i, chunk in enumerate(np.array_split(df, n_chunks)):
        if len(chunk) > 0:
            # 각 청크에 대해 플롯 함수 적용
            if plot_func == sns.scatterplot:
                ax.scatter(chunk['x'], chunk['y'], alpha=0.5, s=10, color=colors[i])
            # 다른 플롯 유형에 대한 처리도 추가 가능
    
    ax.set_title(f"청크 처리된 플롯 (총 {len(df)}개)")
    return fig

# 청크 처리 예시
chunk_fig = process_large_dataset_in_chunks(large_df, chunk_size=20000)
plt.show()
```

## 캐싱 및 재사용

### 플롯 캐싱

```python
from functools import lru_cache
import pickle
import os

class PlotCache:
    def __init__(self, cache_dir="plot_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, data_hash, plot_type):
        """캐시 파일 경로 생성"""
        return os.path.join(self.cache_dir, f"{plot_type}_{data_hash}.pkl")
    
    def _get_data_hash(self, data):
        """데이터 해시 생성"""
        # 간단한 해시 함수 (실제로는 더 복잡한 방법 사용 가능)
        return str(len(data)) + "_" + str(data.select_dtypes(include=[np.number]).sum().sum())
    
    def get_plot(self, data, plot_type, plot_func):
        """캐시된 플롯 가져오기 또는 생성"""
        data_hash = self._get_data_hash(data)
        cache_path = self._get_cache_path(data_hash, plot_type)
        
        # 캐시 확인
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_result = pickle.load(f)
            print(f"캐시된 플롯 로드: {plot_type}")
            return cached_result
        
        # 새로운 플롯 생성
        print(f"새 플롯 생성: {plot_type}")
        fig = plot_func(data)
        
        # 캐시 저장
        with open(cache_path, 'wb') as f:
            pickle.dump(fig, f)
        
        return fig

# 캐시 사용 예시
plot_cache = PlotCache()

def create_scatterplot(data):
    """산점도 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='x', y='y', ax=ax)
    ax.set_title("캐시된 산점도")
    return fig

def create_histogram(data):
    """히스토그램 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['x'], ax=ax)
    ax.set_title("캐시된 히스토그램")
    return fig

# 처음 실행 (캐시 저장)
fig1 = plot_cache.get_plot(large_df, "scatterplot", create_scatterplot)
plt.close(fig1)

# 두 번째 실행 (캐시 로드)
fig2 = plot_cache.get_plot(large_df, "scatterplot", create_scatterplot)
plt.show()

plt.close(fig2)  # 메모리 정리
```

### 전처리 결과 캐싱

```python
class PreprocessingCache:
    def __init__(self, cache_dir="preprocessing_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def preprocess_data(self, df, processing_func, cache_key):
        """전처리 결과 캐싱"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # 캐시 확인
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"캐시된 전처리 결과 로드: {cache_key}")
            return cached_data
        
        # 전처리 실행
        print(f"전처리 실행: {cache_key}")
        processed_data = processing_func(df)
        
        # 캐시 저장
        with open(cache_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        return processed_data

# 전처리 함수 예시
def remove_outliers(df, columns, threshold=3):
    """이상치 제거"""
    result = df.copy()
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        result = result[z_scores < threshold]
    return result

def create_aggregated_data(df, group_cols, agg_cols, agg_func='mean'):
    """집계 데이터 생성"""
    return df.groupby(group_cols)[agg_cols].agg(agg_func).reset_index()

# 전처리 캐시 사용
preprocessing_cache = PreprocessingCache()

# 이상치 제거 (캐시됨)
clean_df = preprocessing_cache.preprocess_data(
    large_df, 
    lambda df: remove_outliers(df, ['x', 'y']),
    "remove_outliers_xy"
)

# 집계 데이터 생성 (캐시됨)
agg_df = preprocessing_cache.preprocess_data(
    large_df,
    lambda df: create_aggregated_data(df, ['category'], ['value', 'x']),
    "agg_by_category"
)

print(f"원본 데이터: {len(large_df)}개")
print(f"이상치 제거된 데이터: {len(clean_df)}개")
print(f"집계 데이터: {len(agg_df)}개")
```

## 모범 사례

### 1. 성능 모니터링

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def measure_plot_performance(self, plot_name, plot_func, *args, **kwargs):
        """플롯 성능 측정"""
        # 시간 측정
        start_time = time.time()
        fig = plot_func(*args, **kwargs)
        end_time = time.time()
        
        # 메모리 사용량 측정 (개념적)
        memory_usage = 100  # 실제로는 psutil 등 사용
        
        self.metrics[plot_name] = {
            'execution_time': end_time - start_time,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        }
        
        print(f"{plot_name}: {end_time - start_time:.3f}초")
        return fig
    
    def get_performance_report(self):
        """성능 보고서 생성"""
        if not self.metrics:
            return "성능 데이터가 없습니다."
        
        report = "성능 보고서:\n"
        report += "=" * 40 + "\n"
        
        for plot_name, metrics in self.metrics.items():
            report += f"{plot_name}:\n"
            report += f"  실행 시간: {metrics['execution_time']:.3f}초\n"
            report += f"  메모리 사용량: {metrics['memory_usage']:.1f}MB\n"
            report += "\n"
        
        return report

# 성능 모니터링 사용 예시
monitor = PerformanceMonitor()

# 다양한 플롯 성능 측정
fig1 = monitor.measure_plot_performance(
    "산점도", 
    lambda: sns.scatterplot(data=large_df.sample(10000), x='x', y='y')
)
plt.close(fig1)

fig2 = monitor.measure_plot_performance(
    "히스토그램", 
    lambda: sns.histplot(large_df['x'].sample(10000))
)
plt.close(fig2)

print(monitor.get_performance_report())
```

### 2. 적응적 시각화

```python
def adaptive_visualization(df, x_col, y_col=None, max_points=50000):
    """데이터 크기에 따른 적응적 시각화"""
    n_points = len(df)
    
    if y_col is None:
        # 단변량 시각화
        if n_points > max_points * 10:
            # 매우 큰 데이터: 히스토그램
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[x_col], bins=50, alpha=0.7)
            ax.set_title(f"히스토그램 (n={n_points:,})")
            
        elif n_points > max_points:
            # 큰 데이터: 밀도 플롯
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(df[x_col].sample(max_points), ax=ax)
            ax.set_title(f"밀도 플롯 (n={n_points:,})")
            
        else:
            # 작은 데이터: 러그 플롯
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.rugplot(df[x_col], ax=ax)
            sns.histplot(df[x_col], kde=True, ax=ax)
            ax.set_title(f"분포 플롯 (n={n_points:,})")
    
    else:
        # 이변량 시각화
        if n_points > max_points:
            # 큰 데이터: 2D 히스토그램 또는 육각형 빈 플롯
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.hexbin(df[x_col], df[y_col], gridsize=30, cmap='viridis')
            plt.colorbar(ax.collections[0], ax=ax, label='데이터 밀도')
            ax.set_title(f"육각형 빈 플롯 (n={n_points:,})")
            
        elif n_points > max_points // 5:
            # 중간 크기: 밀도 플롯
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.kdeplot(data=df.sample(max_points), x=x_col, y=y_col, ax=ax)
            ax.set_title(f"밀도 플롯 (n={n_points:,})")
            
        else:
            # 작은 데이터: 산점도
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"산점도 (n={n_points:,})")
    
    return fig

# 적응적 시각화 예시
fig1 = adaptive_visualization(large_df.sample(5000), 'x')
plt.show()

fig2 = adaptive_visualization(large_df.sample(50000), 'x', 'y')
plt.show()
```

### 3. 자원 관리

```python
def resource_efficient_plotting(df, plot_functions, memory_limit_mb=500):
    """자원 효율적인 플롯 생성"""
    # 현재 메모리 사용량 확인 (개념적)
    current_memory = 200  # 실제로는 psutil 등 사용
    
fig, axes = plt.subplots(1, len(plot_functions), figsize=(5*len(plot_functions), 6))
    
    if len(plot_functions) == 1:
        axes = [axes]
    
    for i, (plot_name, plot_func) in enumerate(plot_functions.items()):
        # 메모리 확인
        if current_memory > memory_limit_mb:
            print(f"메모리 한계 도달: {current_memory}MB > {memory_limit_mb}MB")
            break
        
        # 플롯 생성
        try:
            plot_func(axes[i])
            axes[i].set_title(plot_name)
            
            # 메모리 사용량 업데이트 (개념적)
            current_memory += 50
            
        except Exception as e:
            print(f"플롯 생성 오류 ({plot_name}): {e}")
            axes[i].text(0.5, 0.5, f"오류: {plot_name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    return fig

# 자원 효율적 플롯 예시
plot_functions = {
    "분포": lambda ax: sns.histplot(large_df['x'].sample(10000), ax=ax),
    "상자 그림": lambda ax: sns.boxplot(data=large_df.sample(10000), x='category', y='value', ax=ax),
    "산점도": lambda ax: sns.scatterplot(data=large_df.sample(5000), x='x', y='y', ax=ax)
}

fig = resource_efficient_plotting(large_df, plot_functions)
plt.show()
```

## 다음 단계

성능 최적화를 익혔다면, [문제 해결 실용 가이드](10-troubleshooting-practical.md) 문서에서 일반적인 문제점과 실용적인 해결 방안을 학습해보세요.

## 추가 자료

- [pandas 성능 최적화 가이드](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [matplotlib 성능 팁](https://matplotlib.org/stable/users/explain/performance.html)
- [Python 데이터 과학 성능 최적화](https://realpython.com/python-data-performance/)
- [대용량 데이터 시각화](https://stackoverflow.com/questions/8627025/what-are-good-plotting-and-charting-libraries-for-a-high-no-of-data-points)