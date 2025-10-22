# Seaborn과 pandas 및 matplotlib 연동

Seaborn은 pandas와 matplotlib와 깊이 통합되어 설계되었습니다. 이 문서에서는 이 라이브러리들과의 효과적인 연동 방법을 다룹니다.

## Seaborn과 pandas 연동

### DataFrame을 이용한 기본 시각화

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
data = {
    'date': pd.date_range('2023-01-01', periods=100),
    'sales': np.cumsum(np.random.randn(100)) + 100,
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['서울', '부산', '대구'], 100),
    'temperature': np.random.normal(20, 5, 100)
}
df = pd.DataFrame(data)

# DataFrame 직접 사용
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='sales', hue='category')
plt.title("카테고리별 판매 추세")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### pandas 데이터 구조 활용

```python
# 인덱스 활용
df_indexed = df.set_index('date')

# 인덱스를 이용한 시각화
plt.figure(figsize=(12, 8))

# 서브플롯 1: 인덱스 활용
plt.subplot(2, 2, 1)
sns.lineplot(data=df_indexed['sales'])
plt.title("인덱스를 이용한 판매 추세")

# 서브플롯 2: 그룹화 활용
plt.subplot(2, 2, 2)
monthly_sales = df.groupby(df['date'].dt.month)['sales'].sum().reset_index()
sns.barplot(data=monthly_sales, x='date', y='sales')
plt.title("월별 판매 합계")

# 서브플롯 3: 피벗 테이블 활용
plt.subplot(2, 2, 3)
pivot_data = df.pivot_table(values='sales', index='category', columns='region', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("카테고리-지역별 평균 판매액")

# 서브플롯 4: 리샤피 활용
plt.subplot(2, 2, 4)
melted_data = df.melt(id_vars=['category', 'region'], value_vars=['sales', 'temperature'])
sns.boxplot(data=melted_data, x='category', y='value', hue='variable')
plt.title("카테고리별 변수 분포")

plt.tight_layout()
plt.show()
```

### pandas 데이터 처리와 시각화 연동

```python
# 데이터 전처리와 시각화 연동
def preprocess_and_visualize(df):
    """
    데이터 전처리와 시각화를 결합한 함수
    """
    # 1. 결측치 처리
    df_processed = df.copy()
    df_processed['sales'] = df_processed['sales'].fillna(df_processed['sales'].mean())
    
    # 2. 이상치 처리
    Q1 = df_processed['sales'].quantile(0.25)
    Q3 = df_processed['sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_cleaned = df_processed[
        (df_processed['sales'] >= lower_bound) & 
        (df_processed['sales'] <= upper_bound)
    ]
    
    # 3. 파생 변수 생성
    df_cleaned['sales_category'] = pd.cut(
        df_cleaned['sales'], 
        bins=[0, 50, 100, 150, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # 4. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 원본 데이터
    sns.histplot(df['sales'], ax=axes[0, 0], color='red', alpha=0.5, label='원본')
    sns.histplot(df_cleaned['sales'], ax=axes[0, 0], color='blue', alpha=0.7, label='정제')
    axes[0, 0].set_title("데이터 정제 전후 분포")
    axes[0, 0].legend()
    
    # 카테고리별 분석
    sns.boxplot(data=df_cleaned, x='category', y='sales', ax=axes[0, 1])
    axes[0, 1].set_title("카테고리별 판매액 분포")
    
    # 시계열 추세
    sns.lineplot(data=df_cleaned, x='date', y='sales', hue='category', ax=axes[1, 0])
    axes[1, 0].set_title("카테고리별 시계열 추세")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 상관관계
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_cleaned[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("상관관계 행렬")
    
    plt.tight_layout()
    return df_cleaned

df_cleaned = preprocess_and_visualize(df)
```

## Seaborn과 matplotlib 연동

### matplotlib 축 객체 활용

```python
# matplotlib 축 객체와 Seaborn 연동
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 각 축에 다른 Seaborn 플롯
sns.scatterplot(data=df, x='temperature', y='sales', ax=axes[0, 0])
axes[0, 0].set_title("온도와 판매액의 관계")

sns.boxplot(data=df, x='category', y='sales', ax=axes[0, 1])
axes[0, 1].set_title("카테고리별 판매액")

sns.histplot(df['sales'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("판매액 분포")

sns.violinplot(data=df, x='region', y='sales', ax=axes[1, 1])
axes[1, 1].set_title("지역별 판매액 분포")

# 전체 제목
fig.suptitle("matplotlib과 Seaborn 연동 예제", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()
```

### matplotlib 스타일과 Seaborn 결합

```python
# matplotlib 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')

# Seaborn 테마와 결합
sns.set_palette("deep")

fig, ax = plt.subplots(figsize=(12, 6))

# Seaborn 플롯
sns.scatterplot(
    data=df, 
    x='temperature', 
    y='sales', 
    hue='category',
    size='sales',
    sizes=(50, 200),
    alpha=0.7,
    ax=ax
)

# matplotlib 추가 요소
ax.set_title("온도와 판매액의 관계 (matplotlib + seaborn)", fontsize=14, fontweight='bold')
ax.set_xlabel("온도 (°C)", fontsize=12)
ax.set_ylabel("판매액", fontsize=12)

# matplotlib 주석
ax.axhline(y=df['sales'].mean(), color='red', linestyle='--', alpha=0.7)
ax.text(df['temperature'].max(), df['sales'].mean(), 
         f"평균: {df['sales'].mean():.1f}", 
         ha='right', va='bottom')

# matplotlib 범례 커스터마이징
legend = ax.legend(title='카테고리', bbox_to_anchor=(1.05, 1), loc='upper left')
legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.show()
```

### 복합 matplotlib-Seaborn 시각화

```python
# 복합 시각화: matplotlib과 Seaborn 요소 결합
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. matplotlib 3D 플롯
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
from mpl_toolkits.mplot3d import Axes3D
scatter = ax1.scatter(df['temperature'], df['sales'], df.index, c=df['category'].map({'A': 0, 'B': 1, 'C': 2}), cmap='viridis')
ax1.set_xlabel('온도')
ax1.set_ylabel('판매액')
ax1.set_zlabel('인덱스')
ax1.set_title('3D 산점도')

# 2. Seaborn 히트맵
ax2 = fig.add_subplot(gs[0, 1:])
correlation_data = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_data, annot=True, cmap="RdBu_r", center=0, ax=ax2)
ax2.set_title("상관관계 히트맵")

# 3. matplotlib 파이 차트
ax3 = fig.add_subplot(gs[1, 0])
category_counts = df['category'].value_counts()
ax3.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
ax3.set_title("카테고리 비율")

# 4. Seaborn 박스 플롯
ax4 = fig.add_subplot(gs[1, 1:])
sns.boxplot(data=df, x='category', y='sales', hue='region', ax=ax4)
ax4.set_title("카테고리-지역별 판매액")

# 5. matplotlib 라인 플롯과 Seaborn 신뢰 구간
ax5 = fig.add_subplot(gs[2, :])
# matplotlib 라인 플롯
ax5.plot(df['date'], df['sales'], color='blue', alpha=0.3, label='일별 판매액')
# Seaborn 이동 평균과 신뢰 구간
monthly_data = df.set_index('date').resample('M')['sales'].agg(['mean', 'std']).reset_index()
ax5.errorbar(monthly_data['date'], monthly_data['mean'], yerr=monthly_data['std'], 
             fmt='o-', color='red', capsize=5, label='월별 평균±표준편차')
ax5.set_title("시계열 추세")
ax5.legend()

plt.suptitle("matplotlib과 Seaborn 복합 시각화", fontsize=16, y=0.98)
plt.show()
```

## 고급 연동 기법

### 사용자 정의 함수와 연동

```python
# 재사용 가능한 시각화 함수
def create_comprehensive_analysis(df, target_col, group_cols):
    """
    pandas DataFrame에 대한 종합적 분석 시각화
    """
    n_groups = len(group_cols)
    fig, axes = plt.subplots(2, n_groups, figsize=(5*n_groups, 10))
    
    if n_groups == 1:
        axes = axes.reshape(2, 1)
    
    for i, group_col in enumerate(group_cols):
        # 상단: 분석 플롯
        sns.boxplot(data=df, x=group_col, y=target_col, ax=axes[0, i])
        axes[0, i].set_title(f"{group_col}별 {target_col} 분포")
        
        # 하단: 통계 요약
        stats = df.groupby(group_col)[target_col].agg(['mean', 'std', 'count']).reset_index()
        
        # 막대 그래프 (평균)
        bars = axes[1, i].bar(stats[group_col], stats['mean'], yerr=stats['std'], 
                             capsize=5, alpha=0.7)
        axes[1, i].set_title(f"{group_col}별 평균 {target_col}")
        axes[1, i].set_ylabel(f"평균 {target_col}")
        
        # 값 레이블 추가
        for bar, (idx, row) in zip(bars, stats.iterrows()):
            height = bar.get_height()
            axes[1, i].text(bar.get_x() + bar.get_width()/2., height + row['std'],
                           f'{row["mean"]:.1f}±{row["std"]:.1f}\n(n={row["count"]})',
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

# 사용 예시
fig = create_comprehensive_analysis(df, 'sales', ['category', 'region'])
plt.show()
```

### 인터랙티브 요소 추가

```python
# 인터랙티브한 주석 추가 (개념 예시)
def interactive_annotations(df, x_col, y_col, group_col):
    """
    상호작용적 주석이 포함된 플롯
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 기본 플롯
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, ax=ax, s=100, alpha=0.7)
    
    # 각 그룹의 중심점과 통계 정보
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]
        mean_x = group_data[x_col].mean()
        mean_y = group_data[y_col].mean()
        
        # 중심점 표시
        ax.scatter(mean_x, mean_y, s=200, marker='x', color='black', linewidth=3)
        
        # 주석 추가
        ax.annotate(
            f"{group}\n평균: ({mean_x:.1f}, {mean_y:.1f})\n개수: {len(group_data)}",
            xy=(mean_x, mean_y),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    # 전체 통계 정보
    ax.text(0.02, 0.98, f"전체 데이터: {len(df)}개\n상관계수: {df[x_col].corr(df[y_col]):.3f}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f"{x_col}과 {y_col}의 관계 ({group_col}별)")
    return fig, ax

# 사용 예시
fig, ax = interactive_annotations(df, 'temperature', 'sales', 'category')
plt.show()
```

### 동적 시각화

```python
# 동적 시각화 함수 (개념 예시)
def create_dynamic_visualization(df, time_col, value_col, group_col):
    """
    시간에 따른 동적 시각화
    """
    # 데이터 정렬
    df_sorted = df.sort_values(time_col)
    
    # 전체 기간 플롯
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 상단: 전체 기간
    for group in df_sorted[group_col].unique():
        group_data = df_sorted[df_sorted[group_col] == group]
        ax1.plot(group_data[time_col], group_data[value_col], 
                label=group, alpha=0.7, linewidth=2)
    
    ax1.set_title("전체 기간 추세")
    ax1.legend()
    
    # 하단: 이동 평균과 변화율
    for group in df_sorted[group_col].unique():
        group_data = df_sorted[df_sorted[group_col] == group]
        # 7일 이동 평균
        moving_avg = group_data[value_col].rolling(window=7).mean()
        ax2.plot(group_data[time_col], moving_avg, 
                label=f"{group} (7일 이동평균)", alpha=0.8)
    
    ax2.set_title("이동 평균 추세")
    ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)

# 사용 예시
fig, axes = create_dynamic_visualization(df, 'date', 'sales', 'category')
plt.show()
```

## 모범 사례

### 1. 데이터 파이프라인과 시각화 연동

```python
# 데이터 파이프라인과 시각화를 통합한 클래스
class DataVisualizationPipeline:
    def __init__(self, df):
        self.df = df.copy()
        self.processed_df = None
        self.figures = {}
    
    def preprocess(self):
        """데이터 전처리"""
        # 결측치 처리
        self.processed_df = self.df.copy()
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        self.processed_df[numeric_cols] = self.processed_df[numeric_cols].fillna(
            self.processed_df[numeric_cols].mean()
        )
        
        # 날짜 처리
        if 'date' in self.processed_df.columns:
            self.processed_df['year'] = pd.to_datetime(self.processed_df['date']).dt.year
            self.processed_df['month'] = pd.to_datetime(self.processed_df['date']).dt.month
            self.processed_df['weekday'] = pd.to_datetime(self.processed_df['date']).dt.day_name()
        
        return self.processed_df
    
    def create_basic_analysis(self):
        """기본 분석 시각화"""
        if self.processed_df is None:
            self.preprocess()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 수치형 변수 분포
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:4]):
            row, col_idx = i // 2, i % 2
            sns.histplot(self.processed_df[col], ax=axes[row, col_idx], kde=True)
            axes[row, col_idx].set_title(f"{col} 분포")
        
        plt.tight_layout()
        self.figures['basic_analysis'] = fig
        return fig
    
    def create_correlation_analysis(self):
        """상관관계 분석"""
        if self.processed_df is None:
            self.preprocess()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.processed_df[numeric_cols].corr()
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap="RdBu_r", 
            center=0,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("상관관계 행렬")
        
        self.figures['correlation'] = fig
        return fig
    
    def create_time_series_analysis(self, date_col, value_col):
        """시계열 분석"""
        if self.processed_df is None:
            self.preprocess()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 원본 데이터
        sns.lineplot(data=self.processed_df, x=date_col, y=value_col, ax=axes[0, 0])
        axes[0, 0].set_title("원본 시계열")
        
        # 추세 분해 (개념 예시)
        # 실제로는 statsmodels의 seasonal_decompose 사용
        axes[0, 1].plot(self.processed_df[date_col], 
                       self.processed_df[value_col].rolling(window=7).mean())
        axes[0, 1].set_title("이동 평균")
        
        # 계절성
        if 'month' in self.processed_df.columns:
            monthly_avg = self.processed_df.groupby('month')[value_col].mean()
            axes[1, 0].bar(monthly_avg.index, monthly_avg.values)
            axes[1, 0].set_title("월별 평균")
        
        # 분포
        sns.histplot(self.processed_df[value_col], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title("값 분포")
        
        plt.tight_layout()
        self.figures['time_series'] = fig
        return fig
    
    def generate_report(self, save_path="analysis_report.png"):
        """종합 보고서 생성"""
        if not self.figures:
            self.create_basic_analysis()
            self.create_correlation_analysis()
        
        # 모든 그림을 하나로 결합
        fig = plt.figure(figsize=(20, 15))
        
        # 기존 그림들을 새로운 그림에 추가
        for i, (name, figure) in enumerate(self.figures.items()):
            # 개념적 예시 - 실제 구현은 더 복잡함
            pass
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

# 사용 예시
pipeline = DataVisualizationPipeline(df)
pipeline.preprocess()
pipeline.create_basic_analysis()
pipeline.create_correlation_analysis()
plt.show()
```

### 2. 성능 최적화

```python
# 대용량 데이터 처리를 위한 성능 최적화
def optimized_visualization(df, sample_size=10000):
    """
    대용량 데이터의 효율적 시각화
    """
    # 데이터 샘플링
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
    else:
        df_sampled = df
    
    # 카테고리형 데이터 최적화
    categorical_cols = df_sampled.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_sampled[col].nunique() > 20:
            # 상위 20개 카테고리만 유지
            top_categories = df_sampled[col].value_counts().nlargest(20).index
            df_sampled[col] = df_sampled[col].where(
                df_sampled[col].isin(top_categories), 'Other'
            )
    
    # 효율적인 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 샘플링된 데이터로 플롯
    numeric_cols = df_sampled.select_dtypes(include=[np.number]).columns[:2]
    if len(numeric_cols) >= 2:
        sns.scatterplot(
            data=df_sampled, 
            x=numeric_cols[0], 
            y=numeric_cols[1], 
            alpha=0.5, 
            ax=axes[0, 0]
        )
    
    # 카테고리별 요약
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        sns.boxplot(
            data=df_sampled, 
            x=categorical_cols[0], 
            y=numeric_cols[0], 
            ax=axes[0, 1]
        )
    
    # 히스토그램
    if len(numeric_cols) > 0:
        sns.histplot(df_sampled[numeric_cols[0]], ax=axes[1, 0])
    
    # 상관관계 히트맵
    corr_matrix = df_sampled.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, ax=axes[1, 1], cmap="RdBu_r", center=0)
    
    plt.tight_layout()
    return fig

# 사용 예시
# large_df = generate_large_dataset(100000)  # 대용량 데이터 생성
# fig = optimized_visualization(large_df)
# plt.show()
```

## 다음 단계

pandas 및 matplotlib 연동을 익혔다면, [성능 최적화](09-performance.md) 문서에서 대용량 데이터 처리와 성능 향상 기법을 학습해보세요.

## 추가 자료

- [pandas 시각화 가이드](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [matplotlib 사용자 가이드](https://matplotlib.org/stable/users/index.html)
- [Seaborn과 pandas 통합 예제](https://seaborn.pydata.org/tutorial/relational.html)
- [데이터 과학 워크플로우](https://towardsdatascience.com/end-to-end-data-science-workflow-1fb2eb90e763)