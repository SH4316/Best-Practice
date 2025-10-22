"""
Seaborn 기본 플롯 예제 코드
이 파일은 docs/03-basic-plots.md 문서의 예제 코드들을 포함합니다.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기본 테마 설정
sns.set_theme(style="whitegrid")

# 샘플 데이터 로드
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

def example_scatterplot():
    """기본 산점도 예제"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tips, x="total_bill", y="tip")
    plt.title("총 청구액과 팁의 관계")
    plt.xlabel("총 청구액 ($)")
    plt.ylabel("팁 ($)")
    plt.show()

def example_scatterplot_hue():
    """범주별 색상 구분 산점도"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
    plt.title("요일별 총 청구액과 팁의 관계")
    plt.show()

def example_scatterplot_advanced():
    """다차원 산점도 예제"""
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

def example_lineplot():
    """기본 선 그래프 예제"""
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

def example_lineplot_advanced():
    """신뢰 구간과 범주별 선 그래프"""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'value': np.cumsum(np.random.randn(100)) + 100,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

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

def example_relplot():
    """다중 플롯 생성"""
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

def example_boxplot():
    """기본 박스 플롯"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=tips, x="day", y="total_bill")
    plt.title("요일별 총 청구액 분포")
    plt.show()

def example_boxplot_hue():
    """범주별 박스 플롯"""
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

def example_violinplot():
    """기본 바이올린 플롯"""
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=tips, x="day", y="total_bill")
    plt.title("요일별 총 청구액 분포")
    plt.show()

def example_violinplot_split():
    """분할 바이올린 플롯"""
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

def example_barplot():
    """기본 막대 그래프"""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tips, x="day", y="total_bill")
    plt.title("요일별 평균 청구액")
    plt.show()

def example_barplot_ci():
    """신뢰 구간이 포함된 막대 그래프"""
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

def example_countplot():
    """기본 카운트 플롯"""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=tips, x="day")
    plt.title("요일별 방문자 수")
    plt.show()

def example_countplot_horizontal():
    """수평 카운트 플롯"""
    plt.figure(figsize=(10, 7))
    sns.countplot(data=tips, y="day", hue="time")
    plt.title("요일 및 시간별 방문자 수")
    plt.legend(title="시간")
    plt.show()

def example_histplot():
    """기본 히스토그램"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=tips, x="total_bill", bins=20)
    plt.title("총 청구액 분포")
    plt.xlabel("총 청구액 ($)")
    plt.ylabel("빈도")
    plt.show()

def example_histplot_kde():
    """커널 밀도 추정이 포함된 히스토그램"""
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=tips, 
        x="total_bill", 
        bins=20,
        kde=True,        # 커널 밀도 추정 곡선 추가
        stat="density"   # 밀도로 표시
    )
    plt.title("총 청구액 분포와 밀도")
    plt.xlabel("총 청구액 ($)")
    plt.show()

def example_kdeplot():
    """기본 밀도 플롯"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=tips, x="total_bill")
    plt.title("총 청구액 밀도")
    plt.xlabel("총 청구액 ($)")
    plt.show()

def example_kdeplot_2d():
    """2D 밀도 플롯"""
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=tips, x="total_bill", y="tip", shade=True, cmap="Blues")
    plt.title("총 청구액과 팁의 2D 밀도")
    plt.xlabel("총 청구액 ($)")
    plt.ylabel("팁 ($)")
    plt.show()

def example_ecdfplot():
    """경험적 누적 분포 함수"""
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=tips, x="total_bill")
    plt.title("총 청구액의 누적 분포")
    plt.xlabel("총 청구액 ($)")
    plt.ylabel("누적 확률")
    plt.grid(True, alpha=0.3)
    plt.show()

def example_displot():
    """다중 분포 플롯"""
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

def main():
    """모든 예제 실행"""
    print("Seaborn 기본 플롯 예제 실행")
    
    # 관계형 플롯
    print("1. 산점도 예제")
    example_scatterplot()
    example_scatterplot_hue()
    example_scatterplot_advanced()
    
    print("2. 선 그래프 예제")
    example_lineplot()
    example_lineplot_advanced()
    
    print("3. 관계형 플롯 인터페이스")
    example_relplot()
    
    # 범주형 플롯
    print("4. 박스 플롯 예제")
    example_boxplot()
    example_boxplot_hue()
    
    print("5. 바이올린 플롯 예제")
    example_violinplot()
    example_violinplot_split()
    
    print("6. 막대 그래프 예제")
    example_barplot()
    example_barplot_ci()
    
    print("7. 카운트 플롯 예제")
    example_countplot()
    example_countplot_horizontal()
    
    # 분포 플롯
    print("8. 히스토그램 예제")
    example_histplot()
    example_histplot_kde()
    
    print("9. 밀도 플롯 예제")
    example_kdeplot()
    example_kdeplot_2d()
    
    print("10. 누적 분포 플롯")
    example_ecdfplot()
    
    print("11. 분포 플롯 인터페이스")
    example_displot()
    
    print("모든 예제 실행 완료!")

if __name__ == "__main__":
    main()