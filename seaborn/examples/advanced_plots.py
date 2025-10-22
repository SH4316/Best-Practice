"""
Seaborn 고급 플롯 예제 코드
이 파일은 docs/04-advanced-plots.md 문서의 예제 코드들을 포함합니다.
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

# 샘플 데이터 로드 및 생성
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

def example_heatmap():
    """기본 히트맵"""
    # 상관관계 행렬 데이터 생성
    data = np.random.randn(10, 12)
    corr = np.corrcoef(data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr)
    plt.title("상관관계 히트맵")
    plt.show()

def example_heatmap_annotated():
    """주석과 색상 막대가 포함된 히트맵"""
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

def example_heatmap_center():
    """중심값이 있는 히트맵"""
    # 중심값을 지정하여 발산 색상 팔레트 사용
    data = np.random.randn(10, 12)
    corr = np.corrcoef(data)
    
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

def example_clustermap():
    """기본 클러스터링"""
    # 샘플 데이터 생성
    data = np.random.randn(20, 15)
    data += np.arange(20).reshape(-1, 1)  # 행별 패턴 추가
    data += np.arange(15).reshape(1, -1)  # 열별 패턴 추가

    sns.clustermap(data)
    plt.title("클러스터링된 히트맵")
    plt.show()

def example_clustermap_custom():
    """사용자 정의 클러스터링"""
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

def example_regplot():
    """기본 회귀 플롯"""
    plt.figure(figsize=(10, 6))
    sns.regplot(data=tips, x="total_bill", y="tip")
    plt.title("총 청구액과 팁의 회귀 관계")
    plt.show()

def example_regplot_confidence():
    """신뢰 구간과 이상치 표시"""
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

def example_regplot_polynomial():
    """다항 회귀"""
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

def example_lmplot():
    """범주별 회귀 플롯"""
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

def example_lmplot_logistic():
    """로지스틱 회귀"""
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

def example_residplot():
    """잔차 플롯"""
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

def example_pairgrid():
    """기본 PairGrid"""
    g = sns.PairGrid(iris, hue="species")
    g.map_upper(sns.scatterplot)    # 위쪽 삼각형: 산점도
    g.map_diag(sns.histplot)        # 대각선: 히스토그램
    g.map_lower(sns.kdeplot)        # 아래쪽 삼각형: 밀도 플롯
    g.add_legend()
    plt.suptitle("Iris 데이터 PairGrid", y=1.02)
    plt.show()

def example_pairgrid_custom():
    """사용자 정의 PairGrid"""
    g = sns.PairGrid(iris, hue="species", 
                     vars=["sepal_length", "sepal_width", "petal_length"])
    g.map_upper(sns.regplot, scatter_kws={'alpha':0.3})
    g.map_diag(sns.histplot, kde=True)
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    g.add_legend()
    plt.suptitle("선택적 변수로 구성된 PairGrid", y=1.02)
    plt.show()

def example_facetgrid():
    """기본 FacetGrid"""
    g = sns.FacetGrid(tips, col="day", row="time", height=4, aspect=1.2)
    g.map(sns.histplot, "total_bill", bins=15)
    g.add_legend()
    plt.suptitle("요일 및 시간별 청구액 분포", y=1.02)
    plt.tight_layout()
    plt.show()

def example_facetgrid_combined():
    """복합 플롯 FacetGrid"""
    g = sns.FacetGrid(tips, col="day", height=4, aspect=1.2)
    g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
    g.map(sns.regplot, "total_bill", "tip", scatter=False, color='red')
    g.add_legend()
    plt.suptitle("요일별 청구액과 팁의 관계", y=1.02)
    plt.tight_layout()
    plt.show()

def example_jointgrid():
    """기본 JointGrid"""
    g = sns.JointGrid(data=tips, x="total_bill", y="tip")
    g.plot(sns.scatterplot, sns.histplot)
    plt.suptitle("총 청구액과 팁의 관계 및 분포", y=1.02)
    plt.show()

def example_jointgrid_advanced():
    """고급 JointGrid"""
    g = sns.JointGrid(data=tips, x="total_bill", y="tip", height=8)
    g.plot_joint(sns.regplot, scatter_kws={'alpha':0.6})
    g.plot_marginals(sns.histplot, kde=True, bins=20)
    g.ax_joint.set_xlabel("총 청구액 ($)")
    g.ax_joint.set_ylabel("팁 ($)")
    plt.suptitle("고급 JointGrid 예제", y=1.02)
    plt.show()

def example_multi_axis():
    """다중 축 플롯"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 첫 번째 y축
    flights = sns.load_dataset("flights")
    year_passengers = flights.groupby('year')['passengers'].sum().reset_index()
    
    ax1.plot(year_passengers['year'], year_passengers['passengers'], color="blue")
    ax1.set_ylabel("승객 수", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # 두 번째 y축
    ax2 = ax1.twinx()
    year_std = flights.groupby('year')['passengers'].std().reset_index()
    ax2.plot(year_std['year'], year_std['passengers'], color="red")
    ax2.set_ylabel("표준편차", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title("연도별 승객 수와 표준편차")
    plt.show()

def example_subplot_layout():
    """복잡한 레이아웃"""
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
    sns.violinplot(data=tips, x="day", y="total_bill", hue="time", ax=ax_bottom)
    ax_bottom.set_title("바이올린 플롯")

    plt.suptitle("복합 레이아웃 예제", fontsize=16, y=0.98)
    plt.show()

def main():
    """모든 예제 실행"""
    print("Seaborn 고급 플롯 예제 실행")
    
    # 행렬 플롯
    print("1. 히트맵 예제")
    example_heatmap()
    example_heatmap_annotated()
    example_heatmap_center()
    
    print("2. 클러스터링된 히트맵")
    example_clustermap()
    example_clustermap_custom()
    
    # 회귀 플롯
    print("3. 회귀 플롯 예제")
    example_regplot()
    example_regplot_confidence()
    example_regplot_polynomial()
    
    print("4. 선형 모델 플롯")
    example_lmplot()
    example_lmplot_logistic()
    
    print("5. 잔차 플롯")
    example_residplot()
    
    # 다중 플롯 그리드
    print("6. 쌍 그리드")
    example_pairgrid()
    example_pairgrid_custom()
    
    print("7. 퍼셋 그리드")
    example_facetgrid()
    example_facetgrid_combined()
    
    print("8. 결합 그리드")
    example_jointgrid()
    example_jointgrid_advanced()
    
    # 고급 시각화 기법
    print("9. 다중 축 플롯")
    example_multi_axis()
    
    print("10. 복합 레이아웃")
    example_subplot_layout()
    
    print("모든 예제 실행 완료!")

if __name__ == "__main__":
    main()