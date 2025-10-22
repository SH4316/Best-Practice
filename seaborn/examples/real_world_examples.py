"""
Seaborn 실제 사용 사례 예제 코드
이 파일은 docs/11-examples.md 문서의 예제 코드들을 포함합니다.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기본 테마 설정
sns.set_theme(style="whitegrid")

def generate_sales_data():
    """판매 데이터 생성 (실제와 유사한 구조)"""
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    categories = ['전자제품', '의류', '식품', '가구', '스포츠']
    regions = ['서울', '부산', '대구', '인천']
    
    data = []
    for date in dates:
        for category in categories:
            for region in regions:
                # 기본 판매량에 계절성 추가
                base_sales = np.random.lognormal(8, 0.5)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                # 카테고리별 특성
                category_factor = {
                    '전자제품': 1.5,
                    '의류': 1.2,
                    '식품': 0.8,
                    '가구': 1.1,
                    '스포츠': 0.9
                }[category]
                
                # 지역별 특성
                region_factor = {
                    '서울': 1.3,
                    '부산': 1.1,
                    '대구': 0.9,
                    '인천': 1.0
                }[region]
                
                sales = base_sales * seasonal_factor * category_factor * region_factor
                
                data.append({
                    'date': date,
                    'category': category,
                    'region': region,
                    'sales': sales,
                    'quantity': int(sales / np.random.uniform(50, 200))
                })
    
    return pd.DataFrame(data)

def generate_customer_data():
    """고객 행동 데이터 생성"""
    np.random.seed(42)
    n_customers = 1000
    
    data = []
    for i in range(n_customers):
        # 고객 기본 정보
        age = np.random.normal(35, 10)
        age = max(18, min(80, int(age)))
        
        income = np.random.lognormal(10, 0.5)
        registration_days = np.random.exponential(scale=365)
        
        # 구매 행동 패턴
        if income > 50000:  # 고소득
            purchase_freq = np.random.poisson(5)
            avg_basket = np.random.normal(200, 50)
            churn_prob = 0.1
        elif income > 30000:  # 중소득
            purchase_freq = np.random.poisson(3)
            avg_basket = np.random.normal(100, 30)
            churn_prob = 0.2
        else:  # 저소득
            purchase_freq = np.random.poisson(1)
            avg_basket = np.random.normal(50, 20)
            churn_prob = 0.3
        
        # 이탈 여부 결정
        is_churned = np.random.random() < churn_prob
        
        # 세션 데이터
        sessions = []
        for _ in range(purchase_freq):
            session_duration = np.random.exponential(scale=10)  # 분
            pages_viewed = int(session_duration / 2) + np.random.poisson(2)
            
            # 구매 여부
            if np.random.random() < 0.3:  # 30% 구매 전환율
                purchased = True
                basket_value = max(10, np.random.normal(avg_basket, avg_basket * 0.3))
            else:
                purchased = False
                basket_value = 0
            
            sessions.append({
                'session_duration': session_duration,
                'pages_viewed': pages_viewed,
                'purchased': purchased,
                'basket_value': basket_value
            })
        
        # 고객별 요약
        total_sessions = len(sessions)
        total_purchases = sum(1 for s in sessions if s['purchased'])
        total_revenue = sum(s['basket_value'] for s in sessions if s['purchased'])
        avg_session_duration = np.mean([s['session_duration'] for s in sessions]) if sessions else 0
        avg_pages_viewed = np.mean([s['pages_viewed'] for s in sessions]) if sessions else 0
        
        data.append({
            'customer_id': f'C{i:04d}',
            'age': age,
            'income': income,
            'registration_days': registration_days,
            'total_sessions': total_sessions,
            'total_purchases': total_purchases,
            'total_revenue': total_revenue,
            'avg_session_duration': avg_session_duration,
            'avg_pages_viewed': avg_pages_viewed,
            'is_churned': is_churned
        })
    
    return pd.DataFrame(data)

def generate_ab_test_data():
    """A/B 테스트 데이터 생성"""
    np.random.seed(42)
    n_users = 5000
    
    data = []
    for i in range(n_users):
        # 그룹 할당 (50:50)
        group = np.random.choice(['A', 'B'])
        
        # 기반 전환율
        if group == 'A':  # 기존 디자인
            base_conversion = 0.05
            base_engagement = 30
            base_satisfaction = 3.5
        else:  # 새로운 디자인
            base_conversion = 0.06  # 20% 개선
            base_engagement = 35    # 17% 개선
            base_satisfaction = 3.8  # 9% 개선
        
        # 사용자 특성
        user_type = np.random.choice(['신규', '기존'], p=[0.3, 0.7])
        device = np.random.choice(['모바일', '데스크톱'], p=[0.6, 0.4])
        
        # 특성에 따른 효과 조정
        if user_type == '신규':
            conversion_multiplier = 0.8
        else:
            conversion_multiplier = 1.2
            
        if device == '모바일':
            engagement_multiplier = 1.1
        else:
            engagement_multiplier = 1.0
        
        # 지표 계산
        conversion_prob = base_conversion * conversion_multiplier
        converted = np.random.random() < conversion_prob
        
        engagement_time = max(1, np.random.normal(base_engagement, 10) * engagement_multiplier)
        satisfaction = max(1, min(5, np.random.normal(base_satisfaction, 0.5)))
        
        # 구매 금액 (전환한 경우)
        if converted:
            if group == 'A':
                purchase_amount = np.random.lognormal(4, 0.5)
            else:
                purchase_amount = np.random.lognormal(4.2, 0.5)  # 더 높은 평균
        else:
            purchase_amount = 0
        
        data.append({
            'user_id': f'U{i:05d}',
            'group': group,
            'user_type': user_type,
            'device': device,
            'converted': converted,
            'engagement_time': engagement_time,
            'satisfaction': satisfaction,
            'purchase_amount': purchase_amount
        })
    
    return pd.DataFrame(data)

def create_customer_segments(df):
    """수입과 구매 빈도에 따른 고객 세그먼트"""
    conditions = [
        (df['income'] > 50000) & (df['total_purchases'] > 3),
        (df['income'] > 50000) & (df['total_purchases'] <= 3),
        (df['income'] <= 50000) & (df['total_purchases'] > 3),
        (df['income'] <= 50000) & (df['total_purchases'] <= 3)
    ]
    
    choices = ['VIP', '잠재VIP', '충성고객', '일반고객']
    
    df['segment'] = np.select(conditions, choices, default='일반고객')
    return df

def example_sales_analysis():
    """판매 데이터 분석 예제"""
    print("1. 판매 데이터 분석")
    
    # 데이터 생성 및 전처리
    sales_df = generate_sales_data()
    sales_df['month'] = sales_df['date'].dt.month
    sales_df['quarter'] = sales_df['date'].dt.quarter
    sales_df['weekday'] = sales_df['date'].dt.day_name()
    sales_df['revenue'] = sales_df['sales'] * sales_df['quantity']
    
    print(f"데이터 크기: {sales_df.shape}")
    
    # 카테고리별 성과 분석
    plt.figure(figsize=(15, 12))

    # 1. 카테고리별 총 수익
    plt.subplot(2, 3, 1)
    category_revenue = sales_df.groupby('category')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=category_revenue.values, y=category_revenue.index, palette="viridis")
    plt.title('카테고리별 총 수익')
    plt.xlabel('총 수익 (원)')

    # 2. 카테고리별 평균 판매액
    plt.subplot(2, 3, 2)
    category_avg_sales = sales_df.groupby('category')['sales'].mean().sort_values(ascending=False)
    sns.barplot(x=category_avg_sales.values, y=category_avg_sales.index, palette="plasma")
    plt.title('카테고리별 평균 판매액')
    plt.xlabel('평균 판매액 (원)')

    # 3. 카테고리별 판매량 분포
    plt.subplot(2, 3, 3)
    sns.boxplot(data=sales_df, x='category', y='quantity', palette="muted")
    plt.title('카테고리별 판매량 분포')
    plt.xticks(rotation=45)

    # 4. 카테고리별 월별 추세
    plt.subplot(2, 3, 4)
    monthly_category = sales_df.groupby(['month', 'category'])['revenue'].sum().reset_index()
    sns.lineplot(data=monthly_category, x='month', y='revenue', hue='category', palette="deep")
    plt.title('카테고리별 월별 수익 추세')
    plt.xlabel('월')
    plt.ylabel('수익 (원)')

    # 5. 지역별 카테고리 수익
    plt.subplot(2, 3, 5)
    region_category = sales_df.pivot_table(values='revenue', index='category', columns='region', aggfunc='sum')
    sns.heatmap(region_category, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': '수익 (원)'})
    plt.title('지역별 카테고리 수익')

    # 6. 카테고리별 상관관계
    plt.subplot(2, 3, 6)
    category_corr = sales_df.pivot_table(values='revenue', index='date', columns='category', aggfunc='sum').corr()
    sns.heatmap(category_corr, annot=True, cmap="RdBu_r", center=0, 
                cbar_kws={'label': '상관계수'})
    plt.title('카테고리별 수익 상관관계')

    plt.tight_layout()
    plt.show()

    # 주요 인사이트 도출
    def generate_insights(df):
        """데이터 분석으로 주요 인사이트 도출"""
        insights = []
        
        # 1. 최고 수익 카테고리
        top_category = df.groupby('category')['revenue'].sum().idxmax()
        top_revenue = df.groupby('category')['revenue'].sum().max()
        insights.append(f"최고 수익 카테고리: {top_category} (₩{top_revenue:,.0f})")
        
        # 2. 최고 성장 카테고리
        category_growth = df.groupby(['month', 'category'])['revenue'].sum().unstack()
        growth_rates = {}
        for cat in category_growth.columns:
            if len(category_growth[cat].dropna()) > 1:
                growth = (category_growth[cat].iloc[-1] - category_growth[cat].iloc[0]) / category_growth[cat].iloc[0]
                growth_rates[cat] = growth
        
        if growth_rates:
            best_growth_cat = max(growth_rates, key=growth_rates.get)
            insights.append(f"최고 성장 카테고리: {best_growth_cat} (성장률: {growth_rates[best_growth_cat]:.1%})")
        
        # 3. 최고 수익 지역
        top_region = df.groupby('region')['revenue'].sum().idxmax()
        top_region_revenue = df.groupby('region')['revenue'].sum().max()
        insights.append(f"최고 수익 지역: {top_region} (₩{top_region_revenue:,.0f})")
        
        # 4. 계절성 패턴
        quarterly_sales = df.groupby('quarter')['revenue'].sum()
        best_quarter = quarterly_sales.idxmax()
        insights.append(f"최고 매출 분기: {best_quarter}분기 (₩{quarterly_sales[best_quarter]:,.0f})")
        
        return insights
    
    insights = generate_insights(sales_df)
    print("\n주요 인사이트:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")

def example_customer_analysis():
    """고객 행동 분석 예제"""
    print("\n2. 고객 행동 분석")
    
    # 데이터 생성 및 전처리
    customer_df = generate_customer_data()
    customer_df = create_customer_segments(customer_df)
    
    print(f"고객 데이터: {customer_df.shape}")
    print("고객 세그먼트 분포:")
    print(customer_df['segment'].value_counts())
    
    # 고객 세그먼트 시각화
    plt.figure(figsize=(16, 12))

    # 1. 세그먼트별 분포
    plt.subplot(2, 3, 1)
    segment_counts = customer_df['segment'].value_counts()
    sns.barplot(x=segment_counts.values, y=segment_counts.index, palette="viridis")
    plt.title('고객 세그먼트 분포')
    plt.xlabel('고객 수')

    # 2. 세그먼트별 수익
    plt.subplot(2, 3, 2)
    segment_revenue = customer_df.groupby('segment')['total_revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=segment_revenue.values, y=segment_revenue.index, palette="plasma")
    plt.title('세그먼트별 총 수익')
    plt.xlabel('총 수익 (원)')

    # 3. 세그먼트별 특성 비교
    plt.subplot(2, 3, 3)
    segment_features = customer_df.groupby('segment')[['age', 'income', 'total_sessions', 'total_purchases']].mean()
    sns.heatmap(segment_features.T, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title('세그먼트별 평균 특성')

    # 4. 소득 vs 구매 횟수
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=customer_df, x='income', y='total_purchases', 
                    hue='segment', size='total_revenue', sizes=(20, 200), alpha=0.7)
    plt.title('소득 vs 구매 횟수')
    plt.xlabel('소득')
    plt.ylabel('총 구매 횟수')

    # 5. 이탈률 분석
    plt.subplot(2, 3, 5)
    churn_by_segment = customer_df.groupby('segment')['is_churned'].mean().sort_values(ascending=False)
    sns.barplot(x=churn_by_segment.values, y=churn_by_segment.index, palette="Reds")
    plt.title('세그먼트별 이탈률')
    plt.xlabel('이탈률')

    # 6. 세션 행동 분석
    plt.subplot(2, 3, 6)
    behavior_cols = ['avg_session_duration', 'avg_pages_viewed']
    behavior_by_segment = customer_df.groupby('segment')[behavior_cols].mean()
    sns.heatmap(behavior_by_segment.T, annot=True, fmt=".1f", cmap="Blues")
    plt.title('세그먼트별 행동 특성')

    plt.tight_layout()
    plt.show()

def example_ab_test_analysis():
    """A/B 테스트 결과 분석 예제"""
    print("\n3. A/B 테스트 결과 분석")
    
    # 데이터 생성
    ab_df = generate_ab_test_data()
    print(f"A/B 테스트 데이터: {ab_df.shape}")
    print("그룹별 사용자 수:")
    print(ab_df['group'].value_counts())
    
    # 전환율 계산
    conversion_rates = ab_df.groupby('group')['converted'].mean()
    print(f"\n전환율:")
    print(f"그룹 A (기존): {conversion_rates['A']:.2%}")
    print(f"그룹 B (새로운): {conversion_rates['B']:.2%}")
    print(f"상대적 개선: {(conversion_rates['B']/conversion_rates['A'] - 1):.2%}")
    
    # A/B 테스트 결과 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 전환율 비교
    plt.subplot(2, 3, 1)
    conversion_by_group = ab_df.groupby('group')['converted'].mean().reset_index()
    sns.barplot(data=conversion_by_group, x='group', y='converted', palette=["#3498db", "#e74c3c"])
    plt.title('그룹별 전환율')
    plt.ylabel('전환율')

    # 수치 추가
    for i, rate in enumerate(conversion_by_group['converted']):
        plt.text(i, rate + 0.002, f'{rate:.2%}', ha='center', fontweight='bold')

    # 2. 참여 시간 비교
    plt.subplot(2, 3, 2)
    sns.boxplot(data=ab_df, x='group', y='engagement_time', palette=["#3498db", "#e74c3c"])
    plt.title('그룹별 참여 시간')
    plt.ylabel('참여 시간 (초)')

    # 3. 만족도 비교
    plt.subplot(2, 3, 3)
    sns.violinplot(data=ab_df, x='group', y='satisfaction', palette=["#3498db", "#e74c3c"])
    plt.title('그룹별 만족도')
    plt.ylabel('만족도 (1-5)')

    # 4. 구매 금액 비교 (전환한 사용자만)
    plt.subplot(2, 3, 4)
    converted_users = ab_df[ab_df['converted'] == True]
    sns.histplot(data=converted_users, x='purchase_amount', hue='group', 
                 multiple='layer', alpha=0.7, bins=30, palette=["#3498db", "#e74c3c"])
    plt.title('전환자 구매 금액 분포')
    plt.xlabel('구매 금액')

    # 5. 사용자 유형별 전환율
    plt.subplot(2, 3, 5)
    user_conversion = ab_df.groupby(['group', 'user_type'])['converted'].mean().reset_index()
    sns.barplot(data=user_conversion, x='user_type', y='converted', hue='group', 
                palette=["#3498db", "#e74c3c"])
    plt.title('사용자 유형별 전환율')
    plt.ylabel('전환율')

    # 6. 기기별 전환율
    plt.subplot(2, 3, 6)
    device_conversion = ab_df.groupby(['group', 'device'])['converted'].mean().reset_index()
    sns.barplot(data=device_conversion, x='device', y='converted', hue='group', 
                palette=["#3498db", "#e74c3c"])
    plt.title('기기별 전환율')
    plt.ylabel('전환율')

    plt.tight_layout()
    plt.show()

def example_analysis_workflow():
    """표준 데이터 분석 워크플로우 예제"""
    print("\n4. 데이터 분석 워크플로우")
    
    # 판매 데이터로 워크플로우 시연
    sales_df = generate_sales_data()
    sales_df['revenue'] = sales_df['sales'] * sales_df['quantity']
    
    def analysis_workflow(df, target_col=None):
        """표준 데이터 분석 워크플로우"""
        
        # 1. 데이터 탐색
        print("1. 데이터 탐색")
        print(f"데이터 크기: {df.shape}")
        print(f"결측치: {df.isnull().sum().sum()}")
        print(f"수치형 변수: {len(df.select_dtypes(include=[np.number]).columns)}개")
        print(f"범주형 변수: {len(df.select_dtypes(include=['object', 'category']).columns)}개")
        
        # 2. 기술 통계
        print("\n2. 기술 통계")
        if target_col and target_col in df.columns:
            print(f"목표 변수 분포:")
            print(df[target_col].describe())
        
        # 3. 시각화
        print("\n3. 시각화")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # 처음 4개
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < 4:
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f"{col} 분포")
        
        plt.tight_layout()
        plt.show()
        
        # 4. 상관관계 분석
        print("\n4. 상관관계 분석")
        numeric_data = df.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            correlation_matrix = numeric_data.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0)
            plt.title("상관관계 행렬")
            plt.show()
        
        # 5. 주요 인사이트
        print("\n5. 주요 인사이트")
        insights = []
        
        # 최고 상관관계
        if len(numeric_data.columns) > 1:
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        abs(correlation_matrix.iloc[i, j])
                    ))
            
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            if corr_pairs:
                insights.append(f"가장 강한 상관관계: {corr_pairs[0][0]}와 {corr_pairs[0][1]} (r={corr_pairs[0][2]:.2f})")
        
        # 범주형 변수 분석
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:3]:  # 처음 3개
            if df[col].nunique() <= 10:  # 카테고리가 10개 이하인 경우
                most_common = df[col].value_counts().index[0]
                insights.append(f"{col}: 가장 빈번한 값은 '{most_common}' ({df[col].value_counts().iloc[0]}회)")
        
        for insight in insights:
            print(f"- {insight}")
        
        return insights
    
    # 워크플로우 적용
    insights = analysis_workflow(sales_df, target_col='revenue')

def main():
    """모든 예제 실행"""
    print("Seaborn 실제 사용 사례 예제 실행")
    
    # 사례 1: 판매 데이터 분석
    example_sales_analysis()
    
    # 사례 2: 고객 행동 분석
    example_customer_analysis()
    
    # 사례 3: A/B 테스트 결과 분석
    example_ab_test_analysis()
    
    # 사례 4: 데이터 분석 워크플로우
    example_analysis_workflow()
    
    print("\n모든 예제 실행 완료!")

if __name__ == "__main__":
    main()