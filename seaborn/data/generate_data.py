"""
Seaborn Best Practice 예제를 위한 샘플 데이터 생성 스크립트
다양한 시각화 예제에 사용할 수 있는 샘플 데이터셋을 생성합니다.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """재현성을 위해 랜덤 시드 설정"""
    np.random.seed(seed)

def generate_sales_data(n_records=1000):
    """
    판매 데이터 생성
    - 시간에 따른 판매량
    - 제품 카테고리
    - 지역별 판매 실적
    """
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_records)]
    
    categories = ['전자제품', '의류', '식품', '가구', '도서', '스포츠']
    regions = ['서울', '부산', '대구', '인천', '대전', '광주', '울산', '수원']
    
    data = {
        'date': dates,
        'category': np.random.choice(categories, n_records),
        'region': np.random.choice(regions, n_records),
        'product_id': [f'P{np.random.randint(1000, 9999)}' for _ in range(n_records)],
        'sales': np.random.lognormal(mean=8, sigma=1, size=n_records).round(2),
        'quantity': np.random.poisson(lam=5, size=n_records) + 1,
        'customer_age': np.random.normal(35, 10, n_records).astype(int).clip(15, 80),
        'customer_gender': np.random.choice(['남성', '여성'], n_records),
        'promotion': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
        'satisfaction': np.random.normal(4, 1, n_records).round(1).clip(1, 5)
    }
    
    df = pd.DataFrame(data)
    
    # 카테고리별 가격대 차이 추가
    category_multipliers = {
        '전자제품': 5.0,
        '가구': 3.0,
        '스포츠': 2.0,
        '의류': 1.5,
        '도서': 0.5,
        '식품': 0.3
    }
    
    df['sales'] = df.apply(lambda row: row['sales'] * category_multipliers.get(row['category'], 1), axis=1)
    
    # 시간 경과에 따른 추세 추가
    for i, date in enumerate(df['date']):
        trend = 1 + (i / n_records) * 0.3  # 30% 성장 추세
        seasonal = 0.2 * np.sin(2 * np.pi * i / 365)  # 계절성
        df.loc[i, 'sales'] *= (trend + seasonal)
    
    return df.round(2)

def generate_customer_data(n_records=500):
    """
    고객 데이터 생성
    - 고객 인구통계학적 정보
    - 구매 패턴
    - 만족도 점수
    """
    data = {
        'customer_id': [f'C{np.random.randint(10000, 99999)}' for _ in range(n_records)],
        'age': np.random.normal(38, 12, n_records).astype(int).clip(18, 75),
        'gender': np.random.choice(['남성', '여성'], n_records),
        'income': np.random.lognormal(mean=10, sigma=0.5, size=n_records).round(0),
        'city': np.random.choice(['서울', '부산', '대구', '인천', '대전', '광주'], n_records),
        'membership_years': np.random.exponential(scale=3, size=n_records).round(1).clip(0.1, 15),
        'total_purchases': np.random.poisson(lam=20, size=n_records) + 1,
        'avg_purchase_value': np.random.lognormal(mean=8, sigma=0.8, size=n_records).round(2),
        'last_purchase_days': np.random.exponential(scale=30, size=n_records).astype(int).clip(1, 365),
        'satisfaction_score': np.random.normal(3.8, 0.8, n_records).round(1).clip(1, 5),
        'complaint_count': np.random.poisson(lam=0.5, size=n_records),
        'newsletter_subscribed': np.random.choice([0, 1], n_records, p=[0.4, 0.6]),
        'preferred_category': np.random.choice(['전자제품', '의류', '식품', '가구', '도서'], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # 수입에 따른 구매 패턴 상관관계 추가
    df['avg_purchase_value'] = df['income'] * np.random.uniform(0.01, 0.05, n_records)
    df['total_purchases'] = df['membership_years'] * np.random.uniform(3, 8, n_records)
    
    # 만족도와 불만 건수의 음의 상관관계
    df['satisfaction_score'] = np.maximum(1, df['satisfaction_score'] - df['complaint_count'] * 0.3)
    
    return df.round(2)

def generate_experiment_data(n_records=200):
    """
    실험 결과 데이터 생성
    - 실험 조건
    - 측정값
    - 그룹별 결과 비교
    """
    groups = ['대조군', '실험군A', '실험군B', '실험군C']
    conditions = ['조건1', '조건2', '조건3']
    
    data = {
        'experiment_id': [f'E{np.random.randint(100, 999)}' for _ in range(n_records)],
        'subject_id': [f'S{np.random.randint(1000, 9999)}' for _ in range(n_records)],
        'group': np.random.choice(groups, n_records),
        'condition': np.random.choice(conditions, n_records),
        'pre_score': np.random.normal(50, 10, n_records),
        'post_score': np.random.normal(55, 12, n_records),
        'measurement_time': np.random.choice(['전측정', '후측정'], n_records),
        'response_time': np.random.exponential(scale=2, size=n_records).round(2),
        'accuracy': np.random.beta(2, 5, n_records).round(3),
        'confidence': np.random.randint(1, 6, n_records),
        'age': np.random.normal(30, 8, n_records).astype(int).clip(18, 65),
        'gender': np.random.choice(['남성', '여성'], n_records),
        'experience_years': np.random.exponential(scale=3, size=n_records).round(1).clip(0, 20)
    }
    
    df = pd.DataFrame(data)
    
    # 그룹별 효과 추가
    group_effects = {'대조군': 0, '실험군A': 5, '실험군B': 8, '실험군C': 12}
    
    for i, row in df.iterrows():
        if row['measurement_time'] == '후측정':
            df.loc[i, 'post_score'] = row['pre_score'] + group_effects[row['group']] + np.random.normal(0, 3)
    
    # 경험에 따른 정확도 향상
    df['accuracy'] = df['accuracy'] + df['experience_years'] * 0.02
    df['accuracy'] = df['accuracy'].clip(0, 1)
    
    return df.round(3)

def generate_financial_data(n_records=252):
    """
    금융 데이터 생성 (거래일 기준 1년)
    - 주식 가격
    - 거래량
    - 여러 자산 간 상관관계
    """
    start_date = datetime(2022, 1, 1)
    dates = pd.bdate_range(start=start_date, periods=n_records)
    
    # 여러 자산의 상관관계를 고려한 가격 생성
    assets = ['주식A', '주식B', '채권', '원자재', '환율']
    
    # 상관관계 행렬
    correlation_matrix = np.array([
        [1.0, 0.6, -0.2, 0.3, -0.1],  # 주식A
        [0.6, 1.0, -0.1, 0.4, -0.2],  # 주식B
        [-0.2, -0.1, 1.0, -0.3, 0.1],  # 채권
        [0.3, 0.4, -0.3, 1.0, 0.2],   # 원자재
        [-0.1, -0.2, 0.1, 0.2, 1.0]   # 환율
    ])
    
    # 상관관계를 갖는 랜덤 움직임 생성
    random_moves = np.random.multivariate_normal(
        mean=[0] * len(assets),
        cov=correlation_matrix,
        size=n_records
    )
    
    # 초기 가격 설정
    initial_prices = [100000, 50000, 10000, 5000, 1200]
    
    data = {'date': dates}
    
    for i, asset in enumerate(assets):
        # 가격 경로 생성 (기하 브라운 운동)
        returns = 0.0005 + random_moves[:, i] * 0.02  # 일일 수익률
        prices = [initial_prices[i]]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data[asset] = prices
        data[f'{asset}_volume'] = np.random.lognormal(10, 1, n_records).astype(int)
        data[f'{asset}_volatility'] = np.abs(random_moves[:, i]) * 100
    
    df = pd.DataFrame(data)
    
    # 시장 지표 추가
    df['market_index'] = (df['주식A'] * 0.4 + df['주식B'] * 0.6) / 1000
    df['market_volatility'] = df[['주식A_volatility', '주식B_volatility']].mean(axis=1)
    
    return df.round(2)

def main():
    """모든 데이터셋 생성 및 저장"""
    print("데이터 생성을 시작합니다...")
    
    # 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    
    # 데이터셋 생성
    sales_df = generate_sales_data(1000)
    customer_df = generate_customer_data(500)
    experiment_df = generate_experiment_data(200)
    financial_df = generate_financial_data(252)
    
    # CSV 파일로 저장
    sales_df.to_csv('data/sample_sales.csv', index=False, encoding='utf-8-sig')
    customer_df.to_csv('data/customer_data.csv', index=False, encoding='utf-8-sig')
    experiment_df.to_csv('data/experiment_results.csv', index=False, encoding='utf-8-sig')
    financial_df.to_csv('data/financial_data.csv', index=False, encoding='utf-8-sig')
    
    # 데이터 정보 출력
    print("\n생성된 데이터셋 정보:")
    print(f"판매 데이터: {sales_df.shape} (행 x 열)")
    print(f"고객 데이터: {customer_df.shape} (행 x 열)")
    print(f"실험 데이터: {experiment_df.shape} (행 x 열)")
    print(f"금융 데이터: {financial_df.shape} (행 x 열)")
    
    print("\n판매 데이터 샘플:")
    print(sales_df.head())
    
    print("\n데이터 생성이 완료되었습니다!")
    
    # 데이터 요약 통계
    print("\n데이터 요약 통계:")
    for name, df in [("판매", sales_df), ("고객", customer_df), ("실험", experiment_df), ("금융", financial_df)]:
        print(f"\n{name} 데이터:")
        print(f"  - 수치형 변수: {len(df.select_dtypes(include=[np.number]).columns)}개")
        print(f"  - 범주형 변수: {len(df.select_dtypes(include=['object']).columns)}개")
        print(f"  - 결측치: {df.isnull().sum().sum()}개")

if __name__ == "__main__":
    set_seed(42)
    main()