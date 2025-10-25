# 14주차: LLM 응용

## 강의 목표
- LLM의 다양한 응용 분야와 특성 이해
- 생성형 AI 애플리케이션 개발 방법과 고려사항 습득
- 검색 증강 생성(RAG)의 원리와 구현 방법 파악
- 멀티모달 LLM의 아키텍처와 응용 사례 학습
- LLM 배포와 운영의 실제적 고려사항과 최적화 전략 이해

## 이론 강의 (90분)

### 1. LLM 응용 분야 개요 (20분)

#### 주요 응용 분야
**자연어 처리**
- 텍스트 생성: 창작, 요약, 번역
- 질문 답변: 정보 검색, 대화 시스템
- 텍스트 분류: 감성 분석, 주제 분류, 스팸 탐지
- 정보 추출: 엔티티 인식, 관계 추출, 키워드 추출

**콘텐츠 생성**
- 이미지 생성: 텍스트-이미지 변환, 스타일 전이
- 비디오 생성: 텍스트-비디오 변환, 동영상 제작
- 오디오 생성: 음성 합성, 음악 작곡
- 멀티모달 생성: 여러 모달리티의 통합적 생성

**코드 생성**
- 코드 완성: 부분 코드에서 전체 코드 생성
- 코드 설명: 코드의 기능과 동작 설명
- 디버깅: 코드 오류 탐지와 수정 제안
- 코드 번역: 한 프로그래밍 언어에서 다른 언어로 변환

**전문 분야 응용**
- 의료: 진단 보고서 생성, 의료 질문 답변
- 법률: 법률 조언, 계약서 분석, 판례 검색
- 금융: 시장 분석, 투자 조언, 리스크 평가
- 교육: 개인화된 학습 자료 생성, 질문 답변

#### 응용별 특성과 요구사항
**성능 요구사항**
- 응답 속도: 실시간 응답 vs 배치 처리
- 정확성: 사실적 정확성 vs 창의성
- 일관성: 응답의 논리적 일관성
- 안정성: 장기간 안정적인 성능 유지

**안전성 요구사항**
- 유해 콘텐츠 필터링: 폭력, 증오, 차별적 내용 차단
- 편향성 완화: 성별, 인종, 문화적 편향 감소
- 개인정보 보호: 민감 정보 처리와 보호
- 윤리적 가이드라인: 윤리적 사용 가이드라인 준수

**확장성 요구사항**
- 수평적 확장: 사용자 증가에 따른 성능 유지
- 수직적 확장: 새로운 기능과 도메인 추가 용이성
- 모듈성: 독립적인 모듈로의 구성과 통합
- 호환성: 다양한 플랫폼과 시스템과의 호환성

#### LLM 응용의 발전 방향
**전문화**
- 도메인 특화 모델: 특정 분야에 최적화된 모델
- 작업 특화 모델: 특정 작업에 최적화된 모델
- 산업별 솔루션: 산업별 특화된 솔루션 개발

**개인화**
- 사용자 적응: 개별 사용자의 선호도와 스타일 학습
- 맥락 인식: 사용자의 맥락과 과거 상호작용 이해
- 개인정보 활용: 사용자의 개인정보를 활용한 맞춤형 서비스

**통합**
- 다른 AI 기술과의 통합: 컴퓨터 비전, 음성 인식 등
- 기존 시스템과의 통합: ERP, CRM 등 기존 시스템 연동
- 하이브리드 접근: 규칙 기반과 학습 기반의 결합

### 2. 생성형 AI 애플리케이션 (25분)

#### 생성형 AI의 기본 구조
**프론트엔드**
- 사용자 인터페이스: 채팅, 음성, 텍스트 입력
- 응답 표시: 텍스트, 음성, 이미지 등 다양한 형태
- 상호작용: 실시간 피드백, 수정 요청, 재생성 요청
- 개인화: 사용자 설정, 선호도, 과거 대화 기록

**백엔드**
- LLM API: 모델 추론을 위한 API 서버
- 프롬프트 엔지니어링: 사용자 입력을 모델 입력으로 변환
- 컨텍스트 관리: 대화 기록, 사용자 정보, 세션 정보
- 후처리: 모델 출력의 필터링, 포맷팅, 개인화

**데이터베이스**
- 사용자 정보: 계정 정보, 선호도, 사용 패턴
- 대화 기록: 과거 대화, 피드백, 수정 내역
- 컨텐츠: 생성된 컨텐츠, 메타데이터, 버전 관리
- 분석 데이터: 사용 통계, 성능 지표, 오류 로그

#### 프롬프트 엔지니어링
**기본 프롬프트 구조**
- 시스템 프롬프트: 모델의 역할과 동작 방식 정의
- 사용자 프롬프트: 사용자의 요청과 맥락 정보
- 대화 프롬프트: 대화의 흐름과 맥락 정보
- 출력 프롬프트: 원하는 출력 형식과 스타일 지정

**고급 프롬프트 기법**
- 제로샷 프롬프팅: 예시를 통한 원하는 응답 유도
- 사슬 프롬프팅: 단계별 사고 과정을 통한 복잡한 문제 해결
- 자기 일관성: 모델이 자신의 출력을 일관되게 유지하도록 유도
- 지식 증류: 전문가 지식을 모델에 제공

**프롬프트 최적화**
- A/B 테스트: 다양한 프롬프트의 성능 비교
- 사용자 피드백: 사용자의 만족도를 기반으로 프롬프트 개선
- 자동화: 사용자 입력과 맥락에 따라 동적 프롬프트 생성
- 개인화: 사용자별 최적화된 프롬프트 템플릿

#### 생성형 AI의 실제 구현 고려사항
**성능 최적화**
- 캐싱: 자주 사용되는 응답과 프롬프트 캐싱
- 배치 처리: 여러 요청을 동시에 처리하여 효율성 향상
- 비동기 처리: 긴 응답 생성 시 스트리밍 방식으로 응답 제공
- 모델 선택: 요청의 복잡도에 따라 적절한 모델 동적 선택

**사용자 경험**
- 응답 속도: 사용자가 기다리는 시간 최소화
- 오류 처리: 오류 발생 시 우아한 처리와 복구 메커니즘
- 피드백 수집: 사용자 피드백을 통한 지속적 개선
- 접근성: 다양한 기기와 환경에서의 접근성 보장

**안전성과 윤리**
- 콘텐츠 필터링: 유해한 내용의 자동 감지와 차단
- 편향성 감지: 성별, 인종 등 편향적 표현 감지와 수정
- 투명성: AI가 생성했음을 명확하게 표시
- 제어 가능성: 사용자가 AI의 동작을 제어할 수 있는 옵션 제공

#### 생성형 AI의 사례 연구
**ChatGPT**
- 아키텍처: GPT 계열 모델 기반의 대화형 AI
- 특징: 자연스러운 대화, 다양한 지식 영역
- 영향: 생성형 AI의 대중화와 상업적 성공
- 한계: 환각, 편향성, 지식의 한계

**Claude**
- 아키텍처: Anthropic의 Constititional AI 기반
- 특징: 안전성 중시, 윤리적 가이드라인 내장
- 차별점: 헌법 기반의 자기 수정 메커니즘
- 영향: 안전한 AI 개발 방향성 제시

**Bard/Gemini**
- 아키텍처: Google의 LaMDA/Gemini 모델 기반
- 특징: 실시간 정보 검색, 다양한 모달리티 지원
- 통합: Google 검색과의 통합
- 영향: 검색 증강 생성의 대중화

### 3. 검색 증강 생성 (RAG) (25분)

#### RAG의 기본 원리
**정의와 목적**
- 정의: 검색된 정보를 활용한 생성형 AI
- 목적: 모델의 지식 한계를 외부 정보로 보완
- 특징: 동적 정보 업데이트, 사실 기반 응답, 출처 추적 가능

**RAG의 구성 요소**
- 검색기: 관련 정보를 검색하는 시스템
- 벡터 데이터베이스: 정보를 벡터로 저장하고 검색하는 시스템
- 재순위화: 검색된 정보의 중요도에 따른 순위 결정
- 생성: 검색된 정보를 바탕으로 한 응답 생성

**RAG의 동작 과정**
1. **질문 처리**: 사용자 질문을 벡터로 변환
2. **정보 검색**: 벡터 데이터베이스에서 관련 정보 검색
3. **정보 재순위화**: 검색된 정보의 관련성에 따른 순위 결정
4. **컨텍스트 구성**: 검색된 정보를 컨텍스트로 구성
5. **응답 생성**: 컨텍스트를 바탕으로 응답 생성
6. **출처 표시**: 참조된 정보의 출처 표시

#### 검색 기법
**키워드 검색**
- 원리: 질문의 키워드를 기반으로 문서 검색
- 장점: 구현 단순, 빠른 검색 속도
- 단점: 의미적 검색 한계, 오타에 취약
- 적용: 구조화된 데이터, 기술 문서

**의미 검색**
- 원리: 질문과 문서의 의미적 유사도 기반 검색
- 방법: 임베딩, 코사인 유사도, 벡터 공간
- 장점: 의미적 검색, 오타에 강건
- 단점: 계산 복잡도, 임베딩 품질 의존

**하이브리드 검색**
- 원리: 키워드 검색과 의미 검색의 결합
- 방법: 다단계 필터링, 결과 병합, 가중치 조절
- 장점: 두 방법의 장점 결합
- 단점: 구현 복잡성, 파라미터 튜닝 필요

#### 벡터 데이터베이스
**임베딩 기법**
- Word2Vec: 단어를 벡터로 표현
- GloVe: 단어의 동시 출현 확률 기반 임베딩
- BERT: 문맥을 고려한 단어 임베딩
- Sentence-BERT: 문장 수준의 임베딩

**벡터화 전략**
- 문서 분할: 문서를 의미 있는 단위로 분할
- 임베딩: 각 단위를 벡터로 변환
- 집계화: 문서 수준의 벡터 표현 (평균, 최대 등)
- 인덱싱: 빠른 검색을 위한 인덱스 구조

**유사도 측정**
- 코사인 유사도: $\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$
- 유클리드 거리: $\text{distance} = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$
- 맨해튼 거리: $\text{distance} = \sum_{i=1}^{n}|A_i - B_i|$
- 도트 프로덕트: $\text{similarity} = A \cdot B$

#### RAG의 고급 기법
**다단계 검색**
- 원리: 여러 단계에 걸쳐 정보를 정제하고 검색
- 과정: 초기 검색 → 재순위화 → 확장 검색 → 재검색
- 장점: 검색 정확도 향상, 관련성 높은 정보 확보
- 단점: 검색 시간 증가, 시스템 복잡성

**적응형 검색**
- 원리: 사용자 피드백을 기반으로 검색 전략 동적 조절
- 방법: 클릭률, 체류 시간, 피드백 점수
- 장점: 개인화된 검색 결과, 사용자 만족도 향상
- 단점: 구현 복잡성, 데이터 수집 필요

**다중 소스 검색**
- 원리: 여러 정보 소스에서 동시에 검색
- 소스: 내부 문서, 웹, 데이터베이스, API
- 통합: 여러 소스의 정보를 통합하여 응답 생성
- 장점: 정보의 다양성과 신뢰성 향상
- 단점: 통합의 복잡성, 일관성 유지 어려움

### 4. 멀티모달 LLM (20분)

#### 멀티모달 LLM의 기본 개념
**정의와 목적**
- 정의: 여러 모달리티(텍스트, 이미지, 오디오 등)를 이해하고 생성하는 모델
- 목적: 단일 모달리티를 넘어선 풍부한 이해와 생성 능력
- 특징: 모달리티 간의 정보 교환, 통합적 표현 학습

**주요 모달리티**
- 텍스트: 자연어, 코드, 수학적 표현
- 이미지: 사진, 그림, 차트, 다이어그램
- 오디오: 음성, 음악, 효과음
- 비디오: 동영상, 애니메이션, 실제 영상
- 기타: 3D 모델, 센서 데이터, 표

#### 멀티모달 아키텍처
**인코더 기반 아키텍처**
- 모달리티별 인코더: 각 모달리티를 위한 전용 인코더
- 공통 임베딩 공간: 모든 모달리티를 동일한 공간으로 투영
- 크로스 모달리티 어텐션: 모달리티 간의 관계 모델링
- 디코더: 통합된 표현에서 원하는 모달리티로 생성

**디코더 기반 아키텍처**
- 공통 인코더: 모든 모달리티를 위한 공통 인코더
- 모달리티별 디코더: 각 모달리티를 위한 전용 디코더
- 모달리티 전환: 한 모달리티에서 다른 모달리티로 변환
- 조건부 생성: 조건에 따른 특정 모달리티 생성

**하이브리드 아키텍처**
- 인코더-디코더 결합: 인코더와 디코더 기반의 결합
- 전처리-후처리: 모달리티별 전처리와 후처리
- 단계적 생성: 단계별로 다른 모달리티 생성
- 통합적 접근: 여러 아키텍처의 장점 결합

#### 멀티모달 학습 전략
**사전 훈련**
- 대규모 멀티모달 데이터셋: 웹, 책, 이미지-텍스트 쌍 등
- 모달리티별 사전 훈련: 각 모달리티에 대한 전문 모델 사전 훈련
- 통합 사전 훈련: 모든 모달리티를 통합한 모델 사전 훈련
- 전이 학습: 한 모달리티에서 학습한 지식을 다른 모달리티로 전이

**미세조정**
- 모달리티별 미세조정: 특정 모달리티에 대한 미세조정
- 통합 미세조정: 여러 모달리티를 통합한 미세조정
- 어댑터 기반 미세조정: 모달리티별 어댑터 추가
- 프롬프트 기반 미세조정: 모달리티별 프롬프트 설계

**데이터 증강**
- 모달리티 간 변환: 한 모달리티에서 다른 모달리티로 변환
- 합성 데이터 생성: 여러 모달리티를 결합한 합성 데이터 생성
- 노이즈 추가: 데이터의 다양성과 견고성 향상
- 균형 유지: 모달리티 간의 데이터 균형 유지

#### 멀티모달 LLM의 응용
**이미지 캡셔닝**
- 텍스트-이미지: 텍스트 설명을 기반으로 이미지 생성
- 이미지-텍스트: 이미지를 기반으로 텍스트 설명 생성
- 이미지-이미지: 이미지를 기반으로 유사한 이미지 생성
- 시각적 질문 답변: 이미지에 대한 질문에 텍스트로 답변

**비디오 생성**
- 텍스트-비디오: 텍스트 설명을 기반으로 비디오 생성
- 이미지-비디오: 이미지를 기반으로 비디오 생성
- 비디오-비디오: 비디오를 기반으로 유사한 비디오 생성
- 비디오 편집: 기존 비디오에 대한 텍스트 기반 편집

**오디오 생성**
- 텍스트-음성: 텍스트를 음성으로 변환 (TTS)
- 음악 생성: 텍스트 설명을 기반으로 음악 생성
- 음성-음성: 음성을 기반으로 유사한 음성 생성
- 오디오 효과: 텍스트 설명을 기반으로 오디오 효과 생성

**3D 콘텐츠 생성**
- 텍스트-3D: 텍스트 설명을 기반으로 3D 모델 생성
- 이미지-3D: 이미지를 기반으로 3D 모델 생성
- 3D-텍스트: 3D 모델을 기반으로 텍스트 설명 생성
- 3D 애니메이션: 3D 모델에 대한 애니메이션 생성

### 5. LLM 배포와 운영 (20분)

#### 배포 아키텍처
**클라우드 배포**
- 서버리스 컴퓨팅: AWS, Azure, GCP 등 클라우드 플랫폼
- 컨테이너화: Docker, Kubernetes를 이용한 컨테이너화
- 오토스케일링: 부하에 따른 자동 확장과 축소
- 글로벌 배포: 여러 지역에 분산된 배포

**엣지 배포**
- 온프레미스 배포: 자체 서버나 데이터센터에 배포
- 하이브리드 클라우드: 프라이빗 클라우드와 퍼블릭 클라우드의 결합
- CDN 통합: 콘텐츠 전송 네트워크와의 통합
- 지리적 분산: 사용자와의 가까운 곳에 서버 분산

**모바일 배포**
- 온디바이스 추론: 모바일 기기에서의 직접 추론
- 경량화 모델: 모바일 기기에 최적화된 경량화 모델
- 증분 추론: 모델의 일부만 다운로드하여 점진적 실행
- 푸시 추론: 중앙 서버와 모바일 기기의 협력 추론

#### 성능 최적화
**추론 최적화**
- 모델 압축: 양자화, 프루닝, 지식 증류
- 배치 처리: 여러 요청을 동시에 처리
- 캐싱: 자주 사용되는 결과 캐싱
- 동적 배치 크기: 부하에 따른 배치 크기 동적 조절

**하드웨어 가속**
- GPU 가속: CUDA, OpenCL 등을 이용한 GPU 가속
- TPU 가속: Google의 TPU를 이용한 가속
- 전용 하드웨어: 추론에 특화된 전용 하드웨어 (ASIC, FPGA)
- 혼합 정밀도: FP16, INT8 등 저정밀도 연산

**네트워크 최적화**
- 로드 밸런싱: 여러 서버 간의 부하 분산
- 커넥션 풀링: 불필요한 커넥션 재사용
- 압축: 요청과 응답의 압축
- 프로토콜 최적화: 효율적인 통신 프로토콜 사용

#### 운영 고려사항
**모니터링**
- 성능 모니터링: 응답 시간, 처리량, 오류율
- 자원 모니터링: CPU, GPU, 메모리, 네트워크 사용량
- 비용 모니터링: API 호출, 데이터 전송, 컴퓨팅 비용
- 사용자 행동 모니터링: 사용 패턴, 만족도, 이탈률

**보안**
- 인증: 사용자 인증, API 키 관리, 접근 제어
- 권한: 역할 기반 권한, 기능별 접근 제어
- 암호화: 데이터 전송 암호화, 저장 데이터 암호화
- 감사: 접근 로그, 이상 행동 탐지, 보안 이벤트

**유지보수**
- 버전 관리: 모델 버전 관리, 롤백, 롤포워드
- 패치 관리: 보안 패치, 기능 개선, 버그 수정
- 모델 재훈련: 새로운 데이터로 주기적 모델 재훈련
- 재해: 장애 발생 시의 빠른 복구와 서비스 재개

**비용 최적화**
- 오토스케일링: 사용량에 따른 자원 확장과 축소
- 스팟 인스턴스: 저렴한 스팟 인스턴스 활용
- 예약 인스턴스: 장기 계약을 통한 비용 할인
- 리소스 풀링: 사용하지 않는 리소스 자동 해제

## 실습 세션 (90분)

### 1. 생성형 AI 애플리케이션 구현 (30분)

#### 기본 챗봇 구현
```python
import openai
import os
from typing import List, Dict
import json
import time

class ChatBot:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def generate_response(self, user_message: str, system_prompt: str = None) -> str:
        """사용자 메시지에 대한 응답 생성"""
        
        # 대화 기록 구성
        messages = []
        
        # 시스템 프롬프트 추가
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 이전 대화 기록 추가
        messages.extend(self.conversation_history)
        
        # 현재 사용자 메시지 추가
        messages.append({"role": "user", "content": user_message})
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # 응답 추출
            assistant_message = response.choices[0].message.content
            
            # 대화 기록 업데이트
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # 대화 기록 길이 제한 (최근 10개 메시지)
            if len(self.conversation_history) > 20:  # 사용자와 어시스턴트 메시지 쌍
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_message
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
    
    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
    
    def save_conversation(self, filepath: str):
        """대화 기록 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

# 챗봇 사용 예시
def chat_example():
    """챗봇 사용 예시"""
    
    # API 키 설정 (실제로는 환경 변수에서 가져오기)
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    
    # 챗봇 초기화
    chatbot = ChatBot(api_key)
    
    # 시스템 프롬프트 설정
    system_prompt = "You are a helpful assistant. Provide clear and concise answers."
    
    # 대화 시작
    print("챗봇과의 대화를 시작합니다. '종료'를 입력하면 대화가 종료됩니다.")
    
    while True:
        # 사용자 입력 받기
        user_message = input("사용자: ")
        
        if user_message.lower() == '종료':
            print("대화를 종료합니다.")
            break
        
        # 응답 생성
        start_time = time.time()
        assistant_message = chatbot.generate_response(user_message, system_prompt)
        end_time = time.time()
        
        # 응답 출력
        print(f"어시스턴트: {assistant_message}")
        print(f"응답 시간: {end_time - start_time:.2f}초")
        print()
    
    # 대화 기록 저장
    chatbot.save_conversation("conversation_history.json")

# 챗봇 실행
if __name__ == "__main__":
    chat_example()
```

#### 프롬프트 엔지니어링 구현
```python
class PromptEngineer:
    def __init__(self):
        self.templates = {
            "qa": """
            질문: {question}
            관련 정보: {context}
            
            위 정보를 바탕으로 질문에 답변해주세요.
            """,
            
            "summarization": """
            원문: {text}
            
            위 원문을 다음 형식으로 요약해주세요:
            1. 핵심 내용
            2. 주요 세부 사항
            3. 결론
            """,
            
            "creative_writing": """
            주제: {topic}
            스타일: {style}
            길이: {length}
            
            위 주제와 스타일로 {length} 분량의 글을 작성해주세요.
            """
        }
    
    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """프롬프트 생성"""
        if template_name not in self.templates:
            raise ValueError(f"알 수 없는 템플릿: {template_name}")
        
        template = self.templates[template_name]
        return template.format(**kwargs)
    
    def add_template(self, name: str, template: str):
        """새로운 템플릿 추가"""
        self.templates[name] = template
    
    def few_shot_prompt(self, template_name: str, examples: List[Dict], **kwargs) -> str:
        """제로샷 프롬프트 생성"""
        base_prompt = self.generate_prompt(template_name, **kwargs)
        
        # 예시 추가
        examples_text = ""
        for i, example in enumerate(examples):
            examples_text += f"예시 {i+1}:\n"
            examples_text += f"입력: {example['input']}\n"
            examples_text += f"출력: {example['output']}\n\n"
        
        return f"{examples_text}\n이제 실제 질문에 답변해주세요:\n{base_prompt}"

# 프롬프트 엔지니어링 사용 예시
def prompt_engineering_example():
    """프롬프트 엔지니어링 사용 예시"""
    
    engineer = PromptEngineer()
    
    # QA 프롬프트 생성
    qa_prompt = engineer.generate_prompt(
        "qa",
        question="대한민국의 수도는 어디인가요?",
        context="대한민국의 수도는 서울특별시이다."
    )
    print("=== QA 프롬프트 ===")
    print(qa_prompt)
    print()
    
    # 요약 프롬프트 생성
    summary_prompt = engineer.generate_prompt(
        "summarization",
        text="인공지능(AI)은 기계가 인간의 지능을 모방하거나 초월하는 기술을 말한다. AI는 학습, 추론, 문제 해결, 언어 이해, 인식 등 다양한 인지 능력을 구현한다. 현대 AI는 머신러닝, 딥러닝, 자연어 처리 등 기술을 기반으로 발전하고 있으며, 음성 인식, 이미지 인식, 자율 주행 등 다양한 분야에서 활용되고 있다."
    )
    print("=== 요약 프롬프트 ===")
    print(summary_prompt)
    print()
    
    # 제로샷 프롬프트 생성
    examples = [
        {
            "input": "프랑스의 수도는 어디인가요?",
            "output": "프랑스의 수도는 파리입니다."
        },
        {
            "input": "일본의 수도는 어디인가요?",
            "output": "일본의 수도는 도쿄입니다."
        }
    ]
    
    few_shot_prompt = engineer.few_shot_prompt(
        "qa",
        examples=examples,
        question="독일의 수도는 어디인가요?"
    )
    print("=== 제로샷 프롬프트 ===")
    print(few_shot_prompt)

# 프롬프트 엔지니어링 실행
if __name__ == "__main__":
    prompt_engineering_example()
```

### 2. 검색 증강 생성(RAG) 구현 (30분)

#### 기본 RAG 시스템 구현
```python
import numpy as np
import faiss
import openai
from typing import List, Dict, Tuple
import json
import os

class VectorStore:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def add_documents(self, documents: List[Dict]):
        """문서를 벡터 저장소에 추가"""
        
        # 문서 저장
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # 문서 임베딩 (실제로는 임베딩 모델 사용)
        # 여기서는 간단한 랜덤 임베딩으로 대체
        embeddings = np.random.random((len(documents), self.dimension)).astype('float32')
        
        # 인덱스에 추가
        self.index.add_with_ids(embeddings, np.arange(start_idx, start_idx + len(documents)))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """유사한 문서 검색"""
        
        # 검색 수행
        distances, indices = self.index.search(query_embedding, k)
        
        # 결과 변환
        results = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'distance': float(distance),
                    'rank': i + 1
                })
        
        return results

class RAGSystem:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=api_key)
        self.embedding_model = model
        self.vector_store = VectorStore()
        self.generation_model = "gpt-3.5-turbo"
    
    def add_documents(self, documents: List[Dict]):
        """문서 추가"""
        self.vector_store.add_documents(documents)
    
    def embed_query(self, query: str) -> np.ndarray:
        """질문을 임베딩"""
        response = self.client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        return np.array(response['data'][0]['embedding'])
    
    def generate_response(self, query: str, k: int = 3) -> str:
        """RAG를 통한 응답 생성"""
        
        # 질문 임베딩
        query_embedding = self.embed_query(query)
        
        # 관련 문서 검색
        relevant_docs = self.vector_store.search(query_embedding, k)
        
        # 컨텍스트 구성
        context = "\n\n".join([doc['document']['text'] for doc in relevant_docs])
        
        # 프롬프트 구성
        prompt = f"""
        다음 정보를 바탕으로 질문에 답변해주세요:
        
        정보:
        {context}
        
        질문:
        {query}
        
        답변:
        """
        
        try:
            # GPT API 호출
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
    
    def save_vector_store(self, filepath: str):
        """벡터 저장소 저장"""
        data = {
            'documents': self.vector_store.documents,
            'index': self.vector_store.index.ntotal
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# RAG 시스템 사용 예시
def rag_example():
    """RAG 시스템 사용 예시"""
    
    # API 키 설정
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    
    # RAG 시스템 초기화
    rag = RAGSystem(api_key)
    
    # 문서 추가
    documents = [
        {
            'id': 1,
            'text': '서울특별시는 대한민국의 수도이다.',
            'source': '대한민국 정부'
        },
        {
            'id': 2,
            'text': '파리는 프랑스의 수도이다.',
            'source': '프랑스 정부'
        },
        {
            'id': 3,
            'text': '도쿄는 일본의 수도이다.',
            'source': '일본 정부'
        },
        {
            'id': 4,
            'text': '베를린은 독일의 수도이다.',
            'source': '독일 정부'
        }
    ]
    
    rag.add_documents(documents)
    
    # 질문-답변
    queries = [
        "대한민국의 수도는 어디인가요?",
        "유럽의 주요 수도들은 어디인가요?",
        "일본과 독일의 수도는 각각 어디인가요?"
    ]
    
    for query in queries:
        print(f"질문: {query}")
        
        # RAG를 통한 응답 생성
        response = rag.generate_response(query)
        
        print(f"응답: {response}")
        print()

# RAG 시스템 실행
if __name__ == "__main__":
    rag_example()
```

#### 고급 RAG 기법 구현
```python
class AdvancedRAGSystem(RAGSystem):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.query_expansion = True
        self.reranking = True
    
    def expand_query(self, query: str) -> List[str]:
        """질문 확장"""
        
        # 간단한 질문 확장 (실제로는 더 정교한 방법 사용)
        words = query.split()
        expanded_queries = [query]
        
        # 동의어 추가
        synonyms = {
            '수도': ['도시', '수도시'],
            '대한민국': ['한국', '남한'],
            '프랑스': ['프랑스 공화국'],
            '일본': ['일본국'],
            '독일': ['독일 연방 공화국']
        }
        
        for word in words:
            if word in synonyms:
                for synonym in synonyms[word]:
                    expanded_query = query.replace(word, synonym)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def multi_search(self, queries: List[str], k: int = 3) -> List[Dict]:
        """다중 질문 검색"""
        
        all_results = []
        
        for query in queries:
            query_embedding = self.embed_query(query)
            results = self.vector_store.search(query_embedding, k)
            
            for result in results:
                result['query'] = query
                all_results.append(result)
        
        # 거리 기반 재순위화
        all_results.sort(key=lambda x: x['distance'])
        
        # 중복 제거
        unique_results = []
        seen_docs = set()
        
        for result in all_results:
            doc_id = result['document']['id']
            if doc_id not in seen_docs:
                unique_results.append(result)
                seen_docs.add(doc_id)
        
        return unique_results[:k]
    
    def generate_response_with_reranking(self, query: str) -> str:
        """재순위화를 통한 응답 생성"""
        
        # 질문 확장
        expanded_queries = self.expand_query(query)
        
        # 다중 검색
        relevant_docs = self.multi_search(expanded_queries, k=5)
        
        # 재순위화 (간단한 점수 기반)
        for doc in relevant_docs:
            # 질문과 문서의 단어 중복도를 기반으로 점수 계산
            query_words = set(query.lower().split())
            doc_words = set(doc['document']['text'].lower().split())
            
            overlap = len(query_words & doc_words)
            doc['rerank_score'] = overlap / len(query_words)
        
        # 재순위화된 문서 선택
        relevant_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        top_docs = relevant_docs[:3]
        
        # 컨텍스트 구성
        context = "\n\n".join([f"{doc['document']['text']} (출처: {doc['document']['source']})" for doc in top_docs])
        
        # 프롬프트 구성
        prompt = f"""
        다음 정보를 바탕으로 질문에 답변해주세요. 각 정보의 출처를 명시해주세요.
        
        정보:
        {context}
        
        질문:
        {query}
        
        답변:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

# 고급 RAG 시스템 사용 예시
def advanced_rag_example():
    """고급 RAG 시스템 사용 예시"""
    
    # API 키 설정
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    
    # 고급 RAG 시스템 초기화
    advanced_rag = AdvancedRAGSystem(api_key)
    
    # 문서 추가
    documents = [
        {
            'id': 1,
            'text': '서울특별시는 대한민국의 수도이며, 인구는 약 970만 명이다.',
            'source': '대한민국 정부'
        },
        {
            'id': 2,
            'text': '파리는 프랑스의 수도이며, 인구는 약 220만 명이다.',
            'source': '프랑스 정부'
        },
        {
            'id': 3,
            'text': '도쿄는 일본의 수도이며, 인구는 약 1,400만 명이다.',
            'source': '일본 정부'
        },
        {
            'id': 4,
            'text': '베를린은 독일의 수도이며, 인구는 약 360만 명이다.',
            'source': '독일 정부'
        }
    ]
    
    advanced_rag.add_documents(documents)
    
    # 질문-답변
    query = "수도들의 인구에 대해 알려줘."
    
    print(f"질문: {query}")
    
    # 고급 RAG를 통한 응답 생성
    response = advanced_rag.generate_response_with_reranking(query)
    
    print(f"응답: {response}")
    print()

# 고급 RAG 시스템 실행
if __name__ == "__main__":
    advanced_rag_example()
```

### 3. 멀티모달 LLM 구현 (30분)

#### 기본 멀티모달 모델 구현
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import openai
import os
from typing import Dict, List, Optional

class MultiModalLLM(nn.Module):
    def __init__(self, text_model_name: str, image_model_name: str = None):
        super(MultiModalLLM, self).__init__()
        
        # 텍스트 인코더
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # 이미지 인코더 (간단한 CNN으로 대체)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 512),  # 224x224 이미지를 가정
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 128, 512),  # text_encoder의 hidden_size(768) + image_encoder의 output_size(128)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(128, self.text_tokenizer.vocab_size)
    
    def forward(self, text: str, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 텍스트 인코딩
        text_inputs = self.text_tokenizer(text, return_tensors='pt')
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        
        # 이미지 인코딩
        if image is not None:
            image_features = self.image_encoder(image)  # [batch_size, 128]
            # 텍스트와 이미지 특성 융합
            combined_features = torch.cat([text_features, image_features], dim=1)
        else:
            # 텍스트 특성만 사용
            combined_features = text_features
        
        # 융합 레이어 통과
        fused_features = self.fusion_layer(combined_features)
        
        # 출력 레이어 통과
        logits = self.output_layer(fused_features)
        
        return logits

# 멀티모달 모델 사용 예시
def multimodal_example():
    """멀티모달 모델 사용 예시"""
    
    # 모델 초기화
    model = MultiModalLLM(
        text_model_name="microsoft/DialoGPT-medium",
        image_model_name="resnet18"  # 실제로는 이미지 모델 사용
    )
    
    # 텍스트-이미지 캡션 생성 예시
    text = "A beautiful sunset over the mountains."
    
    # 가짜 이미지 텐서 (실제로는 이미지 로드)
    image = torch.randn(1, 3, 224, 224)  # [batch_size, channels, height, width]
    
    # 모델 순전파
    model.eval()
    with torch.no_grad():
        logits = model(text, image)
    
    # 결과 출력
    predicted_token_ids = torch.argmax(logits, dim=-1)
    predicted_text = model.text_tokenizer.decode(predicted_token_ids[0])
    
    print(f"입력 텍스트: {text}")
    print(f"생성된 캡션: {predicted_text}")

# 이미지 생성을 위한 DALL-E API 사용 예시
def image_generation_example():
    """이미지 생성 예시"""
    
    # API 키 설정
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    
    client = openai.OpenAI(api_key=api_key)
    
    # 이미지 생성
    response = client.images.generate(
        model="dall-e-3",
        prompt="A cute cat wearing a tiny wizard hat",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    
    # 생성된 이미지 URL
    image_url = response['data'][0]['url']
    
    print(f"생성된 이미지 URL: {image_url}")

# 멀티모달 예시 실행
if __name__ == "__main__":
    print("=== 멀티모달 모델 예시 ===")
    multimodal_example()
    
    print("\n=== 이미지 생성 예시 ===")
    image_generation_example()
```

## 과제

### 1. 생성형 AI 애플리케이션 과제
- 다양한 프롬프트 엔지니어링 기법 구현과 비교
- 사용자 피드백을 통한 모델 성능 개선 방안 연구
- 특정 도메인(의료, 법률 등)에 특화된 챗봇 구현

### 2. RAG 시스템 과제
- 다양한 검색 기법(키워드, 의미, 하이브리드) 구현과 비교
- 벡터 데이터베이스의 효율적 구현과 최적화
- 검색 결과의 재순위화와 품질 향상 방안 연구

### 3. 멀티모달 LLM 과제
- 텍스트-이미지, 텍스트-오디오 등 다양한 모달리티 조합 구현
- 모달리티 간의 융합 전략 비교와 분석
- 특정 멀티모달 응용(의료 영상 분석 등)에 특화된 모델 구현

## 추가 학습 자료

### 논문
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "FLAVA: A Foundational Model for Multimodal Understanding" (Singh et al., 2022)

### 온라인 자료
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [LangChain Documentation](https://python.langchain.com/en/latest/)
- [Vector Database Documentation](https://faiss.ai/)

### 구현 참고
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [ChromaDB](https://github.com/chroma-core/chroma)