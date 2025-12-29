# File Name: promotion.py
# Description: 승진 추천 분석 로직
#
# history: 
# 2025/12/27 - 승민 최초 작성
#
# @author: 승민

from typing import List, Dict
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 승진 추천 대상자 타입 
class PromotionCandidate(BaseModel):
    name: str
    department: str
    current_grade: str
    recommended_grade: str
    growth_rate: float
    core_competencies: List[str]
    reason: str

# 승진 추천 대상자 결과 데이터 타입
class PromotionLLMResult(BaseModel):
    core_competencies: List[str]
    reason: str

# AI의 출력을 Pydantic 모델(BaseModel)에 맞게 구조화된 데이터로 변환
promotion_output_parser = PydanticOutputParser(
    pydantic_object=PromotionLLMResult
)

# 직급 순서 데이터
GRADE_ORDER = ["사원", "주임", "대리", "과장", "차장", "부장"]

# 추천 직급 계산 함수
def recommend_grade(current: str) -> str:
    if current not in GRADE_ORDER:
        return current
    idx = GRADE_ORDER.index(current)
    return GRADE_ORDER[idx + 1] if idx + 1 < len(GRADE_ORDER) else current


# 승진 추천 대상자 추출 함수
def extract_top_candidates(dashboard_data: List[Dict]) -> List[Dict]:
    """
    가장 최근 템플릿과 그 직전 템플릿을 비교하여
    전 분기 대비 성장률이 높은 상위 3명을 추출
    """

    # 평가 템플릿 정렬
    templates = sorted(
        dashboard_data,
        key=lambda x: x["evaluationTemplateId"]
    )

    if len(templates) < 2:
        return []

    # 가장 최근 템플릿 2개를 사용 
    prev_template = templates[-2]
    curr_template = templates[-1]

    # 이전 분기 점수 맵
    prev_scores = {}
    for ev in prev_template["evaluations"]:
        for e in ev["evaluatees"]:
            prev_scores[e["evaluationEvaluateeId"]] = e["evaluationEvaluateeTotalScore"]

    growth_candidates = []

    # 이전 분기 점수와 현재 분기 점수 비교 계산
    for ev in curr_template["evaluations"]:
        for e in ev["evaluatees"]:
            pid = e["evaluationEvaluateeId"]

            if (
                pid not in prev_scores
                or prev_scores[pid] is None
                or e["evaluationEvaluateeTotalScore"] is None
            ):
                continue

            prev_score = prev_scores[pid]
            curr_score = e["evaluationEvaluateeTotalScore"]

            if prev_score == 0:
                continue

            growth_rate = ((curr_score - prev_score) / prev_score) * 100

            growth_candidates.append({
                "name": e["evaluationEvaluateeName"],
                "department": e["evaluationEvaluateeDepartmentName"],
                "grade": e["evaluationEvaluateeGrade"],
                "growth": round(growth_rate, 1),
                "formItems": e["formItems"]
            })

    return sorted(
        growth_candidates,
        key=lambda x: x["growth"],
        reverse=True
    )[:3]


# LLM에 보낼 프롬프트 설계
promotion_prompt = ChatPromptTemplate.from_template("""
너는 기업 인사팀의 승진 심사 AI야.

아래 사원 정보를 기반으로
- 핵심 역량 (1~3개)
- 승진 추천 사유
를 작성해.

{format_instructions}

[사원 정보]
이름: {name}
부서: {department}
현재 직급: {grade}
전 분기 대비 성장률: {growth_rate}%

[평가 항목 점수]
{items}

조건:
- 반드시 JSON 형식으로만 응답
- 한국어
- 실무 인사평가 스타일
- 추상적인 표현 금지
""")