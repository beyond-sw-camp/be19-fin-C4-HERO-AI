# File Name: analysis.py
# Description: 사원 평가 분석 로직
#
# history: 
# 2025/12/27 - 승민 최초 작성
#
# @author: 승민

from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 평가서 항목 타입
class FormItem(BaseModel):
    item_name: str
    score: float
    weight: float
    comment: str

# 사원 평가 타입
class MemberEvaluation(BaseModel):
    template_name: str
    employee_name: str
    employee_department: str
    employee_grade: str
    total_score: float
    total_rank: str
    form_items: List[FormItem]


# 사원 분석 결과 타입
class AnalysisResult(BaseModel):
    strengths: List[str]
    improvements: List[str]
    action_plan: List[str]


# AI의 출력을 Pydantic 모델(BaseModel)에 맞게 구조화된 데이터로 변환
member_output_parser = PydanticOutputParser(
    pydantic_object=AnalysisResult
)


# LLM에 보낼 프롬프트 설계
member_analysis_prompt = ChatPromptTemplate.from_template("""
너는 기업 인사팀에서 사용하는 평가 분석 AI야.

아래 사원 평가 데이터를 기반으로 분석해.

{format_instructions}

[평가 템플릿]
{template_name}

[사원 정보]
이름: {employee_name}
부서: {employee_department}
직급: {employee_grade}

[최종 점수]
{total_score}

[최종 등급]
{total_rank}

[세부 평가 항목]
{items}

조건:
- 반드시 JSON 형식으로만 응답
- 한국어
- 실무 인사평가 스타일
- 추상적인 말 금지
- 실행 가능한 피드백 중심
""")