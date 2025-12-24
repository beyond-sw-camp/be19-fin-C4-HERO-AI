# File Name: main.py 
# Description: 파이썬 서버 실행 파일 

# history: 
# 2025/12/20 - 승민 최초 작성

# @author: 승민



import os
import uvicorn
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# 환경 변수 로드 함수
load_dotenv()

# OPENAI_API_KEY 데이터
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 파일 확인)")


# FastAPI 실행 앱
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


# LLM 모델 데이터
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=OPENAI_API_KEY,
)


# 요청 모델 타입
class FormItem(BaseModel):
    item_name: str
    score: float
    weight: float
    comment: str


class MemberEvaluation(BaseModel):
    template_name: str
    employee_name: str
    employee_department: str
    employee_grade: str
    total_score: float
    total_rank: str
    form_items: List[FormItem]


# 응답 모델 타입
class AnalysisResult(BaseModel):
    strengths: List[str]
    improvements: List[str]
    action_plan: List[str]

# AI의 출력을 Pydantic 모델(BaseModel)에 맞게 구조화된 데이터로 변환
output_parser = PydanticOutputParser(
    pydantic_object=AnalysisResult
)
format_instructions = output_parser.get_format_instructions()


# LLM에 보낼 프롬프트 설계
analysis_prompt = ChatPromptTemplate.from_template("""
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


# 사원의 평가 분석 POST 요청 API
@app.post("/api/analyze/member")
async def analyze_member(data: MemberEvaluation):
    # 평가 항목 문자열 생성
    items_text = "\n".join([
        f"- {item.item_name}: {item.score}점 (가중치 {item.weight}%) / 코멘트: {item.comment}"
        for item in data.form_items
    ])

    prompt = analysis_prompt.format(
        template_name=data.template_name,
        employee_name=data.employee_name,
        employee_department=data.employee_department,
        employee_grade=data.employee_grade,
        total_score=data.total_score,
        total_rank=data.total_rank,
        items=items_text,
        format_instructions=format_instructions,
    )

    # LLM 호출
    response = llm.invoke(prompt)

    # LangChain 방식으로 반환
    return output_parser.parse(response.content)


# 헬스 체크 GET 요청 API
@app.get("/")
def health_check():
    return {"status": "ok"}


# main.py 실행
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )