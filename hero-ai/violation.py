# File Name: violation.py
# Description: 평가 가이드 위반 로직
#
# history: 
# 2025/12/27 - 승민 최초 작성
#
# @author: 승민

from typing import List, Dict
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import json
import re

# 가이드 위반 데이터 타입
class GuideViolation(BaseModel):
    evaluationTemplateId: int
    evaluationTemplateName: str
    managerName: str
    departmentName: str
    violations: List[dict]

# 평가 가이드 벡터화시키는 함수
def build_guide_vectorstore(guide_text: str):
    # 텍스트 분할: 가이드 내용을 400자 단위로 쪼갬, 50자 겹침(문맥 손실 방지)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    docs = splitter.split_text(guide_text)

    # 가이드 내용 조각을 벡터로 변환
    embeddings = OpenAIEmbeddings()

    # 가이드 내용 조각을 벡터 DB에 저장
    return FAISS.from_texts(docs, embeddings)

# LLM 모델에서 출력되는 값을 안정화 시키는 함수
def clean_llm_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
    return text


# 프롬프트
violation_prompt = ChatPromptTemplate.from_template("""
너는 기업 인사팀의 평가 감사 AI다.
너의 역할은 '위반 여부를 판단'하는 것이다.
설명이나 요약은 하지 않는다.

아래 [고정 등급 기준]과 [평가 가이드 발췌],
그리고 [평가 데이터]를 비교하여
**명백한 평가 가이드 위반만 식별하라.**

[고정 등급 기준]
- S: 81~100
- A: 61~80
- B: 41~60
- C: 21~40
- F: 0~20

위반으로 판단하는 경우는 아래뿐이다:
1. 점수가 해당 등급의 점수 범위를 벗어난 경우
2. 실적 내용과 점수/등급이 명백히 불일치하는 경우
3. 평가 코멘트가 실적과 논리적으로 연결되지 않는 경우

❗ 아래 조건은 **절대 위반이 아니다**:
- 점수와 등급이 기준에 정확히 부합하는 경우
- 실적·점수·등급·코멘트가 서로 일관된 경우

[평가 가이드 발췌]
{guide_context}

[평가 데이터]
{evaluation_data}

출력 규칙:
- 위반이 하나도 없으면 정확히 다음 문장만 출력:
  "위반 없음"
- 위반이 있는 경우에만 JSON 배열 출력
- **위반이 아닌 항목은 JSON에 절대 포함하지 말 것**
- 설명용 문장, 판단 요약, 중간 해설 절대 금지

JSON 형식:
[
  {{
    "피평가자": "...",
    "항목": "...",
    "위반 사유": "..."
  }}
]
""")


# 평가 가이드 위반 분석 함수
def analyze_guide_violations(
    template: Dict,
    guide_text: str,
    llm: ChatOpenAI
) -> List[GuideViolation]:

    # 평가 가이드 텍스트를 벡터 DB로 변환
    vectorstore = build_guide_vectorstore(guide_text)

    # 벡터 DB를 검색기로 변환 (질문을 주면 관련 문서 조각을 찾아줌, 가장 유사한 조각 3개를 찾아줌)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    results: List[GuideViolation] = []

    for ev in template["evaluations"]:
        evaluation_lines = []

        # 평가 데이터 텍스트화
        for e in ev["evaluatees"]:
            for item in e["formItems"]:
                evaluation_lines.append(
                    f"""
                    피평가자: {e['evaluationEvaluateeName']}
                    항목: {item['formItemName']}
                    실적: {item.get('formItemEvaluateePerformance')}
                    점수: {item.get('formItemScore')}
                    등급: {item.get('formItemRank')}
                    코멘트: {item.get('formItemComment')}
                    """
                )

        evaluation_text = "\n".join(evaluation_lines)

        # RAG 검색
        guide_docs = retriever.invoke(evaluation_text)

        guide_context = "\n".join(
            d.page_content for d in guide_docs
        )

        prompt = violation_prompt.format(
            guide_context=guide_context,
            evaluation_data=evaluation_text
        )

        #LLM 모델 사용
        response = llm.invoke(prompt).content.strip()

        # 정상 평가 필터링 
        if response == "위반 없음":
            continue

        cleaned = clean_llm_json(response)

        # JSON이 아닌 출력 필터링
        if not cleaned.startswith("["):
            continue

        try:
            parsed_violations = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed_violations = [{
                "피평가자": "알 수 없음",
                "항목": "알 수 없음",
                "위반 사유": cleaned
            }]

        # 리스트 필터링
        if not isinstance(parsed_violations, list):
            continue

        # 최종 결과 배열에 데이터 추가
        results.append(
            GuideViolation(
                evaluationTemplateId=template["evaluationTemplateId"],
                evaluationTemplateName=template["evaluationTemplateName"],
                managerName=ev["evaluationManagerName"],
                departmentName=ev["evaluationDepartmentName"],
                violations=parsed_violations
            )
        )

    return results