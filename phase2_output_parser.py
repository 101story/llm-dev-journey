# OutputParser 실습 (최신 버전)
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate          
# langchain → langchain_core
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser  # 동일
from pydantic import BaseModel, Field                          
# langchain_core.pydantic_v1 → pydantic

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 방식 1: 리스트로 받기 ---
print("=== 리스트 파싱 ===")
list_parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("human", "Python 라이브러리 5개를 추천해줘.\n{format_instructions}"),
])
prompt = prompt.partial(format_instructions=list_parser.get_format_instructions())

chain = prompt | llm | list_parser
result = chain.invoke({})
print(type(result))
print(result)

# --- 방식 2: Pydantic 구조체로 받기 ---
print("\n=== 구조체 파싱 ===")

class TechReview(BaseModel):
    name: str = Field(description="기술 이름")
    difficulty: str = Field(description="난이도: 쉬움/보통/어려움")
    use_case: str = Field(description="주요 사용 사례")
    rating: int = Field(description="추천 점수 1~10")

json_parser = JsonOutputParser(pydantic_object=TechReview)

prompt2 = ChatPromptTemplate.from_messages([
    ("human", "LangChain에 대해 평가해줘.\n{format_instructions}")
])
prompt2 = prompt2.partial(format_instructions=json_parser.get_format_instructions())

chain2 = prompt2 | llm | json_parser
result2 = chain2.invoke({})
print(type(result2))
print(result2)
print(f"\n난이도: {result2['difficulty']}, 점수: {result2['rating']}/10")