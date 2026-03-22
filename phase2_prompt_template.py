# PromptTemplate 실습
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 프롬프트 템플릿 정의 — {변수} 자리에 나중에 값을 채워넣음
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다. {style} 스타일로 답변하세요."),
    ("human", "{question}")
])

# Chain 연결: prompt → llm
chain = prompt | llm

# 변수만 바꿔서 재사용
response1 = chain.invoke({
    "role": "Python",
    "style": "초보자도 이해할 수 있는 쉬운",
    "question": "데코레이터가 뭐야?"
})

response2 = chain.invoke({
    "role": "데이터베이스",
    "style": "핵심만 간결하게 말하는",
    "question": "인덱스가 왜 빠른가요?"
})

print("=== 응답 1 ===")
print(response1.content)

print("\n=== 응답 2 ===")
print(response2.content)