# Tool 실습 — 개념 이해용 (최신 버전)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# @tool 데코레이터로 함수를 도구로 만들기
@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 3 * 4'"""
    try:
        return str(eval(expression))
    except:
        return "계산 오류"

@tool
def get_tech_info(tech_name: str) -> str:
    """기술 스택에 대한 간단한 정보를 반환합니다."""
    tech_db = {
        "langchain": "LLM 애플리케이션 개발 프레임워크.",
        "qdrant": "고성능 벡터 데이터베이스.",
        "langgraph": "그래프형 AI 에이전트 프레임워크."
    }
    return tech_db.get(tech_name.lower(), "정보 없음")

# Tool을 LLM에 바인딩
tools = [calculate, get_tech_info]
llm_with_tools = llm.bind_tools(tools)

# 호출
response = llm_with_tools.invoke("qdrant, LangGraph가 뭐야? 그리고 2024 * 2 는 얼마야? 그리고 python에 대해서 설명해줘, 나 오늘 기분이 좋아")

print("LLM 응답:", response.content)
print("\nLLM이 호출하려는 Tool 목록:")
for tool_call in response.tool_calls:
    print(f"  - {tool_call['name']}({tool_call['args']})")

# 실제 Tool 실행
print("\nTool 실행 결과:")
for tool_call in response.tool_calls:
    if tool_call['name'] == 'calculate':
        print(f"  calculate → {calculate.invoke(tool_call['args'])}")
    elif tool_call['name'] == 'get_tech_info':
        print(f"  get_tech_info → {get_tech_info.invoke(tool_call['args'])}")
