"""
### 흐름 정리

HumanMessage("질문")
        ↓
   LLM 1차 호출
        ↓
  tool_calls 반환 (content 비어있음)
        ↓
  Tool 직접 실행
        ↓
  ToolMessage(결과) 추가
        ↓
   LLM 2차 호출 (Tool 결과 포함)
        ↓
  최종 답변 (content 채워짐)
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
        "langgraph": "그래프형 AI 에이전트 프레임워크.",
    }
    return tech_db.get(tech_name.lower(), "정보 없음")

tools = [calculate, get_tech_info]
tools_map = {t.name: t for t in tools}  # 이름으로 빠르게 찾기 위한 딕셔너리

llm_with_tools = llm.bind_tools(tools)

def run_agent(user_input: str):
    messages = [HumanMessage(content=user_input)]
    
    # --- 1단계: LLM 첫 번째 호출 ---
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    print("=== 1단계: LLM 판단 ===")
    print(f"content: '{response.content}'")  # Tool 필요하면 보통 비어있음
    print(f"tool_calls: {[t['name'] for t in response.tool_calls]}")
    
    # --- 2단계: Tool 실행 후 결과를 messages에 추가 ---
    if response.tool_calls:
        print("\n=== 2단계: Tool 실행 ===")
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_result = tools_map[tool_name].invoke(tool_call['args'])
            
            print(f"  {tool_name}({tool_call['args']}) → {tool_result}")
            
            # ToolMessage로 결과를 히스토리에 추가 (tool_call_id 연결 필수!)
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call['id']
            ))
    
    # --- 3단계: Tool 결과 포함해서 LLM 최종 호출 ---
    print("\n=== 3단계: 최종 답변 ===")
    final_response = llm_with_tools.invoke(messages)
    print(final_response.content)
    
    return final_response

run_agent("qdrant, LangGraph가 뭐야? 그리고 2024 * 2 는 얼마야? 그리고 python에 대해서 설명해줘, 나 오늘 기분이 좋아")
