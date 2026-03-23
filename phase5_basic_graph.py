# LangGraph 기초 — State, Node, Edge 이해
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# --- 1. State 정의 ---
# 그래프 전체가 공유하는 데이터 구조
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 메시지 누적
    question: str                             # 원본 질문
    answer: str                               # 최종 답변

# --- 2. Node 정의 ---
# 노드는 State를 받아서 업데이트된 State를 반환하는 함수
def analyze_question(state: AgentState) -> AgentState:
    """질문을 분석하는 노드"""
    print(f"[노드 1] 질문 분석 중: {state['question']}")
    return {
        "messages": [HumanMessage(content=state["question"])]
    }

def generate_answer(state: AgentState) -> AgentState:
    """LLM으로 답변을 생성하는 노드"""
    print(f"[노드 2] 답변 생성 중...")
    response = llm.invoke(state["messages"])
    print(f"[노드 2] 완료")
    return {
        "messages": [response],
        "answer": response.content
    }

# --- 3. 그래프 조립 ---
graph_builder = StateGraph(AgentState)

# 노드 등록
graph_builder.add_node("analyze", analyze_question)
graph_builder.add_node("generate", generate_answer)

# 엣지 연결 (순서 정의)
graph_builder.add_edge(START, "analyze")    # 시작 → analyze
graph_builder.add_edge("analyze", "generate")  # analyze → generate
graph_builder.add_edge("generate", END)     # generate → 종료

# 컴파일 (실행 가능한 객체로 변환)
graph = graph_builder.compile()

# --- 실행 ---
result = graph.invoke({
    "question": "LangGraph가 LangChain이랑 뭐가 달라?",
    "messages": [],
    "answer": ""
})

print(f"\n최종 답변: {result['answer']}")
print(f"메시지 수: {len(result['messages'])}개")