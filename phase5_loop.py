# 루프 — 품질 검증 후 통과 못하면 재생성
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

class AgentState(TypedDict):
    question: str
    answer: str
    retry_count: int    # 재시도 횟수
    is_good: bool       # 품질 통과 여부

def generate_answer(state: AgentState) -> AgentState:
    """답변 생성 노드"""
    count = state["retry_count"]
    print(f"[답변 생성] 시도 {count + 1}회")
    
    response = llm.invoke(state["question"])
    print(f"""[답변 생성] 완료 
미리보기: {response.content[:50]}...
길이: {len(response.content)}자""")
    return {
        "answer": response.content,
        "retry_count": count + 1
    }

def evaluate_answer(state: AgentState) -> AgentState:
    """답변 품질 평가 노드"""
    eval_prompt = f"""다음 답변이 충분히 상세한지 평가하세요. 
답변이 800자가 넘으면 'good', 못 넘으면 'retry' 라고만 답하세요.

답변: {state['answer']}"""
    
    result = llm.invoke(eval_prompt)
    is_good = "good" in result.content.lower()
    print(f"[품질 평가] {'통과' if is_good else '재시도 필요'} (현재 {len(state['answer'])}자)")
    return {"is_good": is_good}

def should_retry(state: AgentState) -> Literal["generate", "end"]:
    """재시도 여부 결정 — 최대 3회"""
    if state["is_good"] or state["retry_count"] >= 3:
        return "end"
    return "generate"

# --- 그래프 조립 ---
graph_builder = StateGraph(AgentState)

graph_builder.add_node("generate", generate_answer)
graph_builder.add_node("evaluate", evaluate_answer)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", "evaluate")

# evaluate 이후 루프 or 종료
graph_builder.add_conditional_edges(
    "evaluate",
    should_retry,
    {"generate": "generate", "end": END}
)

graph = graph_builder.compile()

result = graph.invoke({
    "question": "RAG 시스템의 장단점을 설명해줘",
    "answer": "",
    "retry_count": 0,
    "is_good": False
})

print(f"\n최종 답변 ({result['retry_count']}회 시도):")
print(result["answer"])