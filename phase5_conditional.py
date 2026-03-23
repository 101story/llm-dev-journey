# 조건 엣지 — 질문 유형에 따라 다른 경로로 분기
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    question: str
    category: str   # "tech" | "math" | "other"
    answer: str

# --- 노드들 ---
def classify_question(state: AgentState) -> AgentState:
    """질문을 분류하는 노드"""
    prompt = f"""다음 질문을 분류하세요. 반드시 tech/math/other 중 하나만 답하세요.
질문: {state['question']}"""
    
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    # 혹시 모를 오답 처리
    if category not in ["tech", "math", "other"]:
        category = "other"
    
    print(f"[분류] {state['question']} → {category}")
    return {"category": category}

def handle_tech(state: AgentState) -> AgentState:
    """기술 질문 처리 노드"""
    response = llm.invoke(f"기술 전문가로서 답변해줘: {state['question']}")
    return {"answer": f"[기술 답변] {response.content}"}

def handle_math(state: AgentState) -> AgentState:
    """수학 질문 처리 노드"""
    response = llm.invoke(f"수학 선생님으로서 단계별로 풀어줘: {state['question']}")
    return {"answer": f"[수학 답변] {response.content}"}

def handle_other(state: AgentState) -> AgentState:
    """기타 질문 처리 노드"""
    response = llm.invoke(f"친절하게 답변해줘: {state['question']}")
    return {"answer": f"[일반 답변] {response.content}"}

# --- 조건 함수 ---
# 이 함수의 반환값이 다음 노드 이름이 됨
def route_question(state: AgentState) -> Literal["tech", "math", "other"]:
    return state["category"]

# --- 그래프 조립 ---
graph_builder = StateGraph(AgentState)

graph_builder.add_node("classify", classify_question)
graph_builder.add_node("tech", handle_tech)
graph_builder.add_node("math", handle_math)
graph_builder.add_node("other", handle_other)

graph_builder.add_edge(START, "classify")

# 조건 엣지: classify 이후 route_question 결과에 따라 분기
graph_builder.add_conditional_edges(
    "classify",           # 어떤 노드 다음에
    route_question,       # 이 함수 결과로 판단해서
    {                     # 이 매핑으로 다음 노드 결정
        "tech": "tech",
        "math": "math",
        "other": "other"
    }
)

graph_builder.add_edge("tech", END)
graph_builder.add_edge("math", END)
graph_builder.add_edge("other", END)

graph = graph_builder.compile()

# --- 실행 ---
questions = [
    "Qdrant랑 Milvus 차이가 뭐야?",
    "피타고라스 정리로 빗변 길이 구하는 법 알려줘",
    "오늘 점심 뭐 먹지?"
]

for q in questions:
    result = graph.invoke({"question": q, "category": "", "answer": ""})
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:100]}...")
    print("-" * 50)