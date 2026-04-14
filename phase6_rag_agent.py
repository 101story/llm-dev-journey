# Phase 6 — RAG + LangGraph 통합 에이전트
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ──────────────────────────────────────
# 1. 벡터DB 초기화 (Phase 3~4에서 배운 것)
# ──────────────────────────────────────
def build_vectorstore():
    loader = TextLoader("sample_doc.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="llm_docs",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="llm_docs",
        embedding=embeddings
    )
    vectorstore.add_documents(chunks)
    print(f"벡터DB 준비 완료: {len(chunks)}개 청크 저장")
    return vectorstore

# ──────────────────────────────────────
# 2. State 정의 (Phase 5에서 배운 것)
# ──────────────────────────────────────
class AgentState(TypedDict):
    question: str
    needs_rag: bool          # RAG 필요 여부
    context: str             # 검색된 문서 내용
    answer: str              # 생성된 답변
    source_docs: list        # 참고 문서 원본
    quality_pass: bool       # 품질 통과 여부
    retry_count: int         # 재시도 횟수

# ──────────────────────────────────────
# 3. 노드 정의
# ──────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def analyze_question(state: AgentState, vectorstore) -> AgentState:
    """질문을 분석해서 RAG가 필요한지 판단하는 노드"""
    prompt = f"""다음 질문이 문서 검색이 필요한지 판단하세요.
LangChain, RAG, Qdrant, LangGraph, 임베딩, 벡터DB 등 학습 내용에 관한 질문이면 'rag'
일상적인 질문이나 문서와 무관한 질문이면 'general'
반드시 rag 또는 general 중 하나만 답하세요.

질문: {state['question']}"""

    response = llm.invoke(prompt)
    needs_rag = "rag" in response.content.lower()
    print(f"[분석] '{state['question']}' → {'RAG 검색' if needs_rag else '일반 답변'}")
    return {"needs_rag": needs_rag}

def rag_search(state: AgentState, vectorstore) -> AgentState:
    """Qdrant에서 관련 문서를 검색하는 노드"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"[RAG 검색] {len(docs)}개 문서 검색됨")
    return {"context": context, "source_docs": docs}

def generate_rag_answer(state: AgentState) -> AgentState:
    """검색된 문서를 기반으로 답변을 생성하는 노드"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """아래 컨텍스트만 사용해서 질문에 답하세요.
컨텍스트에 없는 내용은 '문서에 해당 정보가 없습니다'라고 답하세요.

컨텍스트:
{context}"""),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": state["context"], "question": state["question"]})
    print(f"[RAG 답변 생성] 완료")
    return {"answer": answer}

def generate_general_answer(state: AgentState) -> AgentState:
    """RAG 없이 직접 답변하는 노드"""
    response = llm.invoke(state["question"])
    print(f"[일반 답변 생성] 완료")
    return {"answer": response.content, "context": "", "source_docs": []}

def evaluate_quality(state: AgentState) -> AgentState:
    """답변 품질을 평가하는 노드"""
    prompt = f"""다음 답변이 질문에 충분히 답했는지 평가하세요.
구체적인 내용이 있고 100자 이상이면 'pass', 아니면 'retry'라고만 답하세요.

질문: {state['question']}
답변: {state['answer']}"""

    response = llm.invoke(prompt)
    quality_pass = "pass" in response.content.lower()
    retry_count = state.get("retry_count", 0) + 1
    print(f"[품질 평가] {'통과' if quality_pass else '재시도'} ({retry_count}회차)")
    return {"quality_pass": quality_pass, "retry_count": retry_count}

# ──────────────────────────────────────
# 4. 조건 함수 (엣지 분기 결정)
# ──────────────────────────────────────
def route_by_rag(state: AgentState) -> str:
    return "rag_search" if state["needs_rag"] else "generate_general"

def route_by_quality(state: AgentState) -> str:
    # 통과했거나 3회 초과하면 종료
    if state["quality_pass"] or state.get("retry_count", 0) >= 3:
        return "end"
    return "analyze"  # 재시도 시 처음부터

# ──────────────────────────────────────
# 5. 그래프 조립
# ──────────────────────────────────────
def build_agent(vectorstore):
    # 노드 함수에 vectorstore 주입 (클로저 활용)
    def _analyze(state): return analyze_question(state, vectorstore)
    def _rag_search(state): return rag_search(state, vectorstore)

    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("analyze", _analyze)
    graph.add_node("rag_search", _rag_search)
    graph.add_node("generate_rag", generate_rag_answer)
    graph.add_node("generate_general", generate_general_answer)
    graph.add_node("evaluate", evaluate_quality)

    # 엣지 연결
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route_by_rag, {
        "rag_search": "rag_search",
        "generate_general": "generate_general"
    })
    graph.add_edge("rag_search", "generate_rag")
    graph.add_edge("generate_rag", "evaluate")
    graph.add_edge("generate_general", "evaluate")
    graph.add_conditional_edges("evaluate", route_by_quality, {
        "end": END,
        "analyze": "analyze"
    })

    return graph.compile()

# ──────────────────────────────────────
# 6. 실행
# ──────────────────────────────────────
if __name__ == "__main__":
    vectorstore = build_vectorstore()
    agent = build_agent(vectorstore)

    questions = [
        "RAG가 뭐야? 왜 쓰는 거야?",
        "LangGraph의 State, Node, Edge가 각각 뭐야?",
        "Qdrant를 메모리 모드로 쓰면 어떤 장점이 있어?",
        "오늘 점심 뭐 먹지?",  # 문서와 무관 — 일반 답변으로 처리
    ]

    print("\n" + "="*50)
    for q in questions:
        print(f"\n질문: {q}")
        result = agent.invoke({
            "question": q,
            "needs_rag": False,
            "context": "",
            "answer": "",
            "source_docs": [],
            "quality_pass": False,
            "retry_count": 0
        })
        print(f"답변: {result['answer'][:200]}...")
        if result["source_docs"]:
            print(f"참고 문서: {len(result['source_docs'])}개")
        print("-" * 50)