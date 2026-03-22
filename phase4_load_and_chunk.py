# 문서 로딩 + 청킹 실습
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 방식 1: TXT 파일 로딩 ---
# 테스트용 txt 파일 먼저 생성
sample_text = """
LangChain은 LLM 기반 애플리케이션을 만들기 위한 프레임워크입니다.
Chain, Memory, Tool, Agent 등의 핵심 개념을 제공합니다.

RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기법입니다.
외부 문서를 검색해서 LLM에 주입함으로써 더 정확한 답변을 생성합니다.
환각(hallucination) 문제를 줄이는 데 효과적입니다.

Qdrant는 벡터 검색에 특화된 데이터베이스입니다.
고성능 유사도 검색을 지원하며 Python 클라이언트를 제공합니다.
메모리 모드로 로컬에서 바로 실행할 수 있어 개발에 편리합니다.

LangGraph는 LangChain 기반의 그래프형 에이전트 프레임워크입니다.
노드와 엣지로 워크플로우를 정의하고 복잡한 에이전트를 구현할 수 있습니다.
조건 분기, 루프, 멀티 에이전트 협력이 가능합니다.
""" * 5  # 내용을 반복해서 문서를 길게 만들기

with open("sample_doc.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)

# # TXT 로딩
# loader = TextLoader("sample_doc.txt", encoding="utf-8")
# documents = loader.load()
# print(f"로딩된 문서 수: {len(documents)}")
# print(f"전체 텍스트 길이: {len(documents[0].page_content)} 글자")

# # --- 청킹 ---
# # RecursiveCharacterTextSplitter: 단락 → 문장 → 단어 순으로 자연스럽게 분리
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,       # 청크 최대 글자 수
#     chunk_overlap=30,     # 청크 간 겹치는 글자 수 (문맥 유지)
#     length_function=len,
# )

# chunks = splitter.split_documents(documents)
# print(f"\n청킹 결과: {len(chunks)}개 청크")
# print(f"chunk_size=200, chunk_overlap=30")

# print("\n=== 청크 샘플 3개 ===")
# for i, chunk in enumerate(chunks[:3]):
#     print(f"\n[청크 {i+1}] ({len(chunk.page_content)}글자)")
#     print(chunk.page_content)

# # chunk_overlap 효과 확인
# print("\n=== chunk_overlap 효과 ===")
# print(f"청크1 끝: ...{chunks[0].page_content[-30:]}")
# print(f"청크2 시작: {chunks[1].page_content[:30]}...")
# # 겹치는 부분이 보임 — 문맥이 잘리지 않게 해주는 역할