# LangChain VectorStore로 Qdrant 사용하기
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# 임베딩 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Qdrant 메모리 모드
qdrant_client = QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# LangChain VectorStore 래퍼
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="docs",
    embedding=embeddings
)

# 문서 추가
docs = [
    "LangChain은 LLM 기반 애플리케이션을 쉽게 만드는 프레임워크입니다.",
    "LangGraph는 그래프 구조로 AI 에이전트를 만드는 도구입니다.",
    "Qdrant는 벡터 검색에 특화된 고성능 데이터베이스입니다.",
    "RAG는 검색 증강 생성으로 LLM에 외부 지식을 주입하는 기법입니다.",
    "Python은 AI 개발에서 가장 많이 사용되는 프로그래밍 언어입니다.",
]
vectorstore.add_texts(docs)
print(f"{len(docs)}개 문서 저장 완료")

# Retriever로 변환 — 이게 Phase 4 RAG의 핵심 부품이에요
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 검색 테스트
query = "에이전트 개발에 쓰는 도구가 뭐야?"
results = retriever.invoke(query)

print(f"\n=== 쿼리: '{query}' ===")
for doc in results:
    print(f"  - {doc.page_content}")