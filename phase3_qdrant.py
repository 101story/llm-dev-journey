# Qdrant 벡터 DB 실습
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 메모리 모드로 Qdrant 실행 (설치 없이 바로 사용 가능!)
qdrant = QdrantClient(":memory:")

COLLECTION_NAME = "tech_docs"
VECTOR_SIZE = 1536  # text-embedding-3-small 차원 수

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# --- 1단계: 컬렉션 생성 ---
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE  # 유사도 계산 방식
    )
)
print("컬렉션 생성 완료")

# --- 2단계: 문서 저장 ---
documents = [
    {"id": 1, "text": "LangChain은 LLM 기반 애플리케이션을 쉽게 만드는 프레임워크입니다."},
    {"id": 2, "text": "LangGraph는 그래프 구조로 AI 에이전트를 만드는 도구입니다."},
    {"id": 3, "text": "Qdrant는 벡터 검색에 특화된 고성능 데이터베이스입니다."},
    {"id": 4, "text": "RAG는 검색 증강 생성으로 LLM에 외부 지식을 주입하는 기법입니다."},
    {"id": 5, "text": "Python은 AI 개발에서 가장 많이 사용되는 프로그래밍 언어입니다."},
]

points = []
for doc in documents:
    vector = get_embedding(doc["text"])
    points.append(PointStruct(
        id=doc["id"],
        vector=vector,
        payload={"text": doc["text"]}  # 원본 텍스트도 함께 저장
    ))

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"{len(points)}개 문서 저장 완료")

# --- 3단계: 유사도 검색 ---
def search(query: str, top_k: int = 3):
    query_vector = get_embedding(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return results

print("\n=== 검색: 'AI 에이전트 개발 도구' ===")
for hit in search("AI 에이전트 개발 도구"):
    print(f"  [{hit.score:.4f}] {hit.payload['text']}")

print("\n=== 검색: '벡터 데이터베이스' ===")
for hit in search("벡터 데이터베이스"):
    print(f"  [{hit.score:.4f}] {hit.payload['text']}")

print("\n=== 검색: 'LLM에 지식 추가하는 방법' ===")
for hit in search("LLM에 지식 추가하는 방법"):
    print(f"  [{hit.score:.4f}] {hit.payload['text']}")