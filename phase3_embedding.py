# 임베딩 개념 실습 — 텍스트가 벡터로 어떻게 변환되는지
import os
from dotenv import load_dotenv
from openai import OpenAI
import math

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    """텍스트를 벡터로 변환"""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 가장 저렴한 임베딩 모델
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: list, vec2: list) -> float:
    """두 벡터의 코사인 유사도 계산 (1에 가까울수록 유사)"""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    return dot / (norm1 * norm2)

if __name__ == "__main__":
    # 임베딩 생성
    vec1 = get_embedding("LangChain은 LLM 프레임워크입니다")
    vec2 = get_embedding("LangGraph는 에이전트 개발 도구입니다")
    vec3 = get_embedding("오늘 점심은 김치찌개를 먹었습니다")

    print(f"벡터 차원 수: {len(vec1)}")  # 1536
    print(f"벡터 앞 5개 값: {vec1[:5]}")

    print("\n=== 유사도 비교 ===")
    sim_12 = cosine_similarity(vec1, vec2)
    sim_13 = cosine_similarity(vec1, vec3)
    print(f"LangChain ↔ LangGraph  유사도: {sim_12:.4f}")  # 높을 것
    print(f"LangChain ↔ 김치찌개   유사도: {sim_13:.4f}")  # 낮을 것