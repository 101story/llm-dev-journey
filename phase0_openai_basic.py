# OpenAI API 기본 호출 실습
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_gpt(user_message: str, model: str = "gpt-4o-mini") -> str:
    """GPT 모델에 메시지를 보내고 응답을 받는 함수"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

# 실행
if __name__ == "__main__":
    # 기본 호출
    answer = chat_with_gpt("LangChain이 뭔지 한 문장으로 설명해줘")
    print("GPT 응답:", answer)
    
    # 토큰 사용량도 확인해보기
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "안녕!"}]
    )
    print("\n--- 토큰 사용량 ---")
    print(f"입력 토큰: {response.usage.prompt_tokens}")
    print(f"출력 토큰: {response.usage.completion_tokens}")
    print(f"총 토큰: {response.usage.total_tokens}")
