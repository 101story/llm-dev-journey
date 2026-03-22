# temperature, top_p, system prompt가 결과에 미치는 영향 실험
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def compare_temperatures(prompt: str):
    """temperature 값에 따라 응답이 얼마나 달라지는지 비교"""
    
    for temp in [0.0, 0.7, 1.5]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=80
        )
        print(f"[temperature={temp}] {response.choices[0].message.content}")
        print()

def compare_system_prompts(user_message: str):
    """system prompt가 다르면 같은 질문도 다르게 답함"""
    
    personas = [
        ("친절한 선생님", "당신은 초등학생도 이해할 수 있게 쉽게 설명하는 선생님입니다."),
        ("냉철한 엔지니어", "당신은 감정 없이 기술적 사실만 간결하게 말하는 시니어 엔지니어입니다."),
        ("유머러스한 친구", "당신은 모든 것을 재미있는 비유와 농담으로 설명하는 친구입니다.")
    ]
    
    for name, system_prompt in personas:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=100
        )
        print(f"[{name}]")
        print(response.choices[0].message.content)
        print()


if __name__ == "__main__":
    print("=== Temperature 비교 (창의성 조절) ===")
    compare_temperatures("AI 개발자의 하루를 한 문장으로 묘사해줘")
    
    print("\n=== System Prompt 비교 (페르소나 변경) ===")
    compare_system_prompts("벡터 데이터베이스가 뭐야?")