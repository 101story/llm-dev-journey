# 대화 히스토리 유지 + 스트리밍 실습
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ConversationBot:
    """대화 히스토리를 유지하는 챗봇"""
    
    def __init__(self, system_prompt: str = "당신은 친절한 AI 어시스턴트입니다."):
        self.messages = []
        self.messages.append(
            {"role": "system", "content": system_prompt}
        )
    
    def chat(self, user_input: str) -> str:
        # 사용자 메시지 추가
        self.messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        
        # 어시스턴트 응답도 히스토리에 추가 (다음 대화에 기억)
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def stream_chat(self, user_input: str):
        """응답을 스트리밍으로 출력 (ChatGPT처럼 글자가 하나씩 나오는 효과)"""
        self.messages.append({"role": "user", "content": user_input})
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            stream=True  # 핵심!
        )
        
        full_response = ""
        print("AI: ", end="", flush=True)
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_response += text
        
        print()  # 줄바꿈
        self.messages.append({"role": "assistant", "content": full_response})
    
    def show_history(self):
        """현재 대화 히스토리 출력"""
        print("\n--- 대화 히스토리 ---")
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"[{role}]: {content}")


if __name__ == "__main__":
    bot = ConversationBot(system_prompt="당신은 Python 전문가입니다. 간결하게 답변해주세요.")
    
    print("=== 일반 대화 (히스토리 유지) ===")
    print("AI:", bot.chat("리스트 컴프리헨션이 뭐야?"))
    print("AI:", bot.chat("방금 설명한 거 예시 코드로 보여줘"))  # 이전 대화를 기억!
    
    print("\n=== 스트리밍 대화 ===")
    bot2 = ConversationBot()
    bot2.stream_chat("FastAPI가 뭔지 3줄로 설명해줘")
    
    bot2.show_history()
    print("Bot ID:", id(bot))
    print("Bot2 ID:", id(bot2))
