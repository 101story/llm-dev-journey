# LangChain Memory 실습 (최신 버전)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 최신 LangChain은 메모리를 직접 messages 리스트로 관리해요
class ConversationBot:
    def __init__(self):
        self.history = []  # 대화 히스토리 직접 관리
    
    def chat(self, user_input: str) -> str:
        # 히스토리 포함해서 프롬프트 구성
        messages = [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            *[(msg["role"], msg["content"]) for msg in self.history],
            ("human", user_input)
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        response = chain.invoke({})
        
        # 히스토리에 저장
        self.history.append({"role": "human", "content": user_input})
        self.history.append({"role": "assistant", "content": response.content})
        
        return response.content
    
    def show_history(self):
        print("\n--- 대화 히스토리 ---")
        for msg in self.history:
            preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"[{msg['role']}]: {preview}")


if __name__ == "__main__":
    bot = ConversationBot()
    
    print("=== 대화 히스토리 유지 테스트 ===")
    print("AI:", bot.chat("내 이름은 민준이야"))
    print("AI:", bot.chat("나는 백엔드 개발자야"))
    print("AI:", bot.chat("내가 누구라고 했지?"))  # 이전 대화 기억 확인
    
    bot.show_history()