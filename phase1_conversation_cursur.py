# 대화 히스토리 유지 + 스트리밍 실습
import os
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    ("system", "You are a helpful assistant."),
    ("user", "What is the capital of France?"),
    ("assistant", "The capital of France is Paris."),
]

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
print()
