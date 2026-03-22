# 전체 RAG 파이프라인 구축
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# --- 1단계: 문서 로딩 + 청킹 ---
loader = TextLoader("sample_doc.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)
chunks = splitter.split_documents(documents)
print(f"청크 수: {len(chunks)}개")

# --- 2단계: 벡터DB 저장 ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_client = QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="rag_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="rag_docs",
    embedding=embeddings
)
vectorstore.add_documents(chunks)
print(f"벡터DB 저장 완료")

# --- 3단계: Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 4단계: RAG 프롬프트 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """아래 컨텍스트만 사용해서 질문에 답하세요.
컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답하세요.

컨텍스트:
{context}"""),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 검색된 문서들을 하나의 텍스트로 합치는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 5단계: LCEL로 체인 연결 ---
# retriever → format_docs → prompt → llm → parser
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 실행 ---
print("\n=== RAG 질의응답 ===")

questions = [
    "RAG가 뭐야? 어떤 문제를 해결해줘?",
    "LangGraph로 뭘 할 수 있어?",
    "Qdrant의 장점이 뭐야?",
    "오늘 날씨가 어때?",  # 문서에 없는 내용 — 어떻게 답하는지 확인
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")