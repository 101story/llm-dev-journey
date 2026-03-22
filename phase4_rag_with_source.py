# RAG + 출처(source) 반환
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# 세팅 (동일)
loader = TextLoader("sample_doc.txt", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="rag_source",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="rag_source",
    embedding=embeddings
)
vectorstore.add_documents(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", """아래 컨텍스트만 사용해서 질문에 답하세요.
컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답하세요.

컨텍스트:
{context}"""),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 답변과 검색된 문서를 동시에 반환
rag_chain_with_source = RunnableParallel({
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
    "source_docs": retriever,        # 원본 문서도 함께 반환
}) | RunnableParallel({
    "answer": prompt | llm | StrOutputParser(),
    "source_docs": lambda x: x["source_docs"],
})

# 실행
question = "RAG의 장점이 뭐야?"
result = rag_chain_with_source.invoke(question)

print(f"Q: {question}")
print(f"\nA: {result['answer']}")
print(f"\n=== 참고한 문서 ({len(result['source_docs'])}개) ===")
for i, doc in enumerate(result['source_docs']):
    print(f"\n[출처 {i+1}]")
    print(doc.page_content)