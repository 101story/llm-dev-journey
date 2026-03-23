# 🤖 LLM Developer Journey

> LangChain · RAG · LangGraph · Qdrant를 활용한 AI 에이전트 개발 실습 기록

---

## 📌 목적

**실제 AI 에이전트를 설계하고 구축하는 전 과정**을 직접 경험하는 것을 목표로 합니다.

- OpenAI / Gemma 모델 API 호출 및 파라미터 이해
- LangChain으로 LLM 파이프라인 구성
- 벡터 데이터베이스(Qdrant)로 의미 기반 검색 구현
- RAG 시스템으로 LLM에 외부 지식 주입
- LangGraph로 복잡한 AI 에이전트 워크플로우 설계

---

## 🗂️ 전체 커리큘럼

| Phase | 주제 | 핵심 기술 | 상태 |
|-------|------|----------|------|
| Phase 0 | 환경 세팅 | Python, venv, .env | ✅ 완료 |
| Phase 1 | LLM 기초 + API 호출 | OpenAI API, 스트리밍, 파라미터 | ✅ 완료 |
| Phase 2 | LangChain 기초 | PromptTemplate, Memory, Tool, LCEL | ✅ 완료 |
| Phase 3 | 벡터 데이터베이스 | Embedding, Qdrant, 유사도 검색 | ✅ 완료 |
| Phase 4 | RAG 시스템 구축 | 문서 로딩, 청킹, RetrievalQA | ✅ 완료 |
| Phase 5 | LangGraph 에이전트 | 그래프 워크플로우, 멀티 에이전트 | ✅ 완료 |
| Phase 6 | 실전 통합 프로젝트 | RAG + LangGraph + Qdrant 통합 | ⏳ 예정 |

---

## 📁 프로젝트 구조

```
llm-dev-journey/
│
├── .env                          # API 키 (git 제외)
├── .env.example                  # 환경변수 샘플
├── .gitignore
├── requirements.txt
├── sample_doc.txt                # RAG 실습용 샘플 문서
│
├── phase1_openai_basic.py        # OpenAI API 기본 호출
├── phase1_conversation.py        # 대화 히스토리 + 스트리밍
├── phase1_params.py              # temperature, system prompt 실험
│
├── phase2_prompt_template.py     # PromptTemplate + LCEL Chain
├── phase2_memory.py              # 대화 메모리 관리
├── phase2_output_parser.py       # 응답 구조화 파싱
├── phase2_tools.py               # Tool 바인딩 + 에이전트 흐름
│
├── phase3_embedding.py           # 텍스트 임베딩 + 유사도 계산
├── phase3_qdrant.py              # Qdrant CRUD + 유사도 검색
├── phase3_qdrant_langchain.py    # LangChain Retriever 연동
│
├── phase4_load_and_chunk.py      # 문서 로딩 + 청킹
├── phase4_rag_pipeline.py        # 전체 RAG 파이프라인
├── phase4_rag_with_source.py     # RAG + 출처 반환
│
├── phase5_basic_graph.py         # State, Node, Edge 기본 구조
├── phase5_conditional.py         # 조건 분기 — 질문 유형별 라우팅
└── phase5_loop.py                # 루프 — 품질 기준 미달 시 재시도
```

---

## 🔍 Phase별 상세 설명

### Phase 0 — 환경 세팅
Python 가상환경 구성, API 키 관리, 필수 패키지 설치 등 개발 환경을 세팅합니다.

### Phase 1 — LLM 기초 + API 호출
OpenAI GPT 모델을 직접 호출하며 LLM의 기본 동작을 이해합니다.

- `phase1_openai_basic.py` — API 호출 구조, 토큰 사용량 확인
- `phase1_conversation.py` — 대화 히스토리 유지, 스트리밍 출력
- `phase1_params.py` — temperature·system prompt가 응답에 미치는 영향 실험

**핵심 학습**: LLM은 stateless하기 때문에 대화 히스토리를 직접 messages 배열로 관리해야 한다는 것을 이해합니다.

### Phase 2 — LangChain 기초
LangChain의 핵심 개념인 LCEL(LangChain Expression Language)을 익히고, 프롬프트·메모리·툴을 조합하는 방법을 배웁니다.

- `phase2_prompt_template.py` — 재사용 가능한 프롬프트 설계, `prompt | llm` 체인 구성
- `phase2_memory.py` — 대화 히스토리를 messages 리스트로 직접 관리
- `phase2_output_parser.py` — LLM 응답을 리스트·JSON 구조체로 파싱
- `phase2_tools.py` — `@tool` 데코레이터, `bind_tools`, Tool 실행 3단계 흐름

**핵심 학습**: LLM이 Tool을 호출할 때 content가 비어있고 tool_calls만 반환되며, Tool 결과를 ToolMessage로 다시 전달해야 최종 답변이 생성됩니다.

**트러블슈팅**: LangChain v0.3부터 import 경로가 대거 변경됨.
- `langchain.prompts` → `langchain_core.prompts`
- `langchain.output_parsers` → `langchain_core.output_parsers`
- `langchain_core.pydantic_v1` → `pydantic`
- `langchain.tools` → `langchain_core.tools`
- `ConversationChain`, `LLMChain` 등 구버전 클래스 deprecated → LCEL(`|`) 방식이 표준

### Phase 3 — 벡터 데이터베이스
텍스트를 벡터로 변환하고 의미 기반 유사도 검색을 구현합니다.

- `phase3_embedding.py` — 임베딩 개념, 코사인 유사도 직접 계산
- `phase3_qdrant.py` — Qdrant 메모리 모드, 컬렉션 생성·문서 저장·검색
- `phase3_qdrant_langchain.py` — LangChain VectorStore·Retriever 연동

**핵심 학습**: 벡터 공간에서 의미가 비슷한 텍스트는 가까이 위치하며, 이를 통해 키워드 없이도 관련 문서를 찾을 수 있습니다. `as_retriever(search_kwargs={"k": N})`의 k값은 반환할 문서 수로, 높을수록 더 많은 문맥을 가져오지만 토큰 비용이 증가합니다.

### Phase 4 — RAG 시스템 구축
실제 문서를 로딩해 벡터DB에 저장하고, 사용자 질문에 관련 문서를 검색해 LLM 답변에 주입하는 전체 파이프라인을 구축합니다.

- `phase4_load_and_chunk.py` — 문서 로딩, RecursiveCharacterTextSplitter로 청킹, chunk_overlap 효과 확인
- `phase4_rag_pipeline.py` — 문서 로딩 → 청킹 → 임베딩 → 저장 → 검색 → 답변 전체 LCEL 파이프라인
- `phase4_rag_with_source.py` — RunnableParallel로 답변과 참고 문서 출처를 동시에 반환

**핵심 학습**: RAG는 LLM의 환각(hallucination) 문제를 줄이는 핵심 기법입니다. 프롬프트에 "컨텍스트에 없는 내용은 답하지 말라"고 명시하면 문서 기반으로만 답변하게 할 수 있습니다. `chunk_overlap`은 청크 경계에서 문맥이 잘리는 것을 방지합니다.

### Phase 5 — LangGraph 에이전트
그래프 기반 워크플로우로 조건 분기·루프·멀티 에이전트 협력 시스템을 설계합니다.

- `phase5_basic_graph.py` — StateGraph 기본 구조, Node·Edge·START·END 연결
- `phase5_conditional.py` — `add_conditional_edges`로 질문 유형별 노드 분기
- `phase5_loop.py` — 품질 평가 후 기준 미달 시 재시도하는 루프 구조

**핵심 학습**: LangChain 체인(`A | B | C`)은 항상 직선으로만 흐르지만,
LangGraph는 조건 분기·루프·병렬 실행이 가능해 실제 업무처럼 "상황에 따라
다르게 행동"하는 에이전트를 만들 수 있습니다.
State는 그래프 전체가 공유하는 데이터 구조이며, 각 Node는 State를 받아
처리 후 업데이트된 State를 반환합니다.

### Phase 6 — 실전 통합 프로젝트 _(예정)_
RAG + LangGraph + Qdrant를 통합한 완전한 AI 에이전트를 구축합니다.

---

## ⚙️ 설치 방법

```bash
# 저장소 클론
git clone https://github.com/{username}/llm-dev-journey.git
cd llm-dev-journey

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

## 🔑 환경변수

`.env.example` 파일을 복사해 `.env`로 만들고 키를 입력하세요.

```
OPENAI_API_KEY=sk-...
```

> `.env` 파일은 절대 git에 올리지 마세요.

---

## 📝 실습 중 배운 것들

- LangChain은 버전업이 매우 빠르기 때문에 import 경로가 자주 바뀜 (`langchain` → `langchain_core`)
- `ConversationChain`, `LLMChain` 등 구버전 클래스는 deprecated — LCEL(`|`) 방식이 표준
- Tool 호출 시 LLM의 `content`는 비어있고 `tool_calls`만 반환됨 — Tool 결과를 `ToolMessage`로 다시 전달해야 최종 답변 생성
- Qdrant는 도커 없이 메모리 모드(`:memory:`)로 바로 실습 가능
- RAG에서 `chunk_overlap`은 청크 경계의 문맥 손실을 방지하는 중요한 파라미터
- Retriever의 `k`값은 검색 품질과 토큰 비용의 트레이드오프
- 에이전트는 `AgentExecutor` 대신 LangGraph로 구현하는 것이 현재 표준
- LangGraph의 3요소: State(공유 데이터), Node(처리 함수), Edge(연결·분기 조건)
- `add_conditional_edges`의 라우팅 함수 반환값이 곧 다음 노드 이름이 됨
- 루프는 조건 엣지에서 이전 노드로 되돌아가는 엣지를 추가하는 것으로 구현
- LangChain 체인과 달리 LangGraph는 `compile()` 후에야 실행 가능한 객체가 됨

---

## 🛠️ 개발 환경

- Python 3.10+
- Cursor / VS Code + GitHub Copilot
- OpenAI API
