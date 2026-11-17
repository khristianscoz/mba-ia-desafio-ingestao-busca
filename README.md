# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio do MBA em Engenharia de Software com IA (Full Cycle): ingestão de um PDF em PostgreSQL + pgVector e um chat via CLI que responde **somente** com base no conteúdo do PDF (RAG).

## Como funciona

**Ingestão (`src/ingest.py`)**
1. Carrega o `document.pdf` com `PyPDFLoader`.
2. Divide em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`).
3. Gera os embeddings de cada chunk e grava no Postgres via `PGVector`.
   A collection é recriada a cada execução, então rodar a ingestão de novo não duplica dados.

**Busca (`src/search.py`)**
1. Vetoriza a pergunta.
2. Busca os **10 chunks mais relevantes** com `similarity_search_with_score(query, k=10)`.
3. Monta o prompt do desafio com o contexto concatenado e chama a LLM.
4. Retorna a resposta.

**Chat (`src/chat.py`)**: loop no terminal que usa a chain do `search.py`. Digite `sair` (ou `Ctrl+C`) para encerrar.

## Provedores e modelos

O projeto funciona com **OpenAI** ou **Gemini**, escolhido pela variável `LLM_PROVIDER` (se vazia, usa o provedor cuja API key estiver preenchida). Os modelos são configuráveis no `.env`:

| Provedor | Embeddings (padrão) | LLM (padrão) |
|---|---|---|
| OpenAI | `text-embedding-3-small` | `gpt-5-nano` |
| Gemini | `models/gemini-embedding-001` | `gemini-3.1-flash-lite` |

> **Atenção:** cada modelo de embedding gera vetores com uma dimensão diferente. Se trocar o modelo de embeddings (ou o provedor) depois da primeira ingestão, apague o volume do banco (`docker compose down -v`) e rode a ingestão de novo.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API key da OpenAI **ou** do Google (Gemini)

## Execução

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha a API key do provedor escolhido (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`) e o `LLM_PROVIDER`. Os demais valores já vêm prontos para o `docker-compose.yml`:

```env
DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='document_chunks'
PDF_PATH='document.pdf'
```

### 3. Subir o banco de dados

```bash
docker compose up -d
```

### 4. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
PDF carregado: 67 chunks (size=1000, overlap=150).
  67/67 chunks armazenados
Ingestão concluída na collection 'document_chunks'.
```

### 5. Rodar o chat

```bash
python src/chat.py
```

Exemplo:

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da empresa Alfa Energia S.A.?
RESPOSTA: O faturamento da Alfa Energia S.A. foi de R$ 722.875.391,46.

---

Faça sua pergunta:

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Estrutura

```
├── docker-compose.yml    # Postgres 17 + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF (split + embeddings + PGVector)
│   ├── search.py         # Busca vetorial (k=10) + prompt + LLM
│   ├── chat.py           # CLI para interação com o usuário
├── document.pdf          # PDF para ingestão
└── README.md
```
