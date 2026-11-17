import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from ingest import get_provider, get_vector_store

load_dotenv()

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_llm():
    provider = get_provider()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"))

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.1-flash-lite"),
        temperature=0,
    )


def build_context(store, question: str) -> str:
    """Vetoriza a pergunta e concatena os k=10 chunks mais relevantes."""
    results = store.similarity_search_with_score(question, k=TOP_K)
    return "\n\n".join(doc.page_content.strip() for doc, _score in results)


def search_prompt(question=None):
    """Monta a chain de RAG: pergunta -> busca vetorial -> prompt -> LLM -> resposta.

    Se `question` for informada, executa a chain e retorna a resposta (str).
    Caso contrário, retorna a chain para ser reutilizada pelo chat.
    Retorna None se a inicialização falhar.
    """
    missing = [
        name
        for name in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME")
        if not os.getenv(name)
    ]
    if missing:
        print(f"Variáveis de ambiente ausentes: {', '.join(missing)}")
        return None

    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as e:
        print(f"Erro ao inicializar: {e}")
        return None

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "contexto": RunnableLambda(lambda q: build_context(store, q)),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)
    return chain