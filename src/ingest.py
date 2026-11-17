import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 100


def get_provider() -> str:
    """Retorna o provedor configurado (openai | gemini).

    Se LLM_PROVIDER não for informado, usa o provedor cuja API key estiver preenchida.
    """
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider:
        if provider not in ("openai", "gemini"):
            raise ValueError("LLM_PROVIDER deve ser 'openai' ou 'gemini'.")
        return provider
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    raise ValueError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


def get_embeddings():
    provider = get_provider()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )


def get_vector_store(pre_delete_collection: bool = False) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def validate_env() -> None:
    missing = [
        name
        for name, value in (
            ("DATABASE_URL", DATABASE_URL),
            ("PG_VECTOR_COLLECTION_NAME", COLLECTION_NAME),
            ("PDF_PATH", PDF_PATH),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Variáveis de ambiente ausentes: {', '.join(missing)}")
    if not Path(PDF_PATH).is_file():
        raise FileNotFoundError(f"PDF não encontrado em: {PDF_PATH}")
    get_provider()


def load_and_split() -> list[Document]:
    docs = PyPDFLoader(PDF_PATH).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    ).split_documents(docs)

    # Mantém apenas metadados úteis e não vazios
    return [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
        if d.page_content.strip()
    ]


def ingest_pdf():
    try:
        validate_env()
    except (ValueError, FileNotFoundError) as e:
        print(f"Erro de configuração: {e}")
        sys.exit(1)

    chunks = load_and_split()
    print(f"PDF carregado: {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # Recria a collection para que a ingestão seja idempotente (sem duplicar chunks)
    store = get_vector_store(pre_delete_collection=True)

    ids = [f"doc-{i}" for i in range(len(chunks))]
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        store.add_documents(documents=chunks[start:end], ids=ids[start:end])
        print(f"  {min(end, len(chunks))}/{len(chunks)} chunks armazenados")

    print(f"Ingestão concluída na collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()
