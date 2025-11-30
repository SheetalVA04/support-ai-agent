import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_config() -> tuple[Path, Path]:
    load_dotenv()
    data_path = Path(os.getenv("SUPPORT_DATA_PATH", "data/support_kb.csv"))
    persist_dir = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {data_path}. "
            "Create the file or update SUPPORT_DATA_PATH."
        )
    persist_dir.mkdir(parents=True, exist_ok=True)
    return data_path, persist_dir


def read_kb(data_path: Path) -> list[Document]:
    df = pd.read_csv(data_path)
    docs: list[Document] = []
    for idx, row in df.iterrows():
        metadata = {
            "question": row.get("question", ""),
            "tags": row.get("tags", ""),
            "priority": row.get("priority", "medium"),
            "doc_id": f"kb-{idx}",
        }
        content = f"Q: {row.get('question')}\nA: {row.get('answer')}"
        docs.append(Document(page_content=content, metadata=metadata))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_vector_store(docs: list[Document], persist_dir: Path) -> None:
    # Use HuggingFace embeddings (local, no API needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"Vector store updated at {persist_dir.resolve()}")


def main() -> None:
    data_path, persist_dir = load_config()
    docs = read_kb(data_path)
    print(f"Ingesting {len(docs)} text chunks from {data_path}")
    build_vector_store(docs, persist_dir)


if __name__ == "__main__":
    main()

