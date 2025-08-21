import argparse
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from pathlib import Path

# --- Configurable Constants ---
DEFAULT_MODEL_NAME = "BAAI/bge-m3"

# Base dir = repo root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = BASE_DIR / "src" / "schema" / "schema_chunks.txt"
DEFAULT_VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# --- Utility ---
def read_schema_chunks(schema_path: Path) -> List[str]:
    """Read and split schema chunks from a plain text file."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks

# --- Embedding & Storage ---
def embed_chunks_and_save(chunks: List[str], output_dir: Path, model_name: str) -> None:
    """Convert schema chunks to embeddings and store in FAISS."""
    print(" Initializing BGE-M3 embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print(f" Creating documents from {len(chunks)} chunks...")
    documents = [Document(page_content=chunk) for chunk in chunks]

    print(" Building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))

    print(f"\nVector store saved to: {output_dir}")
    print(f" Use it with:\nfrom langchain.vectorstores import FAISS\nretriever = FAISS.load_local('{output_dir}', embeddings, allow_dangerous_deserialization=True).as_retriever()")

# --- CLI Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="Embed schema chunks using BGE-M3 and save to FAISS vector store.")
    parser.add_argument("--schema_path", default=str(DEFAULT_SCHEMA_PATH), help="Path to schema_chunks.txt")
    parser.add_argument("--output_dir", default=str(DEFAULT_VECTORSTORE_DIR), help="Directory to save FAISS index")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="HuggingFace model name for embeddings")
    
    args = parser.parse_args()
    schema_path = Path(args.schema_path)
    output_dir = Path(args.output_dir)

    try:
        chunks = read_schema_chunks(schema_path)
        print(f" Loaded {len(chunks)} schema chunks.")
        embed_chunks_and_save(chunks, output_dir, args.model_name)
    except Exception as e:
        print(f" ERROR: {e}")

if __name__ == "__main__":
    main()
