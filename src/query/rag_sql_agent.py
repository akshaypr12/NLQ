import argparse
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document
import torch

# --- Optional DuckDB for SQL execution ---
try:
    import duckdb
except ImportError:
    duckdb = None

# --- Prompt Template ---
SQL_PROMPT_TEMPLATE = """
You are a SQL expert. Based on the schema provided below, write an SQL query to answer the user's question.

Schema:
{schema}

User Question:
{question}

SQL Query:
"""

# --- Model Loader ---
def get_llm(provider="ollama"):
    if provider == "openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(model_name="gpt-4", temperature=0)
    else:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama2:7b-chat", temperature=0)  # smaller Ollama model

# --- Retriever Loader ---
def load_retriever(vectorstore_path: str, model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Loading embeddings on device: {device}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(), embeddings

# --- Schema Retriever ---
def retrieve_schema(query: str, retriever):
    print(f"\n Retrieving schema chunks for: {query}")
    documents = retriever.get_relevant_documents(query)
    if not documents:
        print(" No relevant schema chunks retrieved.")
        return ""
    return "\n\n".join(doc.page_content for doc in documents)

# --- SQL Generator ---
def generate_sql(schema: str, question: str, llm_provider="ollama"):
    prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=SQL_PROMPT_TEMPLATE.strip()
    )
    llm = get_llm(provider=llm_provider)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(schema=schema, question=question)

# --- SQL Executor ---
def optionally_execute_sql(sql: str, db_path: str):
    if not duckdb:
        print("DuckDB not installed. Skipping execution.")
        return
    try:
        conn = duckdb.connect(db_path)
        df = conn.execute(sql).fetchdf()
        print("\n 🧾 SQL Result Preview:")
        print(df.head())
    except Exception as e:
        print(f" Failed to execute SQL: {e}")

# --- Main CLI Runner ---
def main():
    parser = argparse.ArgumentParser(description="RAG-based SQL generator using schema context + LLM.")
    parser.add_argument("--query", required=True, help="User question to convert into SQL")
    parser.add_argument("--vectorstore_path", default="src/vectorstore", help="Path to saved FAISS index")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3", help="HuggingFace embedding model name")
    parser.add_argument("--llm_provider", default="ollama", choices=["openai", "ollama"], help="LLM backend")
    parser.add_argument("--execute_sql", action="store_true", help="Optionally execute SQL on DuckDB")
    parser.add_argument("--duckdb_path", default="D:/NLA/db/data.duckdb", help="Path to DuckDB file")

    args = parser.parse_args()

    # Step 1: Load retriever
    retriever, _ = load_retriever(args.vectorstore_path, args.embedding_model)

    # Step 2: Retrieve schema
    schema = retrieve_schema(args.query, retriever)
    if not schema:
        print(" No schema found. Cannot proceed.")
        return

    # Step 3: Generate SQL
    print("\n Generating SQL...")
    sql_query = generate_sql(schema=schema, question=args.query, llm_provider=args.llm_provider)
    print("\n Generated SQL Query:\n", sql_query)

    # Step 4: (Optional) Execute SQL
    if args.execute_sql:
        optionally_execute_sql(sql_query, args.duckdb_path)

if __name__ == "__main__":
    main()
