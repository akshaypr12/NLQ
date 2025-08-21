import argparse
from pathlib import Path
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Optional DuckDB for SQL execution ---
try:
    import duckdb
except ImportError:
    duckdb = None

# --- Prompt Template ---
SQL_PROMPT_TEMPLATE = """
You are a SQL expert. Based on the schema provided below, write a correct SQL query
to answer the user's question. Return ONLY the SQL query.

Schema:
{schema}

User Question:
{question}

SQL Query:
"""

# --- Load SQLCoder Model ---
def load_sqlcoder(model_id="defog/sqlcoder-7b"):
    """Load SQLCoder from HuggingFace."""
    print(f"[INFO] Loading SQLCoder model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",   # GPU if available
        trust_remote_code=True,
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- Retriever Loader ---
def load_retriever(vectorstore_path: str, model_name: str):
    """Load FAISS retriever for schema search."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading embeddings on device: {device}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3}), embeddings

# --- Schema Retriever ---
def retrieve_schema(query: str, retriever, top_k: int = 3):
    """Retrieve top-k schema chunks relevant to the query."""
    print(f"[INFO] Retrieving schema for query: '{query}'")
    documents = retriever.get_relevant_documents(query)
    if not documents:
        print("[WARN] No relevant schema chunks retrieved.")
        return ""
    return "\n\n".join(doc.page_content for doc in documents[:top_k])

# --- SQL Generator ---
def generate_sql(schema: str, question: str, sqlcoder_pipe):
    """Generate SQL query from schema + question."""
    prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, question=question)
    output = sqlcoder_pipe(prompt, max_new_tokens=256, do_sample=False)

    generated = output[0]["generated_text"]

    # Keep only SQL after "SQL Query:"
    if "SQL Query:" in generated:
        generated = generated.split("SQL Query:")[-1]

    # Remove explanations if LLM added them
    generated = generated.strip().split("\n\n")[0]
    return generated.strip("; ") + ";"

# --- SQL Executor ---
def optionally_execute_sql(sql: str, db_path: str):
    """Run generated SQL against DuckDB if available."""
    if not duckdb:
        print("[WARN] DuckDB not installed. Skipping execution.")
        return
    try:
        conn = duckdb.connect(db_path)
        df = conn.execute(sql).fetchdf()
        print("\n[RESULT] SQL Execution Preview:")
        print(df.head())
    except Exception as e:
        print(f"[ERROR] Failed to execute SQL: {e}")

# --- Main CLI Runner ---
def main():
    parser = argparse.ArgumentParser(description="RAG-based SQL generator using SQLCoder + FAISS schema retriever.")
    parser.add_argument("--query", required=False, help="User question to convert into SQL")
    parser.add_argument("--vectorstore_path", default="src/vectorstore", help="Path to saved FAISS index")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3", help="HuggingFace embedding model name")
    parser.add_argument("--execute_sql", action="store_true", help="Execute SQL on DuckDB if available")
    parser.add_argument("--duckdb_path", default="db/data.duckdb", help="Path to DuckDB file")
    parser.add_argument("--test", action="store_true", help="Run with a sample test query")

    args = parser.parse_args()

    # --- Step 0: Pick query ---
    user_query = args.query or "average of iscommited in sap_oitm_sheet1 table?"
    if args.test:
        print("[TEST] Running with sample query:", user_query)

    # --- Step 1: Load retriever ---
    retriever, _ = load_retriever(args.vectorstore_path, args.embedding_model)

    # --- Step 2: Retrieve schema ---
    schema = retrieve_schema(user_query, retriever)
    if not schema:
        print("[ERROR] No schema found. Cannot proceed.")
        return

    # --- Step 3: Load SQLCoder & Generate SQL ---
    sqlcoder_pipe = load_sqlcoder()
    sql_query = generate_sql(schema=schema, question=user_query, sqlcoder_pipe=sqlcoder_pipe)
    print("\n[SQL] Generated Query:\n", sql_query)

    # --- Step 4: (Optional) Execute SQL ---
    if args.execute_sql:
        optionally_execute_sql(sql_query, args.duckdb_path)

if __name__ == "__main__":
    main()
