import duckdb
import argparse
from pathlib import Path
import sys

# --- Utility ---
def clean_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def log_info(msg: str):
    print(f"\033[94m[INFO]\033[0m {msg}")

def log_success(msg: str):
    print(f"\033[92m[SUCCESS]\033[0m {msg}")

def log_warning(msg: str):
    print(f"\033[93m[WARNING]\033[0m {msg}")

def log_error(msg: str):
    print(f"\033[91m[ERROR]\033[0m {msg}")

# --- Extract schema for a single table ---
def get_table_schema(conn, table_name: str) -> str:
    try:
        info = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
        if not info:
            log_warning(f"No columns found for table: {table_name}")
            return ""
        columns = [f"{row[1]} ({row[2].upper()})" for row in info]
        return f"Table: {table_name}\nColumns: {', '.join(columns)}"
    except Exception as e:
        log_warning(f"Failed to extract schema from '{table_name}': {e}")
        return ""

# --- Extract all schema chunks from DB ---
def extract_schema_chunks(db_path: Path, output_path: Path) -> None:
    if not db_path.exists():
        log_error(f"DuckDB file not found: {db_path}")
        sys.exit(1)

    try:
        conn = duckdb.connect(str(db_path))
    except Exception as e:
        log_error(f"Failed to connect to DuckDB: {e}")
        sys.exit(1)

    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        log_info(f"Found {len(tables)} tables.")
    except Exception as e:
        log_error(f"Failed to list tables: {e}")
        conn.close()
        sys.exit(1)

    chunks = []
    for table in tables:
        chunk = get_table_schema(conn, table)
        if chunk:
            chunks.append(chunk)
            log_success(f"Extracted: {table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(chunks))
        log_success(f"\nSaved {len(chunks)} schema chunks → {output_path}")
    except Exception as e:
        log_error(f"Failed to save schema chunks: {e}")

    conn.close()

# --- Entry Point ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent   # repo root

    default_db_path = BASE_DIR / "db" / "data.duckdb"
    default_output_path = BASE_DIR / "src" / "schema" / "schema_chunks.txt"

    parser = argparse.ArgumentParser(description="Extract DuckDB schema chunks for RAG.")
    parser.add_argument("--db_path", default=str(default_db_path), help="Path to DuckDB file")
    parser.add_argument("--output_path", default=str(default_output_path), help="Output text file for schema")

    args = parser.parse_args()
    extract_schema_chunks(Path(args.db_path), Path(args.output_path))
