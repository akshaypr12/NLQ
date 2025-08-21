import os
import duckdb
import logging
import argparse
import time
import pandas as pd
from pathlib import Path

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Utility function ---
def clean_name(name: str) -> str:
    """Standardize names to lowercase + underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

# --- Load Excel into DuckDB ---
def load_excel_to_duckdb(excel_dir: str, db_path: str) -> None:
    start_time = time.time()

    excel_dir = Path(excel_dir)
    db_path = Path(db_path)

    if not excel_dir.exists():
        logging.error(f"Excel directory not found: {excel_dir}")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    # Try to install/load DuckDB Excel extension
    duckdb_excel_available = True
    try:
        conn.execute("INSTALL 'excel';")
        conn.execute("LOAD 'excel';")
        logging.info("DuckDB Excel extension loaded successfully.")
    except Exception as e:
        duckdb_excel_available = False
        logging.warning(f"DuckDB Excel extension not available, using pandas fallback: {e}")

    # Collect Excel files
    excel_files = list(excel_dir.glob("*.xlsx"))
    logging.info(f"Found {len(excel_files)} Excel files.")

    if not excel_files:
        logging.warning("No Excel files found. Exiting.")
        conn.close()
        return

    for file_path in excel_files:
        file_base = clean_name(file_path.stem)
        sheet_names = []

        # Get sheet names
        if duckdb_excel_available:
            try:
                sheet_names = conn.execute(
                    f"SELECT sheet_name FROM read_excel('{file_path.as_posix().replace("'", "''")}', all_sheets=true);"
                ).fetchall()
                sheet_names = list(set(s[0] for s in sheet_names))
            except Exception as e:
                logging.warning(f"[DuckDB] Could not read sheet names from {file_path.name}: {e}")
                duckdb_excel_available = False  # Switch permanently
                try:
                    sheet_names = pd.ExcelFile(file_path).sheet_names
                except Exception as e:
                    logging.error(f"[pandas] Failed reading sheet names: {e}")
                    continue
        else:
            try:
                sheet_names = pd.ExcelFile(file_path).sheet_names
            except Exception as e:
                logging.error(f"[pandas] Could not read sheet names from {file_path.name}: {e}")
                continue

        logging.info(f"{file_path.name}: {len(sheet_names)} sheets found")

        for sheet in sheet_names:
            table_name = f"{file_base}_{clean_name(sheet)}"
            try:
                if duckdb_excel_available:
                    query = f"""
                    CREATE OR REPLACE TABLE "{table_name}" AS
                    SELECT * FROM read_excel('{file_path.as_posix().replace("'", "''")}', sheet='{sheet}');
                    """
                    conn.execute(query)
                    count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
                    df.dropna(how="all", inplace=True)
                    df.columns = [clean_name(str(col)) for col in df.columns]
                    conn.unregister("temp_df") if "temp_df" in conn.execute("SHOW TABLES").fetchall() else None
                    conn.register("temp_df", df)
                    conn.execute(f"CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM temp_df")
                    count = len(df)

                logging.info(f" Loaded table: {table_name} ({count} rows)")

            except Exception as e:
                logging.warning(f" Failed to load sheet '{sheet}' from '{file_path.name}': {e}")

    # Show all tables
    tables = conn.execute("SHOW TABLES").fetchall()
    logging.info("\nTables created in DB:")
    for t in tables:
        logging.info(f" - {t[0]}")

    conn.close()
    duration = round(time.time() - start_time, 2)
    logging.info(f"\n Done. Time taken: {duration} seconds.")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Excel files into DuckDB.")

    # relative paths (works in GitHub, Codespaces, Linux, Windows)
    default_excel_dir = Path("Data")                
    default_db_path = Path("db/data.duckdb")        

    parser.add_argument("--excel_dir", default=str(default_excel_dir), help="Path to folder with Excel files")
    parser.add_argument("--db_path", default=str(default_db_path), help="Path to output DuckDB file")
    args = parser.parse_args()

    load_excel_to_duckdb(args.excel_dir, args.db_path)

