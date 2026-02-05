import sqlite3
from typing import Any, Dict, List, Optional
from pathlib import Path

BANNED_TOKENS = [
    "pragma", "attach", "detach",
    "insert", "update", "delete",
    "drop", "alter", "create",
]

def connect_ro(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Banco não encontrado em '{db_path}'. "
            "Rode: uv run python scripts/seed_demo_db.py"
        )

    # read-only via URI
    uri = f"file:{p.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con

def is_safe_select(sql: str) -> bool:
    s = (sql or "").strip().lower()

    # bloqueia múltiplos statements
    if ";" in s:
        return False

    # só SELECT
    if not s.startswith("select"):
        return False

    # bloqueios comuns
    return not any(tok in s for tok in BANNED_TOKENS)

def run_query_impl(
    db_path: str,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if limit < 1 or limit > 200:
        limit = 50

    if not is_safe_select(query):
        return [{"error": "Query bloqueada. Permitido apenas SELECT sem ';' e sem comandos perigosos."}]

    con = connect_ro(db_path)
    try:
        cur = con.cursor()
        cur.execute(query, params or {})
        rows = cur.fetchmany(limit)
        return [dict(r) for r in rows]
    finally:
        con.close()

def search_docs_impl(
    db_path: str,
    text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if top_k < 1 or top_k > 20:
        top_k = 5

    con = connect_ro(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, title, category, substr(content, 1, 180) AS snippet
            FROM docs
            WHERE title LIKE :q OR content LIKE :q
            ORDER BY id DESC
            LIMIT :k
            """,
            {"q": f"%{text}%", "k": top_k},
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

def get_doc_impl(db_path: str, doc_id: int) -> str:
    con = connect_ro(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT id, title, category, content FROM docs WHERE id = :id", {"id": doc_id})
        row = cur.fetchone()
        if not row:
            return "Documento não encontrado."
        return (
            f"ID: {row['id']}\n"
            f"Título: {row['title']}\n"
            f"Categoria: {row['category']}\n\n"
            f"{row['content']}"
        )
    finally:
        con.close()
