import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

@dataclass
class StoredMessage:
    role: str
    content: str
    ts: str

class ChatMemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init(self) -> None:
        con = self._connect()
        try:
            con.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id)")
            con.commit()
        finally:
            con.close()

    def create_conversation(self, conv_id: str, title: Optional[str] = None) -> None:
        con = self._connect()
        try:
            now = _now_iso()
            con.execute(
                """
                INSERT OR IGNORE INTO conversations (id, title, summary, created_at, updated_at)
                VALUES (?, ?, '', ?, ?)
                """,
                (conv_id, title or "", now, now),
            )
            con.execute("UPDATE conversations SET title=COALESCE(?, title), updated_at=? WHERE id=?",
                        (title or "", now, conv_id))
            con.commit()
        finally:
            con.close()

    def set_title(self, conv_id: str, title: str) -> None:
        con = self._connect()
        try:
            con.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, _now_iso(), conv_id))
            con.commit()
        finally:
            con.close()

    def list_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT id, title, substr(summary, 1, 80) as summary_preview, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()

    def append_message(self, conv_id: str, role: str, content: str) -> None:
        con = self._connect()
        try:
            ts = _now_iso()
            con.execute(
                "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (conv_id, role, content, ts),
            )
            con.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conv_id))
            con.commit()
        finally:
            con.close()

    def load_messages(self, conv_id: str, limit: int = 50) -> List[StoredMessage]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT role, content, ts
                FROM messages
                WHERE conversation_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conv_id, limit),
            ).fetchall()

            rows = list(reversed(rows))
            return [StoredMessage(role=r["role"], content=r["content"], ts=r["ts"]) for r in rows]
        finally:
            con.close()

    def clear_conversation(self, conv_id: str) -> None:
        con = self._connect()
        try:
            con.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
            con.execute("UPDATE conversations SET summary='', updated_at=? WHERE id=?", (_now_iso(), conv_id))
            con.commit()
        finally:
            con.close()
