from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("DB_PATH", str(Path("data") / "demo.db"))

    # servidor MCP
    mcp_name: str = os.getenv("MCP_NAME", "Ops Knowledge MCP")
    transport: str = os.getenv("MCP_TRANSPORT", "streamable-http")

    def resolved_db_path(self) -> str:
        return str(Path(self.db_path).resolve())

settings = Settings()
