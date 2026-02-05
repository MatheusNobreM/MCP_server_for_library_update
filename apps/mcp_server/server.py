from mcp.server.fastmcp import FastMCP
from .config import settings
from .tools.db_tools import run_query_impl, search_docs_impl, get_doc_impl

db_path = settings.resolved_db_path()

mcp = FastMCP(settings.mcp_name, json_response=True)

@mcp.tool()
def run_query(query: str, params: dict | None = None, limit: int = 50):
    """Executa SQL read-only (somente SELECT) no banco."""
    return run_query_impl(db_path, query=query, params=params, limit=limit)

@mcp.tool()
def search_docs(text: str, top_k: int = 5):
    """Busca documentos (KB/Runbooks) por palavra-chave."""
    return search_docs_impl(db_path, text=text, top_k=top_k)

@mcp.resource("doc://{doc_id}")
def get_doc(doc_id: int) -> str:
    """Retorna o documento completo (endereçável por ID)."""
    return get_doc_impl(db_path, doc_id=doc_id)

if __name__ == "__main__":
    mcp.run(transport=settings.transport)
