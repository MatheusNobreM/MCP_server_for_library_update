# MCP + LLM Local (Ollama) + Memória Persistente (SQLite)

Projeto demo de backend de IA com arquitetura separada:

- **MCP Server** expõe tools/resources sobre um banco **sintético**
- **Bot CLI** integra LLM local (Ollama) e usa **tool calling**
- **Memória persistente** (SQLite) com `/list` e `/load <id>`

## Setup

```bash
uv add mcp python-dotenv ollama
```

## Criar banco demo (limpo e sintético)

```bash
uv run python scripts/seed_demo_db.py
```

## Rodar MCP Server

```bash
uv run python -m apps.mcp_server.server
```

## Rodar Bot

```bash
uv run python -m apps.bot_cli.bot
```

### Comandos do bot

- `/list` lista conversas
- `/load <id>` carrega conversa
- `/new` cria conversa nova
- `/clear` limpa conversa atual
- `/title` <texto> define título

### Tools disponíveis

- `search_docs(text, top_k)`
- `run_query(query, params, limit)`
- resource: `doc://{id}`
