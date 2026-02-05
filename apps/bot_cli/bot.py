import asyncio
import json
import os
import re
import sys
import uuid
from dotenv import load_dotenv

from ollama import chat, list as list_models
from ollama._types import ResponseError

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from apps.packages.memory.memory_store import ChatMemoryStore
from .prompts import base_system_prompt

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
MEMORY_DB = os.getenv("MEMORY_DB", "memory.db")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Busca documentos (KB/Runbooks) por palavra-chave.",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Executa SQL read-only (somente SELECT) no banco demo.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "object"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
            },
        },
    },
]

def check_model_available(model_name: str) -> tuple[bool, list[str], str]:
    try:
        models_response = list_models()
        available = [m.model for m in models_response.models] if hasattr(models_response, "models") else []

        if model_name in available:
            return True, available, model_name

        base = model_name.split(":")[0]
        preferred = [f"{base}:0.6b", f"{base}:1.7b", f"{base}:3b", f"{base}:latest"]
        for cand in preferred:
            if cand in available:
                return True, available, cand

        matches = [m for m in available if m.split(":")[0] == base]
        if matches:
            return True, available, matches[0]

        return False, available, model_name
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos do Ollama: {e}")
        return False, [], model_name

def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "__dict__", {"value": str(obj)})

EVENT_TRIGGER = re.compile(r"\b(evento|eventos|logs?)\b", re.IGNORECASE)
ASSET_ALIASES = {
    "gateway": "gateway",
    "gw": "gateway",
    "worker": "worker",
    "workers": "worker",
}

def parse_event_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not EVENT_TRIGGER.search(lower):
        return None

    asset = None
    for token, normalized in ASSET_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", lower):
            asset = normalized
            break

    limit = 10
    m = re.search(
        r"\b(\d{1,3})\s*(?:eventos|events|registros|entradas|ultimos|recentes)\b",
        lower,
    )
    if m:
        try:
            limit = max(1, min(int(m.group(1)), 50))
        except ValueError:
            limit = 10

    return {"asset": asset, "limit": limit}

def format_event_response(result, asset: str | None, limit: int) -> str:
    if isinstance(result, dict) and "text" in result:
        return result["text"]

    if not isinstance(result, list):
        return f"Resposta inesperada do tool: {result}"

    if not result:
        return "Nenhum evento encontrado."

    if isinstance(result[0], dict) and "error" in result[0]:
        return f"Erro ao consultar eventos: {result[0]['error']}"

    header = "Eventos"
    if asset:
        header += f" do {asset}"
    header += f" (ultimos {min(len(result), limit)}):"

    lines = []
    for row in result:
        ts = row.get("ts", "")
        row_asset = row.get("asset", "")
        severity = row.get("severity", "")
        message = row.get("message", "")
        lines.append(f"{ts} | {row_asset} | {severity} | {message}")

    return header + "\n" + "\n".join(lines)

def print_conversation_list(store: ChatMemoryStore, limit: int = 20) -> None:
    convs = store.list_conversations(limit=limit)
    if not convs:
        print("Nenhuma conversa encontrada.")
        return

    print("\nConversas salvas (mais recentes primeiro):")
    for c in convs:
        title = (c.get("title") or "").strip() or "(sem título)"
        print(f"- {c['id']} | {title} | updated={c['updated_at']}")

def load_conversation_into_messages(
    store: ChatMemoryStore,
    conv_id: str,
    base_system_message: dict,
    limit: int = 50,
) -> list[dict]:
    store.create_conversation(conv_id)
    stored = store.load_messages(conv_id, limit=limit)

    msgs = [base_system_message]
    for m in stored:
        if m.role == "tool":
            msgs.append({"role": "assistant", "content": f"[tool] {m.content}"})
        elif m.role in ("user", "assistant", "system"):
            msgs.append({"role": m.role, "content": m.content})
        else:
            msgs.append({"role": "assistant", "content": m.content})
    return msgs

async def call_mcp_tool(session: ClientSession, name: str, args: dict):
    result = await session.call_tool(name, arguments=args)

    payload = getattr(result, "structured_content", None)
    if payload is not None:
        return payload

    content = getattr(result, "content", []) or []
    texts = []
    for c in content:
        t = getattr(c, "text", None)
        texts.append(t if t else str(c))
    return {"text": "\n".join(texts)}

async def main():
    store = ChatMemoryStore(MEMORY_DB)
    conv_id = str(uuid.uuid4())
    store.create_conversation(conv_id, title="Demo Chat")

    ok, available, actual_model = check_model_available(MODEL)
    if not ok:
        print(f"Modelo '{MODEL}' não encontrado.")
        if available:
            print("Modelos disponíveis:", ", ".join(sorted(set(available))[:10]))
        sys.exit(1)

    system_message = {"role": "system", "content": base_system_prompt()}
    messages = [system_message]

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print(f"MCP: {MCP_URL}")
            print(f"Ollama model: {actual_model}")
            print("Comandos: /list | /load <id> | /new | /clear | /title <texto> | sair")
            print("Pergunte algo: (ex: 'busque runbook de deploy' ou 'listar eventos do gateway')")

            while True:
                user_text = input("\n> ").strip()

                if user_text.lower() in ("sair", "exit", "quit"):
                    break

                # comandos de controle (não vão para o LLM)
                if user_text.startswith("/list"):
                    print_conversation_list(store, limit=20)
                    continue

                if user_text.startswith("/load"):
                    parts = user_text.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].strip():
                        print("Uso: /load <conversation_id>")
                        continue
                    conv_id = parts[1].strip()
                    messages = load_conversation_into_messages(store, conv_id, system_message, limit=80)
                    print(f"Conversa carregada: {conv_id} (msgs={len(messages)-1})")
                    continue

                if user_text.startswith("/new"):
                    conv_id = str(uuid.uuid4())
                    store.create_conversation(conv_id, title="Nova conversa")
                    messages = [system_message]
                    print(f"Nova conversa: {conv_id}")
                    continue

                if user_text.startswith("/clear"):
                    store.clear_conversation(conv_id)
                    messages = [system_message]
                    print("Conversa limpa.")
                    continue

                if user_text.startswith("/title"):
                    parts = user_text.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Uso: /title <novo título>")
                        continue
                    store.set_title(conv_id, parts[1].strip())
                    print("Título atualizado.")
                    continue

                event_intent = parse_event_intent(user_text)
                if event_intent:
                    messages.append({"role": "user", "content": user_text})
                    store.append_message(conv_id, "user", user_text)

                    asset = event_intent["asset"]
                    limit = event_intent["limit"]

                    if asset:
                        query = (
                            "SELECT id, asset, ts, severity, message "
                            "FROM events "
                            "WHERE asset = :asset "
                            "ORDER BY ts DESC"
                        )
                        params = {"asset": asset}
                    else:
                        query = (
                            "SELECT id, asset, ts, severity, message "
                            "FROM events "
                            "ORDER BY ts DESC"
                        )
                        params = {}

                    tool_result = await call_mcp_tool(
                        session,
                        "run_query",
                        {"query": query, "params": params, "limit": limit},
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": "run_query",
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    store.append_message(
                        conv_id,
                        "tool",
                        f"run_query: {json.dumps(tool_result, ensure_ascii=False)}",
                    )

                    response_text = format_event_response(tool_result, asset, limit)
                    messages.append({"role": "assistant", "content": response_text})
                    store.append_message(conv_id, "assistant", response_text)
                    print(response_text)
                    continue

                # conversa normal (persiste + vai pro LLM)
                messages.append({"role": "user", "content": user_text})
                store.append_message(conv_id, "user", user_text)

                try:
                    resp = await asyncio.to_thread(
                        chat,
                        model=actual_model,
                        messages=messages,
                        tools=TOOLS,
                        stream=False,
                        options={"num_predict": 256},
                    )
                except ResponseError as e:
                    print(f"Erro Ollama: {e}")
                    continue

                resp_dict = _to_dict(resp)
                assistant_msg = resp_dict.get("message") or _to_dict(getattr(resp, "message", {}))
                messages.append(assistant_msg)
                store.append_message(conv_id, "assistant", assistant_msg.get("content", ""))

                tool_calls = assistant_msg.get("tool_calls") or []
                if tool_calls:
                    for call in tool_calls:
                        fn = call.get("function", {})
                        tool_name = fn.get("name")
                        tool_args = fn.get("arguments", {}) or {}
                        if isinstance(tool_args, str):
                            tool_args = json.loads(tool_args)

                        tool_result = await call_mcp_tool(session, tool_name, tool_args)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": tool_name,
                                "content": json.dumps(tool_result, ensure_ascii=False),
                            }
                        )
                        store.append_message(
                            conv_id,
                            "tool",
                            f"{tool_name}: {json.dumps(tool_result, ensure_ascii=False)}",
                        )

                    # resposta final após tools
                    try:
                        final = await asyncio.to_thread(
                            chat,
                            model=actual_model,
                            messages=messages,
                            tools=TOOLS,
                            stream=False,
                            options={"num_predict": 256},
                        )
                    except ResponseError as e:
                        print(f"Erro Ollama: {e}")
                        continue

                    final_dict = _to_dict(final)
                    final_msg = final_dict.get("message") or _to_dict(getattr(final, "message", {}))
                    messages.append(final_msg)
                    store.append_message(conv_id, "assistant", final_msg.get("content", ""))
                    print(final_msg.get("content", ""))
                else:
                    print(assistant_msg.get("content", ""))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEncerrando...")
        sys.exit(0)
