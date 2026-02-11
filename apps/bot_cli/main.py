from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from ollama import chat
from ollama._types import ResponseError
from pydantic import ValidationError

from domain.schemas.ollama import OllamaResponse
from infra.mcp_client import call_mcp_tool, to_tool_payload
from infra.settings import settings
from persistence.memory_store import ChatMemoryStore

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_sop",
            "description": "Busca SOPs por palavra-chave (título ou conteúdo).",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string", "description": "Termo de busca"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Executa SQL read-only (somente SELECT) no banco local.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT sem ';'"},
                    "params": {"type": "object"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
            },
        },
    },
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Você é um assistente de fábrica com acesso a ferramentas.\n"
        "- Se o usuário pedir para listar/consultar eventos, status, logs, histórico de manutenção, SEMPRE use run_sql.\n"
        "- Se o usuário pedir SOP, procedimento, instrução, checklist, SEMPRE use search_sop.\n"
        "- Nunca diga que não tem acesso ao banco: você TEM acesso via ferramentas.\n"
        "- Tabelas SQL disponíveis: equipment, compressor_events, maintenance_log, alarm_history, sop.\n"
        "- Não existe tabela chamada events. Para eventos, use compressor_events com join em equipment.\n"
        "- Após usar a ferramenta, responda com um resumo objetivo e cite os campos relevantes."
    ),
}

FORCE_TOOL_MESSAGE = {
    "role": "system",
    "content": (
        "Use uma ferramenta agora. "
        "Nao responda que a consulta nao esta disponivel. "
        "Se for SOP/procedimento use search_sop. "
        "Se for evento/log/status/historico use run_sql."
    ),
}


def parse_tool_args(arguments: Any) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            v = json.loads(arguments)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def should_force_tool_retry(user_text: str, assistant_text: str) -> bool:
    user = user_text.lower()
    assistant = assistant_text.lower()

    tool_intent_keywords = (
        "sop",
        "procedimento",
        "checklist",
        "instrucao",
        "instru",
        "listar",
        "consulta",
        "sql",
        "banco",
        "evento",
        "status",
        "log",
        "historico",
        "manutenc",
        "compressor",
    )

    refusal_markers = (
        "nao esta disponivel",
        "não está disponível",
        "nao tenho acesso",
        "não tenho acesso",
        "nao consigo consultar",
        "não consigo consultar",
        "lista de ferramentas",
        "ferramenta",
    )

    has_tool_intent = any(k in user for k in tool_intent_keywords)
    looks_like_refusal = any(m in assistant for m in refusal_markers)
    return has_tool_intent or looks_like_refusal


def is_unhelpful_assistant_text(text: str) -> bool:
    normalized = text.lower().strip()
    if not normalized:
        return True

    refusal_markers = (
        "nao esta disponivel",
        "não está disponível",
        "nao tenho acesso",
        "não tenho acesso",
        "nao consigo consultar",
        "não consigo consultar",
        "consulta esta bloqueada",
        "consulta está bloqueada",
        "query bloqueada",
    )
    return any(marker in normalized for marker in refusal_markers)


def infer_fallback_tool(user_text: str) -> tuple[str, Dict[str, Any]] | None:
    text = user_text.lower()
    tag_match = re.search(r"\b([a-zA-Z]+-\d+)\b", user_text)
    equipment_tag = tag_match.group(1).upper() if tag_match else None

    if any(k in text for k in ("sop", "procedimento", "checklist", "instru")):
        return ("search_sop", {"text": user_text, "top_k": 5})

    if any(
        k in text
        for k in (
            "evento",
            "event",
            "status",
            "log",
            "historico",
            "manutenc",
            "compressor",
            "banco",
            "sql",
            "listar",
        )
    ):
        if equipment_tag:
            query = """
                SELECT
                    ce.id,
                    e.tag,
                    ce.event_ts,
                    ce.event_type,
                    ce.severity,
                    ce.value,
                    ce.unit,
                    ce.description
                FROM compressor_events ce
                JOIN equipment e ON e.id = ce.equipment_id
                WHERE UPPER(e.tag) = :tag
                ORDER BY ce.event_ts DESC
            """
            return (
                "run_sql",
                {
                    "query": query,
                    "params": {"tag": equipment_tag},
                    "limit": 20,
                },
            )

        query = """
            SELECT
                ce.id,
                e.tag,
                ce.event_ts,
                ce.event_type,
                ce.severity,
                ce.value,
                ce.unit,
                ce.description
            FROM compressor_events ce
            JOIN equipment e ON e.id = ce.equipment_id
            ORDER BY ce.event_ts DESC
        """
        return ("run_sql", {"query": query, "limit": 20})

    return None


def is_query_blocked_result(tool_result: Any) -> bool:
    if isinstance(tool_result, dict):
        error = f"{tool_result.get('error', '')} {tool_result.get('text', '')}".lower()
        return "query bloqueada" in error or "bloqueada" in error

    if isinstance(tool_result, list):
        for item in tool_result:
            if isinstance(item, dict):
                error = f"{item.get('error', '')} {item.get('text', '')}".lower()
                if "query bloqueada" in error or "bloqueada" in error:
                    return True
    return False


def is_missing_table_result(tool_result: Any) -> bool:
    if isinstance(tool_result, dict):
        message = f"{tool_result.get('text', '')} {tool_result.get('error', '')}".lower()
        return "no such table" in message

    if isinstance(tool_result, list):
        for item in tool_result:
            if isinstance(item, dict):
                message = f"{item.get('text', '')} {item.get('error', '')}".lower()
                if "no such table" in message:
                    return True
    return False


def to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    as_dict = getattr(value, "__dict__", None)
    return as_dict if isinstance(as_dict, dict) else {}


def ollama_chat(messages: List[Dict[str, Any]]) -> OllamaResponse:
    resp = chat(
        model=settings.OLLAMA_MODEL,
        messages=messages,
        tools=TOOLS,
        stream=False,
    )
    return OllamaResponse.model_validate(to_plain_dict(resp))


async def main() -> None:
    store = ChatMemoryStore(settings.MEMORY_DB)

    conv_id = str(uuid.uuid4())
    store.create_conversation(conv_id, title="Chat inicial")

    messages: List[Dict[str, Any]] = [SYSTEM_MESSAGE]

    print(f"MCP: {settings.MCP_URL}")
    print(f"Ollama model: {settings.OLLAMA_MODEL}")
    print("Comandos: /list | /load <id> | /new | /next | sair")

    async with streamable_http_client(settings.MCP_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            while True:
                user_text = input("\n> ").strip()
                if user_text.lower() in ("sair", "exit", "quit"):
                    break

                # 1) user -> contexto + persistência
                messages.append({"role": "user", "content": user_text})
                store.append_message(conv_id, "user", user_text)

                # 2) 1ª chamada
                try:
                    first = await asyncio.to_thread(ollama_chat, messages)
                except ResponseError as e:
                    print(f"Ollama error: {e}")
                    continue
                except ValidationError as e:
                    print(f"Ollama response parse error: {e}")
                    continue

                assistant_msg = first.message

                if not assistant_msg.tool_calls and should_force_tool_retry(
                    user_text, assistant_msg.content
                ):
                    forced_messages = messages + [FORCE_TOOL_MESSAGE]
                    try:
                        forced = await asyncio.to_thread(ollama_chat, forced_messages)
                    except ResponseError:
                        forced = None
                    except ValidationError:
                        forced = None

                    if forced and forced.message.tool_calls:
                        assistant_msg = forced.message

                messages.append(
                    {"role": assistant_msg.role, "content": assistant_msg.content}
                )
                store.append_message(conv_id, "assistant", assistant_msg.content)

                # 3) se não tem tool_calls, tenta fallback determinístico
                if not assistant_msg.tool_calls:
                    fallback_tool = infer_fallback_tool(user_text)
                    if fallback_tool is not None:
                        tool_name, tool_args = fallback_tool
                        tool_result = await call_mcp_tool(session, tool_name, tool_args)

                        messages.append(to_tool_payload(tool_name, tool_result))
                        store.append_message(
                            conv_id,
                            "tool",
                            f"{tool_name}: {json.dumps(tool_result, ensure_ascii=False)}",
                        )

                        try:
                            final = await asyncio.to_thread(ollama_chat, messages)
                            final_msg = final.message
                            if not is_unhelpful_assistant_text(final_msg.content):
                                messages.append(
                                    {"role": final_msg.role, "content": final_msg.content}
                                )
                                store.append_message(
                                    conv_id, "assistant", final_msg.content
                                )
                                print(final_msg.content)
                                continue
                        except ResponseError:
                            pass
                        except ValidationError:
                            pass

                        print(json.dumps(tool_result, ensure_ascii=False, indent=2))
                        continue

                    print(assistant_msg.content)
                    continue

                # 4) executa tools
                tool_results: List[Dict[str, Any]] = []
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = parse_tool_args(tc.function.arguments)

                    tool_result = await call_mcp_tool(session, tool_name, tool_args)
                    if tool_name == "run_sql":
                        must_retry_sql = is_query_blocked_result(
                            tool_result
                        ) or is_missing_table_result(tool_result)
                        if must_retry_sql:
                            fallback_tool = infer_fallback_tool(user_text)
                            if fallback_tool is not None and fallback_tool[0] == "run_sql":
                                _, fallback_args = fallback_tool
                                tool_result = await call_mcp_tool(
                                    session, "run_sql", fallback_args
                                )
                    tool_results.append({"tool_name": tool_name, "tool_result": tool_result})

                    messages.append(to_tool_payload(tool_name, tool_result))
                    store.append_message(
                        conv_id,
                        "tool",
                        f"{tool_name}: {json.dumps(tool_result, ensure_ascii=False)}",
                    )

                # 5) 2ª chamada final
                try:
                    final = await asyncio.to_thread(ollama_chat, messages)
                except ResponseError as e:
                    print(f"Ollama error: {e}")
                    continue
                except ValidationError as e:
                    print(f"Ollama response parse error: {e}")
                    continue

                final_msg = final.message
                messages.append({"role": final_msg.role, "content": final_msg.content})
                store.append_message(conv_id, "assistant", final_msg.content)

                if not is_unhelpful_assistant_text(final_msg.content):
                    print(final_msg.content)
                elif tool_results:
                    print(
                        json.dumps(
                            tool_results[-1]["tool_result"],
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                else:
                    print(final_msg.content)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEncerrando...")
        sys.exit(0)
