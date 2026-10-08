import json
import re
from typing import Any

import duckdb
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import backend.config as config


SQL_SYSTEM_PROMPT = """You are a DuckDB expert. Generate exactly one SQL query that answers the user's question.

Rules:
- Return only raw SQL. Do not use Markdown fences, labels, explanations, or reasoning.
- Use only tables and columns present in the supplied schema.
- Always include the relevant name column when one exists (for example, generation name, line name, or bus name).
- Never use SELECT *.
- Quote column names with double quotes.
- For state/county spatial questions, functions such as ST_GeomFromText may be used when appropriate.
- Generate a read-only query (SELECT or WITH ... SELECT). Do not modify the database.
"""

ANSWER_SYSTEM_PROMPT = """Answer the user's question using the supplied DuckDB query result.
Be concise and do not invent information that is not present in the result.
If the result is empty, say that the database did not return a matching result.
"""

ANSWER_ROW_LIMIT = 100


def clean_sql_output(value: str) -> str:
    """Normalize the small amount of formatting models sometimes add."""
    value = value.strip()
    value = re.sub(r"^```(?:sql)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(r"^SQLQuery:\s*", "", value, flags=re.IGNORECASE)
    return value.strip()


def _message_text(message: Any) -> str:
    """Extract text from a LangChain chat-model response."""
    content = message.content
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _build_llm():
    provider = getattr(config, "llm_provider", "anthropic").lower()

    if provider == "anthropic":
        return ChatAnthropic(
            api_key=config.anthropic_key,
            model=getattr(config, "anthropic_model", "claude-sonnet-4-5-20250929"),
            temperature=0,
            max_tokens=2048,
        )

    if provider == "openai":
        return ChatOpenAI(
            api_key=config.openai_key,
            model=getattr(config, "openai_model", "gpt-5.6-luna"),
            reasoning={
                "effort": "low",
                "summary": "concise",
            },
            temperature=0,
        )

    if provider == "ollama":
        return ChatOllama(
            model=getattr(config, "ollama_model", "llama3"),
            base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
            temperature=0,
        )

    raise ValueError(f"Unsupported llm_provider: {provider!r}")


def _generate_sql(llm, question: str, table_info: str) -> str:
    response = llm.invoke(
        [
            ("system", SQL_SYSTEM_PROMPT),
            (
                "human",
                f"Database schema:\n{table_info}\n\nQuestion: {question}",
            ),
        ]
    )
    sql = clean_sql_output(_message_text(response))

    return sql


def _answer_question(llm, question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    visible_rows = rows[:ANSWER_ROW_LIMIT]
    truncated = len(rows) > ANSWER_ROW_LIMIT

    response = llm.invoke(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            (
                "human",
                "Question:\n"
                f"{question}\n\n"
                "SQL:\n"
                f"{sql}\n\n"
                "Result rows:\n"
                f"{json.dumps(visible_rows, default=str)}\n\n"
                f"Result truncated for answering: {truncated}",
            ),
        ]
    )
    return _message_text(response).strip()

def _retrieve_table_schema() -> str:
    return duckdb.sql("SHOW ALL TABLES").fetchall()


def sqlchain(input_text: str) -> dict[str, Any]:
    llm = _build_llm()

    sql_cmd = ""
    try:
        table_info = _retrieve_table_schema()
        sql_cmd = _generate_sql(llm, input_text, table_info)

        result = duckdb.sql(sql_cmd)
        columns = [desc[0] for desc in result.description]
        query_dict = [dict(zip(columns, row)) for row in result.fetchall()]

        text_result = _answer_question(llm, input_text, sql_cmd, query_dict)

    except Exception as error:
        print(f"error: {error}")
        text_result = "Sorry, I can't find the answer to your question"
        query_dict = []

    return {
        "text": text_result,
        "result_list": query_dict,
    }
