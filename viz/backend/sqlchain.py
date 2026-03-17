from langchain import SQLDatabase, SQLDatabaseChain
from langchain.llms.base import LLM
from anthropic import Anthropic
from openai import OpenAI
from typing import Optional, List, Any
import config
from langchain.prompts import PromptTemplate
import sqlalchemy as sqldb
from sqlalchemy import text
import requests as http_requests
import json
import re


def clean_sql_output(text_input):
    # Remove ```sql ... ``` or ``` ... ``` blocks
    cleaned = re.sub(r'```(?:sql)?\s*\n?(.*?)\n?\s*```', r'\1', text_input, flags=re.DOTALL)
    return cleaned.strip()


# Custom Anthropic LLM wrapper to avoid version conflicts
class AnthropicLLM(LLM):
    model: str = "claude-sonnet-4-5-20250929"

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'  # Allow extra fields to avoid proxies error

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929", **kwargs):
        # Don't pass kwargs to super to avoid proxies issue
        super().__init__()
        # Initialize client
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are a SQL query generator. Respond with ONLY the raw SQL query — no prefixes like 'SQLQuery:', no explanations, no reasoning, no markdown. Just the bare SELECT/INSERT/UPDATE statement itself.",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.content[0].text.strip()
        # Remove SQLQuery: prefix if Claude includes it, since LangChain adds its own
        if result.upper().startswith("SQLQUERY:"):
            result = result[len("SQLQUERY:"):].strip()
        return clean_sql_output(result)

    @property
    def _llm_type(self) -> str:
        return "anthropic"


# Custom OpenAI LLM wrapper
class OpenAILLM(LLM):
    model: str = "gpt-3.5-turbo"

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a SQL query generator. Respond with ONLY the raw SQL query — no prefixes like 'SQLQuery:', no explanations, no reasoning, no markdown. Just the bare SELECT/INSERT/UPDATE statement itself."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip()
        if result.upper().startswith("SQLQUERY:"):
            result = result[len("SQLQUERY:"):].strip()
        return clean_sql_output(result)

    @property
    def _llm_type(self) -> str:
        return "openai"


# Custom Ollama LLM wrapper using the Ollama REST API
class OllamaLLM(LLM):
    model: str = "llama3"

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__()
        self.model = model
        self.base_url = base_url

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = http_requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": "You are a SQL query generator. Respond with ONLY the raw SQL query — no prefixes like 'SQLQuery:', no explanations, no reasoning, no markdown. Just the bare SELECT/INSERT/UPDATE statement itself.",
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()["response"].strip()
        # Remove SQLQuery: prefix if model includes it, since LangChain adds its own
        if result.upper().startswith("SQLQUERY:"):
            result = result[len("SQLQUERY:"):].strip()
        return clean_sql_output(result)

    @property
    def _llm_type(self) -> str:
        return "ollama"


def sqlchain(input_text):
    provider = getattr(config, "llm_provider", "anthropic")
    if provider == "ollama":
        llm = OllamaLLM(
            model=getattr(config, "ollama_model", "llama3"),
            base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
        )
    elif provider == "openai":
        llm = OpenAILLM(
            api_key=config.openai_key,
            model=getattr(config, "openai_model", "gpt-3.5-turbo"),
        )
    else:
        llm = AnthropicLLM(api_key=config.anthropic_key, model="claude-sonnet-4-5-20250929")

    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://postgres:{config.sql_key}@localhost:5432/{config.database_name}")
    mydb = sqldb.create_engine(
        f"postgresql+psycopg2://postgres:{config.sql_key}@localhost:5432/{config.database_name}")
    myconnection = mydb.connect()

    _CUSTOMIZE__TEMPLATE = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run then look at the results of the query and return the answer to the input question.
    You must always query for the name column (e.g., generation name, line name, bus name). Never query for all columns from a table.  Wrap each column name in double quotes (") to denote them as delimited identifiers.
    When users query about state or county, you can use ST_GeomFromText(wkt) and postgis functions to calculate spatial relationship between geo entities.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    First look the postgre database for an answer, if you can't find related answer from the database, you can use you own knowlwedge to answer the questions.

    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: text answer

    """
    MY_PROMPT_SUFFIX = """Only use the following tables:
    {table_info}

    Question: {input}"""

    MY_POSTGRES_PROMPT = PromptTemplate(
        input_variables=["input", "table_info"],
        template=_CUSTOMIZE__TEMPLATE + MY_PROMPT_SUFFIX,
    )

    text_chain = SQLDatabaseChain.from_llm(
        llm, db, verbose=True, return_intermediate_steps=True, prompt=MY_POSTGRES_PROMPT)

    text_result = ""
    sql_cmd = ""
    query_dict = []
    try:
        output = text_chain(input_text)
        text_result = output['result']
        sql_cmd = output["intermediate_steps"][1]

        tempr = myconnection.execute(text(sql_cmd)).fetchall()

        query_dict = [dict(record._mapping) for record in tempr]
    except Exception as error:
        error_msg = str(error)
        if ("maximum" in error_msg and "length" in error_msg):
            text_result = "Check the visualization for updated results"
            sql_cmd = error.intermediate_steps[1]
            tempr = myconnection.execute(text(sql_cmd)).fetchall()
            query_dict = [dict(record._mapping) for record in tempr]
        elif ("Rate" in error_msg and "limit" in error_msg):
            text_result = "Check the visualization for updated results"
            sql_cmd = error.intermediate_steps[1]
            tempr = myconnection.execute(text(sql_cmd)).fetchall()
            query_dict = [dict(record._mapping) for record in tempr]
        else:
            print("error:")
            print(error)
            text_result = "Sorry, I can't find the answer to your question"
            query_dict = []

    print(text_result)
    print(sql_cmd)
    return {
        "text": text_result,
        "result_list": query_dict
    }
