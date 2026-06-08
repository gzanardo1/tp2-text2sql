# -*- coding: utf-8 -*-
"""
Template de prompt e serialização de esquema para a tarefa Text-to-SQL (Spider).

Este módulo define o TEMPLATE FIXO usado de forma idêntica em:
  - Fase 2 (baseline, modelo não treinado),
  - Fase 4 (modelos fine-tuned),
garantindo comparabilidade. O mesmo formato também deve ser usado para construir
os dados de treino (Fase 3) para evitar descasamento treino/avaliação.

Composição do prompt (conforme enunciado 2.1):
  [system]  instrução da tarefa Text-to-SQL
  [few-shot] 3 exemplos (esquema + pergunta + SQL) extraídos do TRAINING split
  [user]    esquema do banco-alvo + pergunta a ser respondida
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, Dict


SYSTEM_PROMPT = (
    "Você é um especialista em SQLite. Dado o esquema de um banco de dados e uma "
    "pergunta em linguagem natural, gere UMA única consulta SQL que a responda. "
    "Responda APENAS com a consulta SQL, sem explicações, sem comentários e sem "
    "blocos de código markdown."
)


def serialize_schema(db_path: str) -> str:
    """Serializa o esquema do banco como as instruções CREATE TABLE (DDL).

    Usa o DDL real do SQLite (inclui colunas, tipos, PKs e FKs), que é a
    representação mais eficaz para 'schema linking'. Conexão somente-leitura.
    """
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    ddl = "\n".join(r[0].strip().rstrip(";") + ";" for r in rows if r[0])
    return ddl


def render_user_turn(schema_ddl: str, question: str) -> str:
    """Conteúdo de um turno de usuário: esquema + pergunta."""
    return (
        f"Esquema do banco de dados:\n{schema_ddl}\n\n"
        f"Pergunta: {question}\n"
        f"SQL:"
    )


def build_messages(target_schema_ddl: str,
                   question: str,
                   few_shot: List[Dict]) -> List[Dict[str, str]]:
    """Monta a lista de mensagens (chat) com few-shot como turnos anteriores.

    `few_shot` é uma lista de dicts com chaves: 'schema_ddl', 'question', 'query'.
    Os exemplos entram como pares (user -> assistant) antes da pergunta-alvo,
    aproveitando o chat template do modelo instruct.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot:
        messages.append({"role": "user",
                         "content": render_user_turn(ex["schema_ddl"], ex["question"])})
        messages.append({"role": "assistant", "content": ex["query"].strip()})
    messages.append({"role": "user",
                     "content": render_user_turn(target_schema_ddl, question)})
    return messages


# ---------------------------------------------------------------------------
# Auto-teste: valida a serialização de esquema e a montagem das mensagens
# sem precisar de transformers/torch/deepeval. Cria um SQLite temporário.
#   python -m scripts.spider_prompt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, json

    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "concert_singer.sqlite")
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE singer (
            singer_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER
        );
        CREATE TABLE concert (
            concert_id INTEGER PRIMARY KEY,
            singer_id INTEGER,
            year INTEGER,
            FOREIGN KEY (singer_id) REFERENCES singer(singer_id)
        );
    """)
    con.commit(); con.close()

    ddl = serialize_schema(db_path)
    print("== serialize_schema ==")
    print(ddl)
    print()

    few_shot = [
        {"schema_ddl": "CREATE TABLE t (id INTEGER, v INTEGER);",
         "question": "Quantas linhas há em t?",
         "query": "SELECT count(*) FROM t"},
    ]
    msgs = build_messages(ddl, "Qual a idade média dos cantores?", few_shot)
    print("== build_messages ==")
    print(json.dumps(msgs, ensure_ascii=False, indent=2))
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user" and "SQL:" in msgs[-1]["content"]
    assert any(m["role"] == "assistant" for m in msgs)
    print("\nOK: estrutura de mensagens válida.")