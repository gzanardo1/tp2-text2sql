# -*- coding: utf-8 -*-
"""
Preparação de dados do Spider para o TP2.

Faz duas coisas, ambas determinísticas:
  1) Congela os 3 exemplos FEW-SHOT (do training split) usados no template fixo
     (Fases 2 e 4)  ->  <out>/few_shot.json
  2) Gera os dados de fine-tuning (Fase 3) em chat format, ZERO-SHOT, usando
     EXATAMENTE o mesmo SYSTEM_PROMPT e o mesmo render_user_turn da avaliação
     (alinhamento de formato treino/avaliação)  ->  <out>/train_chat.jsonl

Layout esperado do Spider (download oficial):
  <spider_root>/
    train_spider.json
    train_others.json   (opcional)
    dev.json
    tables.json
    database/<db_id>/<db_id>.sqlite

Uso:
  python -m scripts.prepare_data --spider_root spider --out data --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Permite importar o módulo de prompt rodando a partir da raiz do repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.spider_prompt import SYSTEM_PROMPT, serialize_schema, render_user_turn  # noqa: E402


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_train(spider_root: str) -> List[dict]:
    data = _load_json(os.path.join(spider_root, "train_spider.json"))
    others = os.path.join(spider_root, "train_others.json")
    if os.path.exists(others):
        data = data + _load_json(others)
    return data


def db_path_for(spider_root: str, db_id: str) -> str:
    return os.path.join(spider_root, "database", db_id, f"{db_id}.sqlite")


def tables_count(spider_root: str) -> Dict[str, int]:
    """#tabelas por db_id, a partir de tables.json."""
    tj = _load_json(os.path.join(spider_root, "tables.json"))
    return {t["db_id"]: len(t["table_names_original"]) for t in tj}


def pick_few_shot(train: List[dict], spider_root: str, n: int = 3) -> List[dict]:
    """Seleciona n exemplos few-shot de forma determinística.

    Critério: bancos com MENOS tabelas (esquema curto -> prompt enxuto), de
    db_ids distintos, escolhidos em ordem alfabética estável. Cada exemplo
    carrega o DDL serializado do seu próprio banco.
    """
    ntab = tables_count(spider_root)
    # Indexa o primeiro exemplo de cada db_id presente no train.
    first_by_db: Dict[str, dict] = {}
    for ex in train:
        first_by_db.setdefault(ex["db_id"], ex)

    candidatos = sorted(
        first_by_db.keys(),
        key=lambda db: (ntab.get(db, 999), db),  # menos tabelas, depois alfabético
    )
    few_shot = []
    for db in candidatos:
        if len(few_shot) >= n:
            break
        ex = first_by_db[db]
        few_shot.append({
            "db_id": db,
            "question": ex["question"],
            "query": ex["query"],
            "schema_ddl": serialize_schema(db_path_for(spider_root, db)),
        })
    return few_shot


def to_chat_example(spider_root: str, ex: dict) -> dict:
    """Converte um exemplo do Spider para chat format ZERO-SHOT (treino)."""
    schema_ddl = serialize_schema(db_path_for(spider_root, ex["db_id"]))
    return {
        "db_id": ex["db_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_turn(schema_ddl, ex["question"])},
            {"role": "assistant", "content": ex["query"].strip()},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_root", required=True, help="diretório do Spider")
    ap.add_argument("--out", default="data", help="diretório de saída")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_few_shot", type=int, default=3)
    ap.add_argument("--limit_train", type=int, default=0,
                    help="se >0, limita o nº de exemplos de treino (debug)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train = load_train(args.spider_root)

    # 1) Few-shot congelado
    few_shot = pick_few_shot(train, args.spider_root, n=args.n_few_shot)
    fs_path = os.path.join(args.out, "few_shot.json")
    with open(fs_path, "w", encoding="utf-8") as f:
        json.dump(few_shot, f, ensure_ascii=False, indent=2)
    print(f"[few-shot] {len(few_shot)} exemplos -> {fs_path}")
    for ex in few_shot:
        print(f"   - db={ex['db_id']:22s} q={ex['question'][:50]!r}")

    # 2) Dados de treino (chat format, zero-shot)
    if args.limit_train > 0:
        train = train[:args.limit_train]
    train_path = os.path.join(args.out, "train_chat.jsonl")
    n_ok = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            try:
                row = to_chat_example(args.spider_root, ex)
            except Exception as e:
                print(f"   [skip] {ex.get('db_id')}: {e}")
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[treino] {n_ok} exemplos em chat format -> {train_path}")


if __name__ == "__main__":
    main()