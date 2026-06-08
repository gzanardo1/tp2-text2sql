# -*- coding: utf-8 -*-
"""
Fase 2 — Estabelecimento do baseline.

Submete o MODELO BASE (não treinado) ao Spider dev split usando o template fixo
de few-shot, com decodificação GULOSA (determinística), e mede o desempenho com a
métrica de Execution Accuracy da Fase 1. Registra, por entrada, a consulta gerada
e o resultado (sucesso/falha).

O MESMO script/procedimento é reutilizado na Fase 4 para os modelos fine-tuned
(basta apontar --model para o adaptador/modelo treinado).

Uso (Colab T4):
  python -m scripts.baseline_eval \
      --model Qwen/Qwen2.5-3B-Instruct \
      --spider_root spider \
      --few_shot data/few_shot.json \
      --out results/baseline.csv \
      --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Desliga telemetria do DeepEval antes de qualquer import dele.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import set_seed  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402
from custom_metrics.execution_accuracy import ExecutionAccuracyMetric, extract_sql  # noqa: E402
from scripts.spider_prompt import build_messages, serialize_schema  # noqa: E402
from scripts.model_loader import load_model, load_tokenizer  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # fallback simples
    def tqdm(x, **k):
        return x


def db_path_for(spider_root: str, db_id: str) -> str:
    return os.path.join(spider_root, "database", db_id, f"{db_id}.sqlite")


@torch.no_grad()
def generate_sql(tok, model, messages, max_new_tokens: int) -> str:
    """Geração GULOSA (do_sample=False) — determinística."""
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="diretório do adaptador LoRA (Fase 4); omita para baseline")
    ap.add_argument("--spider_root", required=True)
    ap.add_argument("--few_shot", default="data/few_shot.json")
    ap.add_argument("--split", default="dev.json", help="arquivo do split (dev.json)")
    ap.add_argument("--limit", type=int, default=0, help=">0 limita nº de exemplos")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default="results/baseline.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)  # reprodutibilidade

    with open(args.few_shot, encoding="utf-8") as f:
        few_shot = json.load(f)
    with open(os.path.join(args.spider_root, args.split), encoding="utf-8") as f:
        dev = json.load(f)
    if args.limit > 0:
        dev = dev[:args.limit]

    print(f"Modelo: {args.model}"
          f"{f' + adapter: {args.adapter}' if args.adapter else ''}"
          f" | dev: {len(dev)} exemplos | 4bit={args.load_in_4bit}")
    tok = load_tokenizer(args.model)
    model = load_model(args.model, adapter=args.adapter,
                       load_in_4bit=args.load_in_4bit)
    metric = ExecutionAccuracyMetric(db_root=os.path.join(args.spider_root, "database"))

    rows, total = [], 0.0
    schema_cache = {}
    for i, ex in enumerate(tqdm(dev, desc="baseline")):
        db_id = ex["db_id"]
        gold = ex["query"]
        if db_id not in schema_cache:
            schema_cache[db_id] = serialize_schema(db_path_for(args.spider_root, db_id))
        messages = build_messages(schema_cache[db_id], ex["question"], few_shot)

        raw = generate_sql(tok, model, messages, args.max_new_tokens)
        tc = LLMTestCase(
            input=ex["question"],
            actual_output=raw,
            expected_output=gold,
            metadata={"db_id": db_id,
                      "db_path": db_path_for(args.spider_root, db_id)},
        )
        score = metric.measure(tc)
        total += score
        rows.append({
            "idx": i, "db_id": db_id, "question": ex["question"],
            "gold_query": gold, "generated_raw": raw,
            "extracted_sql": extract_sql(raw),
            "score": score, "reason": metric.reason,
        })

    acc = total / max(len(dev), 1)

    # Salva resultados detalhados + resumo
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    per_db = df.groupby("db_id")["score"].mean().sort_values()
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "n": len(dev),
        "execution_accuracy": acc,
        "seed": args.seed,
        "load_in_4bit": args.load_in_4bit,
        "worst_dbs": per_db.head(10).round(3).to_dict(),
    }
    summ_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n== Execution Accuracy (baseline): {acc:.4f}  ({int(total)}/{len(dev)}) ==")
    print(f"Detalhes -> {args.out}\nResumo  -> {summ_path}")


if __name__ == "__main__":
    main()