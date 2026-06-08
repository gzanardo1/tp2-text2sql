# -*- coding: utf-8 -*-
"""
Fase 5 — Análise Quantitativa de Regressão de Capacidade (MMLU).

Avalia um modelo (base OU fine-tuned via --adapter) em uma suíte fixa de
150 questões do MMLU, dividida em 3 categorias × 50 questões:

  STEM            -> college_computer_science
  Humanidades     -> philosophy
  Ciências Sociais-> high_school_macroeconomics

Configurações fiéis ao item 5.1 do enunciado:
  • Modo 5-shot, usando como exemplares as 5 questões do `dev` split de cada
    subcategoria (padrão MMLU). Esses 5 exemplares são IDÊNTICOS para modelo
    base e fine-tuned -> única diferença é o modelo.
  • Decodificação gulosa (do_sample=False), determinística.
  • Seed fixa em toda amostragem.

Para a análise de regressão (item 5.3), rode este script para cada modelo a
ser comparado e calcule deltas a partir dos JSONs salvos.

Uso:
  # Modelo base
  python -m scripts.eval_mmlu --model Qwen/Qwen2.5-3B-Instruct \\
      --out results/mmlu_base.json --load_in_4bit

  # Fine-tuned (config a / b)
  python -m scripts.eval_mmlu --model Qwen/Qwen2.5-3B-Instruct \\
      --adapter runs/qlora_a --out results/mmlu_qlora_a.json --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import set_seed  # noqa: E402

from scripts.model_loader import load_model, load_tokenizer  # noqa: E402

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x


# ---------------------------------------------------------------------------
# Suíte fixa: 3 categorias x 1 subcategoria do MMLU x 50 questões
# (a subcategoria é o "subject" config dentro de cais/mmlu)
# ---------------------------------------------------------------------------
SUBJECTS = {
    "STEM":             ("college_computer_science", "computer science"),
    "Humanidades":      ("philosophy",                "philosophy"),
    "Ciências Sociais": ("high_school_macroeconomics", "macroeconomics"),
}
LETTERS = "ABCD"
N_FEW_SHOT = 5


# ---------------------------------------------------------------------------
# Construção do prompt no formato canônico do MMLU (raw text completion)
# Não usamos chat template aqui: o protocolo padrão do MMLU é completion,
# e queremos comparar base e fine-tuned no mesmo formato.
# ---------------------------------------------------------------------------
def _format_one(question: str, choices, answer_idx: int = None) -> str:
    out = question + "\n"
    for letter, choice in zip(LETTERS, choices):
        out += f"{letter}. {choice}\n"
    out += "Answer:"
    if answer_idx is not None:
        out += f" {LETTERS[answer_idx]}\n\n"
    return out


def build_mmlu_prompt(few_shot, question, choices, subject_desc):
    header = (f"The following are multiple choice questions (with answers) "
              f"about {subject_desc}.\n\n")
    body = header
    for ex in few_shot:
        body += _format_one(ex["question"], ex["choices"], ex["answer"])
    body += _format_one(question, choices, answer_idx=None)
    return body


# ---------------------------------------------------------------------------
# Predição: gera poucos tokens e extrai a primeira letra ABCD encontrada.
# Robusto a tokenizadores e a saídas com prefixo de espaço/quebra de linha.
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_letter(model, tokenizer, prompt: str, max_new_tokens: int = 5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                           skip_special_tokens=True)
    m = re.search(r"[ABCD]", gen.upper())
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Amostragem determinística do subject
# ---------------------------------------------------------------------------
def sample_subject(subject: str, n: int, seed: int):
    """Retorna (few_shot[:5], test_questions[:n]) determinísticos."""
    dev = load_dataset("cais/mmlu", subject, split="dev")
    test = load_dataset("cais/mmlu", subject, split="test")

    few_shot = [dev[i] for i in range(min(N_FEW_SHOT, len(dev)))]
    test = test.shuffle(seed=seed).select(range(min(n, len(test))))
    return few_shot, list(test)


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="diretório do adaptador LoRA (omita para o modelo base)")
    ap.add_argument("--out", required=True,
                    help="arquivo JSON de saída com resumo + detalhes")
    ap.add_argument("--n_per_category", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="(recomendado em T4 e para casar com o setup do train)")
    args = ap.parse_args()

    set_seed(args.seed)

    label = args.adapter or args.model
    print(f"Modelo: {args.model}" +
          (f"  +  adapter: {args.adapter}" if args.adapter else "") +
          f"  |  4bit={args.load_in_4bit}")
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, adapter=args.adapter,
                       load_in_4bit=args.load_in_4bit)

    details = []
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})

    for category, (subject, desc) in SUBJECTS.items():
        print(f"\n=== {category}  ({subject})  -> {args.n_per_category} questões ===")
        few_shot, questions = sample_subject(subject, args.n_per_category, args.seed)
        for q in tqdm(questions, desc=category):
            prompt = build_mmlu_prompt(few_shot, q["question"], q["choices"], desc)
            pred = predict_letter(model, tokenizer, prompt)
            gold = LETTERS[q["answer"]]
            ok = (pred == gold)
            by_cat[category]["correct"] += int(ok)
            by_cat[category]["total"] += 1
            details.append({
                "category": category,
                "subject": subject,
                "question": q["question"][:120],
                "predicted": pred,
                "correct": gold,
                "ok": ok,
            })

    overall_c = sum(c["correct"] for c in by_cat.values())
    overall_t = sum(c["total"] for c in by_cat.values())
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "label": label,
        "overall_accuracy": overall_c / overall_t,
        "by_category": {
            cat: {
                "subject": SUBJECTS[cat][0],
                "accuracy": c["correct"] / c["total"],
                "correct": c["correct"],
                "total": c["total"],
            }
            for cat, c in by_cat.items()
        },
        "seed": args.seed,
        "n_per_category": args.n_per_category,
        "n_few_shot": N_FEW_SHOT,
        "load_in_4bit": args.load_in_4bit,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details},
                  f, ensure_ascii=False, indent=2)

    print("\n== MMLU Accuracy ==")
    for cat, c in summary["by_category"].items():
        print(f"  {cat:18s} {c['accuracy']:.4f}  ({c['correct']}/{c['total']})  "
              f"[{c['subject']}]")
    print(f"  {'AGREGADO':18s} {summary['overall_accuracy']:.4f}  "
          f"({overall_c}/{overall_t})")
    print(f"\nResultados -> {args.out}")


if __name__ == "__main__":
    main()