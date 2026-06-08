# -*- coding: utf-8 -*-
"""
Fase 4 — Avaliação Automatizada via pytest.

Lê o CSV de gerações (produzido por baseline_eval.py, com ou sem --adapter)
e roda a métrica ExecutionAccuracyMetric da Fase 1 sobre cada exemplo do dev
split como um caso de teste parametrizado.

  • Cada exemplo do dev vira UM teste pytest.
  • Teste passa  -> SQL gerada é semanticamente equivalente à gold.
  • Teste falha  -> SQL gerada produz resultado diferente / erro de sintaxe.
  • A acurágia agregada é impressa pelo hook pytest_terminal_summary (conftest).

Por que rodar em pytest mesmo já tendo a coluna `score` no CSV?
  -> O enunciado (4.1) exige integração da métrica em pytest. Aqui a métrica é
     RE-EXECUTADA: o CSV traz a geração bruta; a métrica processa cada linha
     do zero, garantindo que o número reportado é fruto da própria métrica
     auditável (e não de um valor pré-computado).

Uso típico (rodar a partir da raiz do repo):

  # Modelo base (Fase 2)
  pytest scripts/test_execution_accuracy.py \\
      --generations=results/baseline.csv --spider_root=spider \\
      -q --tb=no

  # Modelo fine-tuned (Fase 4)
  pytest scripts/test_execution_accuracy.py \\
      --generations=results/qlora_a.csv --spider_root=spider \\
      -q --tb=no
"""

import csv
import os

import pytest

from custom_metrics.execution_accuracy import ExecutionAccuracyMetric
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Carregamento do CSV de gerações (chamado pelo pytest_generate_tests)
# ---------------------------------------------------------------------------
def _load_examples(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV de gerações não encontrado: {csv_path}\n"
            f"Gere com: python -m scripts.baseline_eval [--adapter ...] "
            f"--out {csv_path}"
        )
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pytest_generate_tests(metafunc):
    """Parametriza `example` (e o id do teste) com cada linha do CSV."""
    if "example" in metafunc.fixturenames:
        csv_path = metafunc.config.getoption("--generations")
        examples = _load_examples(csv_path)
        ids = [f"{i:04d}_{ex.get('db_id','?')}" for i, ex in enumerate(examples)]
        metafunc.parametrize("example", examples, ids=ids)


# ---------------------------------------------------------------------------
# Fixture com a métrica (uma instância para toda a sessão)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def metric(spider_root):
    return ExecutionAccuracyMetric(db_root=os.path.join(spider_root, "database"))


# ---------------------------------------------------------------------------
# Teste parametrizado: 1 exemplo do dev = 1 caso
# ---------------------------------------------------------------------------
def test_execution_accuracy(example, metric, spider_root):
    db_id = example["db_id"]
    db_path = os.path.join(spider_root, "database", db_id, f"{db_id}.sqlite")

    tc = LLMTestCase(
        input=example.get("question", ""),
        actual_output=example["generated_raw"],
        expected_output=example["gold_query"],
        metadata={"db_id": db_id, "db_path": db_path},
    )
    metric.measure(tc)
    # Mensagem informativa em caso de falha (SQL incorreta).
    assert metric.is_successful(), (
        f"[{db_id}] {metric.reason}\n"
        f"  pergunta: {example.get('question','')[:120]}\n"
        f"  gold:     {example['gold_query'][:200]}\n"
        f"  gerada:   {example.get('extracted_sql', '')[:200]}"
    )