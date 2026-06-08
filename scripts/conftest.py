# -*- coding: utf-8 -*-
"""
Configuração do pytest para a Fase 4.

Define duas opções de linha de comando consumidas por test_execution_accuracy.py:
  --generations  CSV produzido por baseline_eval.py (geração do modelo)
  --spider_root  Diretório raiz do Spider (com database/<db_id>/<db_id>.sqlite)

Também adiciona um hook que imprime a Execution Accuracy agregada
(passed / (passed+failed)) ao final da rodada — cada teste que falha é um
exemplo do dev split em que a SQL gerada não bate com a gold.
"""

import os
import sys

import pytest

# Coloca a raiz do repo no sys.path para que os tests importem
# custom_metrics.* e scripts.* sem hack adicional.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--generations", action="store",
        default="results/baseline.csv",
        help="CSV de gerações produzido por baseline_eval.py",
    )
    parser.addoption(
        "--spider_root", action="store",
        default="spider",
        help="Diretório raiz do Spider",
    )


@pytest.fixture(scope="session")
def spider_root(pytestconfig):
    return pytestconfig.getoption("--spider_root")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Imprime Execution Accuracy = passed / (passed + failed)."""
    stats = terminalreporter.stats
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    total = passed + failed
    if total == 0:
        return
    terminalreporter.write_sep("=", "Execution Accuracy")
    terminalreporter.write_line(
        f"  {passed}/{total} = {passed / total:.4f}   "
        f"(passed = SQL correta;  failed = SQL incorreta)"
    )
    gen_path = config.getoption("--generations")
    terminalreporter.write_line(f"  fonte: {gen_path}")