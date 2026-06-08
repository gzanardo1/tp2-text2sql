# -*- coding: utf-8 -*-
"""
Execution Accuracy (Fase 1 do TP2).

Métrica customizada de DeepEval para a tarefa Text-to-SQL no benchmark Spider.
Avaliação por EXECUÇÃO: compara o resultado da consulta gerada pelo modelo com o
resultado da consulta de referência (gold), retornando 1.0 (acerto) ou 0.0 (erro).

Esta é a ÚNICA métrica de avaliação da tarefa Text-to-SQL e deve ser aplicada de
forma idêntica no baseline (Fase 2) e nos modelos fine-tuned (Fase 4).

Estrutura esperada do test case (DeepEval LLMTestCase):
    - input            : prompt enviado ao modelo (esquema + few-shot + pergunta)
    - actual_output    : SAÍDA BRUTA do modelo (pode conter markdown / texto extra)
    - expected_output  : SQL de referência (gold) — executado diretamente
    - additional_metadata : {"db_id": "<nome_do_banco>"}  (e, opcionalmente,
                            {"db_path": "/caminho/para/banco.sqlite"})

Layout de banco assumido (padrão do Spider):
    <db_root>/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Import do DeepEval com fallback.
# Em produção, DeepEval estará instalado e usaremos a BaseMetric real.
# O fallback permite testar a LÓGICA (extração/execução/comparação) em ambientes
# sem DeepEval — útil para CI e para validação rápida.
# ---------------------------------------------------------------------------
try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
    _DEEPEVAL_AVAILABLE = True
except Exception:  # pragma: no cover - apenas fallback de ambiente
    _DEEPEVAL_AVAILABLE = False

    class BaseMetric:  # type: ignore
        """Stub mínimo apenas para permitir import sem DeepEval."""
        pass

    class LLMTestCase:  # type: ignore
        def __init__(self, input=None, actual_output=None,
                     expected_output=None, metadata=None,
                     additional_metadata=None):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            self.metadata = metadata or additional_metadata or {}
            self.additional_metadata = self.metadata  # alias para retrocompat


# ===========================================================================
# Funções de lógica pura (testáveis sem DeepEval)
# ===========================================================================

# Sinaliza onde começa uma consulta SQL "de verdade".
_SQL_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)
# Bloco de código markdown ```sql ... ``` ou ``` ... ```
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Detecção de ORDER BY (na consulta gold decide se a ordem importa).
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
# Tokens que indicam que uma linha faz parte de uma consulta SQL (para limpar prosa).
_SQL_TOKENS = re.compile(
    r"[(),=<>*]|\b(from|where|join|select|group|order|having|on|and|or|limit|"
    r"as|by|count|sum|avg|min|max|distinct|union|intersect|except|inner|left|"
    r"right|outer|not|in|like|between|is|null|asc|desc)\b",
    re.IGNORECASE,
)


def extract_sql(raw_output: str) -> str:
    """Extrai uma consulta SQL a partir da saída bruta do modelo (item 1.1.a).

    Estratégia robusta:
      1. Se houver bloco markdown (```sql ... ```), considera apenas seu conteúdo.
      2. Recorta a partir do primeiro WITH/SELECT (descarta preâmbulo explicativo).
      3. Corta no primeiro ';' (descarta texto após o fim da consulta).
      4. Normaliza espaços em branco.
    """
    if not raw_output:
        return ""

    text = raw_output

    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1)

    start = _SQL_START.search(text)
    if start:
        text = text[start.start():]

    # Mantém apenas a primeira instrução (até o primeiro ';').
    text = text.split(";")[0]

    # Para no primeiro parágrafo em branco (separa SQL de explicações).
    text = re.split(r"\n\s*\n", text)[0]

    # Remove linhas finais que claramente NÃO são SQL (prosa explicativa),
    # preservando linhas com tokens/keywords SQL.
    lines = text.splitlines()
    while len(lines) > 1:
        last = lines[-1].strip()
        if last and not _SQL_TOKENS.search(last):
            lines.pop()
        else:
            break
    text = "\n".join(lines)

    return text.strip()


def has_order_by(sql: str) -> bool:
    """True se a consulta contém ORDER BY (a ordem das linhas passa a importar)."""
    return _ORDER_BY.search(sql or "") is not None


def _resolve_db_path(db_root: str, db_id: str) -> str:
    """Monta o caminho do .sqlite no layout padrão do Spider."""
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


def execute_query(db_path: str, sql: str,
                  timeout: float = 30.0) -> Tuple[Optional[List[tuple]], Optional[str]]:
    """Executa `sql` em conexão SOMENTE-LEITURA (transação segura, item 1.1.b/c/d).

    Retorna (linhas, None) em sucesso ou (None, mensagem_erro) em falha de sintaxe
    ou execução. O modo `?mode=ro` impede qualquer escrita acidental no banco.
    """
    if not sql:
        return None, "consulta vazia"
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    con = None
    try:
        con = sqlite3.connect(uri, uri=True, timeout=timeout)
        # Evita erro de decodificação em bancos com bytes inválidos.
        con.text_factory = lambda b: b.decode("utf-8", "ignore") if isinstance(b, bytes) else b
        cur = con.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:  # erros de sintaxe / execução tratados aqui
        return None, str(e)
    finally:
        if con is not None:
            con.close()


def _normalize_cell(value):
    """Normaliza uma célula para comparação estável (evita ruído de float)."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def _normalize_rows(rows: Sequence[Sequence]) -> List[tuple]:
    return [tuple(_normalize_cell(c) for c in row) for row in rows]


def compare_results(gold_rows: Sequence[Sequence],
                    pred_rows: Sequence[Sequence],
                    ordered: bool) -> bool:
    """Compara conjuntos de resultados (item 1.1.e).

    - ordered=False  -> comparação como MULTICONJUNTO (insensível à ordem das linhas,
                        mas sensível à multiplicidade — respeita duplicatas).
    - ordered=True   -> comparação posição a posição (respeita a ordem do ORDER BY).

    Observação: a comparação é sensível à ORDEM DAS COLUNAS, conforme o enunciado
    (que relaxa apenas a ordem das LINHAS). O avaliador oficial do Spider
    (test-suite evaluation) é mais permissivo quanto à ordem de colunas; isso é
    citado no relatório como referência e limitação conhecida.
    """
    g = _normalize_rows(gold_rows)
    p = _normalize_rows(pred_rows)
    if ordered:
        return g == p
    return Counter(g) == Counter(p)


def execution_match(db_path: str,
                    predicted_sql: str,
                    gold_sql: str,
                    timeout: float = 30.0) -> Tuple[float, str]:
    """Núcleo da métrica: retorna (score, motivo).

    score = 1.0 se os resultados batem; 0.0 caso contrário.
    """
    gold_rows, gold_err = execute_query(db_path, gold_sql, timeout)
    if gold_err is not None:
        # Falha na gold geralmente indica problema de dados/ambiente, não do modelo.
        return 0.0, f"Falha ao executar a consulta GOLD: {gold_err}"

    pred_rows, pred_err = execute_query(db_path, predicted_sql, timeout)
    if pred_err is not None:
        return 0.0, f"Falha ao executar a consulta gerada: {pred_err}"

    ordered = has_order_by(gold_sql)
    ok = compare_results(gold_rows, pred_rows, ordered)
    if ok:
        return 1.0, f"Resultados idênticos (ordered={ordered})."
    return 0.0, (f"Resultados divergentes (ordered={ordered}). "
                 f"gold={len(gold_rows)} linhas, pred={len(pred_rows)} linhas.")


# ===========================================================================
# Métrica DeepEval
# ===========================================================================

class ExecutionAccuracyMetric(BaseMetric):
    """Execution Accuracy para Text-to-SQL (Spider), compatível com DeepEval.

    Uso:
        metric = ExecutionAccuracyMetric(db_root="spider/database")
        tc = LLMTestCase(
            input=prompt,
            actual_output=saida_bruta_do_modelo,
            expected_output=sql_gold,
            additional_metadata={"db_id": "concert_singer"},
        )
        metric.measure(tc)   # -> 1.0 ou 0.0
    """

    def __init__(self, db_root: str, threshold: float = 0.5, timeout: float = 30.0):
        self.db_root = db_root
        self.threshold = threshold
        self.timeout = timeout
        # Atributos esperados pelo ecossistema do DeepEval:
        self.score: float = 0.0
        self.success: bool = False
        self.reason: Optional[str] = None
        self.error: Optional[str] = None
        self.include_reason: bool = True
        self.async_mode: bool = False
        self.strict_mode: bool = False
        self.verbose_mode: bool = False
        self.evaluation_model: Optional[str] = None

    def _db_path_for(self, test_case: "LLMTestCase") -> str:
        # Aceita 'metadata' (DeepEval ≥ 3.x) ou 'additional_metadata' (antigo).
        meta = (getattr(test_case, "metadata", None)
                or getattr(test_case, "additional_metadata", None)
                or {})
        if meta.get("db_path"):
            return meta["db_path"]
        db_id = meta.get("db_id")
        if not db_id:
            raise ValueError(
                "metadata deve conter 'db_id' (ou 'db_path') "
                "para localizar o banco SQLite do Spider."
            )
        return _resolve_db_path(self.db_root, db_id)

    def measure(self, test_case: "LLMTestCase") -> float:
        try:
            db_path = self._db_path_for(test_case)
            predicted_sql = extract_sql(test_case.actual_output)
            gold_sql = (test_case.expected_output or "").strip()

            self.score, self.reason = execution_match(
                db_path, predicted_sql, gold_sql, timeout=self.timeout
            )
            self.success = self.score >= self.threshold
            self.error = None
            if self.verbose_mode:
                print(f"[ExecutionAccuracy] score={self.score} | {self.reason}")
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: "LLMTestCase", *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return bool(self.success)

    @property
    def __name__(self):
        return "Execution Accuracy"


# ===========================================================================
# Auto-teste (não requer DeepEval nem o Spider): valida a lógica end-to-end
# criando um banco SQLite temporário no layout do Spider.
# Rode com:  python -m custom_metrics.execution_accuracy
# ===========================================================================
if __name__ == "__main__":
    import tempfile

    print(f"DeepEval disponível: {_DEEPEVAL_AVAILABLE}\n")

    # --- Testes da extração de SQL ------------------------------------------
    casos_extracao = [
        ("```sql\nSELECT name FROM singer;\n```\nEssa é a resposta.",
         "SELECT name FROM singer"),
        ("Claro! Aqui está:\nSELECT * FROM t WHERE x > 1; -- comentário",
         "SELECT * FROM t WHERE x > 1"),
        ("WITH c AS (SELECT 1) SELECT * FROM c",
         "WITH c AS (SELECT 1) SELECT * FROM c"),
        ("A consulta correta é: select count(*) from singer\nObrigado!",
         "select count(*) from singer"),
    ]
    print("== Extração de SQL ==")
    for raw, esperado in casos_extracao:
        got = extract_sql(raw)
        status = "OK " if got == esperado else "ERRO"
        print(f"  [{status}] {got!r}")
    print()

    # --- Banco temporário no layout do Spider -------------------------------
    tmp = tempfile.mkdtemp()
    db_root = os.path.join(tmp, "database")
    db_id = "concert_singer"
    db_dir = os.path.join(db_root, db_id)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{db_id}.sqlite")

    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE singer (id INTEGER, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1,'Ana',30),(2,'Bia',25),(3,'Caio',40);
    """)
    con.commit()
    con.close()

    # --- Casos de avaliação --------------------------------------------------
    def tc(actual, expected):
        return LLMTestCase(actual_output=actual, expected_output=expected,
                           metadata={"db_id": db_id})

    metric = ExecutionAccuracyMetric(db_root=db_root)

    print("== Execution Accuracy ==")
    # 1) Mesma semântica, ordem das linhas diferente -> acerto (sem ORDER BY)
    s = metric.measure(tc("```sql\nSELECT name FROM singer ORDER BY id DESC\n```",
                          "SELECT name FROM singer"))
    print(f"  [{ 'OK ' if s==1.0 else 'ERRO'}] ordem das linhas ignorada (esperado 1.0): {s} | {metric.reason}")

    # 2) ORDER BY na gold -> a ordem importa -> ordem errada = erro
    s = metric.measure(tc("SELECT name FROM singer ORDER BY age ASC",
                          "SELECT name FROM singer ORDER BY age DESC"))
    print(f"  [{ 'OK ' if s==0.0 else 'ERRO'}] ORDER BY respeitado (esperado 0.0): {s} | {metric.reason}")

    # 3) Resultado correto com texto explicativo em volta -> acerto
    s = metric.measure(tc("A resposta é:\nSELECT count(*) FROM singer;\nEspero ter ajudado!",
                          "SELECT count(*) FROM singer"))
    print(f"  [{ 'OK ' if s==1.0 else 'ERRO'}] count(*) (esperado 1.0): {s} | {metric.reason}")

    # 4) SQL inválida -> erro tratado -> 0.0
    s = metric.measure(tc("SELECT FROM WHERE", "SELECT name FROM singer"))
    print(f"  [{ 'OK ' if s==0.0 else 'ERRO'}] SQL inválida (esperado 0.0): {s} | {metric.reason}")

    # 5) Coluna errada -> resultado divergente -> 0.0
    s = metric.measure(tc("SELECT age FROM singer", "SELECT name FROM singer"))
    print(f"  [{ 'OK ' if s==0.0 else 'ERRO'}] coluna errada (esperado 0.0): {s} | {metric.reason}")