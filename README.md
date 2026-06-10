# TP2 — Fine-tuning para Text-to-SQL com QLoRA

Fine-tuning supervisionado de um LLM para a tarefa **Text-to-SQL** no benchmark **Spider**, usando **QLoRA** (quantização em 4 bits + adaptadores LoRA). O projeto avalia ganho de desempenho na tarefa-alvo e possível regressão de capacidade geral via MMLU.

## Estrutura do projeto

```
tp2/
├── custom_metrics/
│   └── execution_accuracy.py   # Fase 1: métrica de Execution Accuracy (DeepEval)
├── scripts/
│   ├── prepare_data.py          # Fase 1/3: prepara few-shot e dados de treino
│   ├── baseline_eval.py         # Fase 2/4: avalia modelo no Spider dev split
│   ├── train_lora.py            # Fase 3: fine-tuning QLoRA (configs A e B)
│   ├── eval_mmlu.py             # Fase 5: avalia regressão com MMLU
│   ├── analyze_results.py       # Gera tabelas e figuras dos resultados
│   ├── spider_prompt.py         # Template de prompt e serialização de esquema
│   └── model_loader.py          # Carrega modelo base ou com adaptador LoRA
├── results/                     # CSVs, JSONs e análises geradas
│   └── analysis/
│       ├── tables.md / tables.tex
│       ├── error_examples.md
│       └── figures/             # Gráficos em PDF
└── TP2.pdf                      # Relatório do trabalho
```

## Modelo e configurações

**Modelo base**: `Qwen/Qwen2.5-3B-Instruct`

**Configuração LoRA** (igual nas duas variações):

| Parâmetro | Valor |
|---|---|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `task_type` | CAUSAL_LM |

**Hiperparâmetros de treino** (duas configurações):

| Config | Learning rate | Épocas | Batch efetivo | Scheduler |
|---|---|---|---|---|
| A | 2 × 10⁻⁴ | 1 | 8 | cosine |
| B | 1 × 10⁻⁴ | 1 | 8 | cosine |

Treinamento executado em **Google Colab T4** (16 GB VRAM), com quantização NF4 em 4 bits e `paged_adamw_32bit`. A loss é calculada apenas nos turnos do `assistant` (`assistant_only_loss=True`).

## Resultados

### Execution Accuracy — Spider dev split (1034 exemplos)

| Modelo | EA | Δ (pp) | Δ (% relativo) |
|---|---|---|---|
| Baseline (Qwen2.5-3B) | 49.13% | — | — |
| Fine-tuned config A | **64.02%** | +14.89 | +30.3% |
| Fine-tuned config B | 61.70% | +12.57 | +25.6% |

### MMLU 5-shot — regressão de capacidade geral (150 questões, 3 categorias)

| Categoria (subcategoria) | Baseline | Config A | Config B |
|---|---|---|---|
| STEM (college_computer_science) | 62% | 70% | 64% |
| Humanidades (philosophy) | 56% | 58% | 58% |
| Ciências Sociais (high_school_macroeconomics) | 74% | 76% | 74% |
| **Agregado** | **64%** | **68%** | **65.3%** |

O fine-tuning não provocou regressão mensurável — pelo contrário, ambas as configs apresentaram leve melhora no MMLU, sugerindo que a especialização em SQL não degradou capacidades gerais do modelo.

## Reprodução

### Pré-requisitos

O treinamento requer GPU com suporte a CUDA (recomendado: T4 ou superior). A avaliação pode ser executada localmente com as dependências do `pyproject.toml`.

```bash
# Instalar dependências (ambiente local, sem GPU)
uv sync

# Ou para o Colab, usar o requirements-colab.txt
pip install -r requirements-colab.txt
```

### Fase 1 — Testar a métrica

```bash
python -m custom_metrics.execution_accuracy
```

### Fase 2/3 — Preparar dados do Spider

```bash
# Baixe o Spider em https://yale-nlp.github.io/spider/ e extraia em spider/
python -m scripts.prepare_data --spider_root spider --out data
```

### Fase 3 — Fine-tuning (Colab T4)

```bash
# Config A: lr=2e-4, 1 época
python -m scripts.train_lora --config a --output_dir runs/qlora_a

# Config B: lr=1e-4, 1 época
python -m scripts.train_lora --config b --output_dir runs/qlora_b

# Retomar de checkpoint (se interrompido)
python -m scripts.train_lora --config a --output_dir runs/qlora_a --resume
```

### Fase 2 e 4 — Avaliação no Spider

```bash
# Baseline (sem adaptador)
python -m scripts.baseline_eval \
    --model Qwen/Qwen2.5-3B-Instruct \
    --spider_root spider \
    --few_shot data/few_shot.json \
    --out results/baseline.csv \
    --load_in_4bit

# Fine-tuned config A
python -m scripts.baseline_eval \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapter runs/qlora_a \
    --spider_root spider \
    --few_shot data/few_shot.json \
    --out results/qlora_a.csv \
    --load_in_4bit

# Fine-tuned config B
python -m scripts.baseline_eval \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapter runs/qlora_b \
    --spider_root spider \
    --few_shot data/few_shot.json \
    --out results/qlora_b.csv \
    --load_in_4bit
```

### Fase 5 — Avaliação MMLU

```bash
python -m scripts.eval_mmlu --model Qwen/Qwen2.5-3B-Instruct \
    --out results/mmlu_base.json --load_in_4bit

python -m scripts.eval_mmlu --model Qwen/Qwen2.5-3B-Instruct \
    --adapter runs/qlora_a --out results/mmlu_qlora_a.json --load_in_4bit

python -m scripts.eval_mmlu --model Qwen/Qwen2.5-3B-Instruct \
    --adapter runs/qlora_b --out results/mmlu_qlora_b.json --load_in_4bit
```

### Análise consolidada

```bash
# Gera tables.md, tables.tex, error_examples.md e figures/*.pdf
uv run python -m scripts.analyze_results
```

### Testes unitários

```bash
uv run pytest scripts/test_execution_accuracy.py -v
```

## Detalhes de implementação

**Métrica (Fase 1)**: A `ExecutionAccuracyMetric` executa tanto a consulta gerada quanto a consulta gold em modo somente-leitura (`?mode=ro`) no SQLite do Spider. A comparação é feita como multiconjunto (insensível à ordem de linhas), exceto quando a gold contém `ORDER BY`, caso em que a ordem é respeitada.

**Extração de SQL**: A saída bruta do modelo é tratada por `extract_sql`, que remove prefixos explicativos, blocos markdown e texto após o `;`, tornando a avaliação robusta a modelos verbosos.

**Dados de treino**: O script `prepare_data.py` converte o Spider train split para chat format (campo `messages`) com prompt zero-shot, usando o mesmo `SYSTEM_PROMPT` e `render_user_turn` da avaliação — garantindo alinhamento treino/inferência.

**Few-shot fixo**: Os 3 exemplos few-shot usados no baseline e na avaliação são selecionados deterministicamente (bancos com menos tabelas, ordem alfabética estável) e congelados em `data/few_shot.json`.
