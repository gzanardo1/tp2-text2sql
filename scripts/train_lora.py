# -*- coding: utf-8 -*-
"""
Fase 3 — Fine-tuning LoRA/QLoRA para Text-to-SQL (Spider).

Treina um adaptador LoRA sobre o modelo-base usando QLoRA (4-bit) + TRL SFT.
A loss é calculada APENAS nos turnos do `assistant` (assistant_only_loss),
preservando o sinal de aprendizado na consulta SQL alvo.

Atende ao item 3.3 do enunciado (≥2 configurações distintas de hiperparâmetros)
via --config {a,b}. A configuração de LoRA é igual em ambas; o que varia é a
taxa de aprendizado e o nº de épocas (documentado abaixo).

Uso (Colab T4):
  # Config A: lr alto, 1 época
  python -m scripts.train_lora --config a --output_dir runs/qlora_a

  # Config B: lr menor, 2 épocas
  python -m scripts.train_lora --config b --output_dir runs/qlora_b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

# Telemetria off antes de qualquer import pesado.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Configurações de hiperparâmetros (2 variações OBRIGATÓRIAS — item 3.3)
# Mesmo LoRA, mesma arquitetura; varia learning_rate e num_train_epochs.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    name: str
    learning_rate: float
    num_train_epochs: float


CONFIGS = {
    "a": TrainConfig(name="a", learning_rate=2e-4, num_train_epochs=1.0),
    "b": TrainConfig(name="b", learning_rate=1e-4, num_train_epochs=2.0),
}


# ---------------------------------------------------------------------------
# Configuração de LoRA (item 3.2 — explicitamente documentada)
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Atende ao enunciado (que cita "q_proj, v_proj" como exemplo); incluímos k/o
# para cobrir todo o bloco de atenção. Em T4 com QLoRA, é um bom equilíbrio.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def log_environment():
    """Bloco para o relatório: hardware, VRAM e versões."""
    print("=" * 60)
    print("AMBIENTE DE TREINAMENTO")
    print("=" * 60)
    import transformers, trl, peft, bitsandbytes
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  trl          {trl.__version__}")
    print(f"  peft         {peft.__version__}")
    print(f"  bitsandbytes {bitsandbytes.__version__}")
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU          {name}  ({vram_gb:.1f} GB)")
    else:
        print("  GPU          (indisponível — treino em CPU não é viável)")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_file", default="data/train_chat.jsonl")
    ap.add_argument("--config", choices=list(CONFIGS), required=True,
                    help="a (lr=2e-4, 1 época) ou b (lr=1e-4, 2 épocas)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=1024,
                    help="máx. de tokens por exemplo; trunca esquemas longos")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="batch efetivo = batch_size * grad_accum")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 limita exemplos de treino (smoke test)")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    set_seed(args.seed)  # reprodutibilidade
    log_environment()
    print(f"\nConfig {cfg.name}: lr={cfg.learning_rate}, epochs={cfg.num_train_epochs}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, "
          f"target_modules={TARGET_MODULES}\n")

    # --- Tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Modelo base em 4-bit (QLoRA) --------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 não suporta bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatível com gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # --- LoRA --------------------------------------------------------------
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Dataset (chat format, campo `messages`) ---------------------------
    ds = load_dataset("json", data_files=args.train_file, split="train")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Exemplos de treino: {len(ds)}")

    # --- SFTConfig ---------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        assistant_only_loss=True,  # loss apenas nos turnos do assistant
        packing=False,             # incompatível com assistant_only_loss
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataset_text_field=None,   # usa `messages` (conversacional)
    )

    # --- Treinamento -------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Confirma quantos parâmetros são treináveis (deve ser <1% do total)
    trainer.model.print_trainable_parameters()

    trainer.train()

    # --- Salva adaptador final + manifest para o relatório -----------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.model,
        "config": cfg.name,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": TARGET_MODULES},
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "seed": args.seed,
        "n_train_examples": len(ds),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else "cpu"),
    }
    with open(os.path.join(args.output_dir, "training_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nAdaptador LoRA salvo em: {args.output_dir}")
    print(f"Manifest: {os.path.join(args.output_dir, 'training_manifest.json')}")


if __name__ == "__main__":
    main()