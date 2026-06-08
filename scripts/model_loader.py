# -*- coding: utf-8 -*-
"""
Carregamento compartilhado de modelo + tokenizer + adaptador LoRA.

Usado por:
  - scripts.baseline_eval  (Fase 2 / Fase 4: avaliação Text-to-SQL)
  - scripts.eval_mmlu      (Fase 5: regressão de capacidade)
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str,
               adapter: Optional[str] = None,
               load_in_4bit: bool = False):
    """Carrega modelo (opcionalmente 4-bit) e, se informado, aplica adaptador LoRA.

    Para modelos fine-tuned (Fase 4 e Fase 5), passe `adapter=<output_dir do
    train_lora.py>` — o PEFT carregará as matrizes LoRA por cima do modelo-base.
    """
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model