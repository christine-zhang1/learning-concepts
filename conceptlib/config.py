"""Training configuration for concept-loss fine-tuning.

Defaults below are sensible starting points; train.py overrides several of
these (model_name, dataset, output_dir, wandb_run_name, ablation flags) from
command-line arguments at runtime.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch

from conceptlib.paths import CHECKPOINT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-3B"
    max_length: int = 256
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # Concept-loss configuration
    concept_loss_weight = 0
    max_synonyms_per_content_word = 10
    max_content_words = None  # None means all content words; otherwise sample this many
    target_mass = 0.6
    use_weighted_ce: bool = False
    ce_concept_token_weight: float = 5.0

    # Training configuration
    output_dir: str = os.path.join(
        CHECKPOINT_ROOT, f"concept-weight-{concept_loss_weight}-mass-{target_mass}"
    )
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 7e-5  # LoRA LR of 2e-4 overfits on small datasets
    weight_decay: float = 0.05
    warmup_ratio: float = 0.08
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 50
    save_steps: int = 1
    eval_steps: int = 1
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.2
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Data configuration
    dataset: str = "c4"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    randomized_synonyms: bool = False

    # Weights & Biases configuration
    use_wandb: bool = True
    wandb_project: str = "concept-loss-synonyms"
    wandb_run_name: Optional[str] = f"concept-weight-{concept_loss_weight}"

    # Other configuration
    seed: int = 42
    fp16: bool = False  # set True automatically if bf16 is unsupported
    bf16: bool = False  # set True automatically if supported
    dataloader_num_workers: int = 0

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.fp16 and torch.cuda.is_bf16_supported():
            self.bf16 = True
            logger.info("Using bf16 mixed precision for better GPU utilization")
        elif not self.bf16:
            self.fp16 = True
            logger.info("Using fp16 mixed precision for better GPU utilization")
