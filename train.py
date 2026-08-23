"""Fine-tune a causal LM with the concept loss over synonym sets.

The concept loss encourages the model to place probability mass on a *set* of
interchangeable tokens (a content word plus its contextual synonyms) at each
target position, rather than only on the single ground-truth token. The total
loss blends standard cross-entropy with a margin loss on the concept-set mass;
several ablation modes (randomized synonyms, weighted-CE, data augmentation,
train-size / content-word sweeps) are selectable via flags.

Data is read from / checkpoints are written to the locations defined in
conceptlib.paths (configurable via environment variables).
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import random
import json
import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb

from tqdm import tqdm

from conceptlib.config import TrainingConfig
from conceptlib.paths import dataset_dir, adapter_dir, model_suffix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConceptDataset(Dataset):
    """Dataset for loading concept training data with synonyms.

    Each JSON line has the format produced by the synonym pipeline:
        {"input_sequence": "...", "content_word_responses": [{"word": "...", "position": N, "synonyms": [...]}, ...]}

    One dataset item per input_sequence. The concept loss is computed over all content words
    in the sequence during training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        max_samples: Optional[int] = None,
        use_data_augmentation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []
        skipped_position = 0
        skipped_token_mismatch = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading {data_path}"):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                input_sequence = item['input_sequence']
                encoding = tokenizer(
                    input_sequence,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors='pt',
                )
                input_ids_seq = encoding['input_ids'].squeeze(0)
                attention_mask_seq = encoding['attention_mask'].squeeze(0)
                seq_len = int(input_ids_seq.shape[0])

                # Augmented sequences have content words rewritten to synonyms,
                # so the position-indexed tokens no longer align. Train as plain
                # CE on the (already-augmented) input_sequence with no targets.
                if use_data_augmentation:
                    self.data.append({
                        'input_ids': input_ids_seq,
                        'attention_mask': attention_mask_seq,
                        'content_words': [],
                    })
                    continue

                content_words_raw = list(item.get('content_word_responses', item.get('content_words', [])))
                if not content_words_raw:
                    continue

                content_words_processed = []
                for tw in content_words_raw:
                    position = int(tw['position'])
                    if position <= 0 or position >= seq_len:
                        skipped_position += 1
                        continue
                    # Verify the actual mid-sequence token matches what we'd get by
                    # encoding the content word with a leading space. If it doesn't,
                    # the assumed space-prefix convention is off for this case and
                    # the resulting concept set would be inconsistent (target token
                    # in a different vocab slot than the synonym encodings) — skip.
                    target_token_id = int(input_ids_seq[position].item())
                    word = tw['word']
                    expected = tokenizer.encode(' ' + word.strip(), add_special_tokens=False)
                    if not expected or expected[0] != target_token_id:
                        logger.warning(
                            f"Target '{word}' first-token mismatch at position {position}: "
                            f"encoded ' {word.strip()}' -> {expected[:1] if expected else []}, "
                            f"actual token id {target_token_id}; skipping"
                        )
                        skipped_token_mismatch += 1
                        continue

                    seen = {target_token_id}
                    synonym_first_ids = []
                    for synonym in tw.get('synonyms', []):
                        try:
                            encoded = tokenizer.encode(' ' + synonym, add_special_tokens=False)
                        except Exception:
                            continue
                        if encoded and encoded[0] not in seen:
                            seen.add(encoded[0])
                            synonym_first_ids.append(int(encoded[0]))

                    content_words_processed.append({
                        'position': position,
                        'shift_position': position - 1,
                        'target_token_id': target_token_id,
                        'synonym_token_ids': synonym_first_ids,
                    })

                if not content_words_processed:
                    continue

                self.data.append({
                    'input_ids': input_ids_seq,
                    'attention_mask': attention_mask_seq,
                    'content_words': content_words_processed,
                })

        if max_samples and max_samples > 0:
            self.data = self.data[:max_samples]

        logger.info(
            f"Loaded {len(self.data)} samples from {data_path} "
            f"(skipped {skipped_position} content words out of range, "
            f"{skipped_token_mismatch} content words with token mismatch)"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'content_words': item['content_words'],
        }


class ConceptLossTrainer(Trainer):
    """Custom Trainer that uses concept loss based on synonyms."""

    def __init__(self, concept_loss_weight=0, max_synonyms_per_content_word=None, max_content_words=None, target_mass=0.3, model_name="", randomized_synonyms=False, use_weighted_ce=False, ce_concept_token_weight=5.0, **kwargs):
        super().__init__(**kwargs)
        self.concept_loss_weight = concept_loss_weight
        self.max_synonyms_per_content_word = max_synonyms_per_content_word
        self.max_content_words = max_content_words
        self.target_mass = target_mass
        self.use_weighted_ce = use_weighted_ce
        self.ce_concept_token_weight = ce_concept_token_weight
        self.processing_class = kwargs.get('processing_class')
        self.model_name = model_name or ""
        self._eval_ce_losses = []
        self._eval_concept_losses = []
        self._eval_perplexities = []
        self._is_evaluating = False
        self._last_eval_metrics = {}
        self.randomized_synonyms = randomized_synonyms

    def get_eval_dataloader(self, eval_dataset=None):
        """Create eval dataloader that adds labels for loss computation."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        base_collator = self.data_collator
        def collate_with_labels(examples):
            batch = base_collator(examples)
            # Add labels so HF eval can compute eval_loss without affecting training
            batch['labels'] = batch['input_ids'].clone()
            return batch

        dataloader_params = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": collate_with_labels,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
            "drop_last": self.args.dataloader_drop_last,
        }

        dl = torch.utils.data.DataLoader(eval_dataset, **dataloader_params)
        return self.accelerator.prepare(dl)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation using concept loss.
        Inputs should contain: input_ids, attention_mask, content_words (list of lists)
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        # content_words_list[b] is a list of pre-tokenized target dicts for sequence b
        content_words_list = inputs.get('content_words', [])

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.get("logits", outputs[0])

        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_len, vocab_size = logits.shape
        shift_len = seq_len - 1

        per_token_ce = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(batch_size, shift_len)  # [B, S-1]

        valid = shift_labels.ne(-100)  # [B, S-1]

        # Skip concept-loss work entirely when its weight is 0 and we're not in
        # weighted-CE mode (which needs concept timesteps for its own weighting).
        skip_concept_work = (self.concept_loss_weight == 0) and not self.use_weighted_ce

        if skip_concept_work:
            concept_loss = torch.tensor(0.0, device=logits.device)
            N_total = valid.float().sum().clamp(min=1e-8)
            ce_loss = (per_token_ce * valid.float()).sum() / N_total
            training_loss = ce_loss
        else:
            flat_b_idx, flat_shift_idx, flat_concept_ids = self._build_concept_targets(
                content_words_list, batch_size, shift_len
            )

            concept_loss = self._compute_concept_loss_vectorized(
                logits, flat_b_idx, flat_shift_idx, flat_concept_ids
            )

            if self.use_weighted_ce:
                ce_weights = torch.ones(batch_size, shift_len, device=logits.device, dtype=per_token_ce.dtype)
                if flat_b_idx:
                    b_t = torch.tensor(flat_b_idx, device=logits.device, dtype=torch.long)
                    s_t = torch.tensor(flat_shift_idx, device=logits.device, dtype=torch.long)
                    ce_weights[b_t, s_t] = self.ce_concept_token_weight
                denom = (ce_weights * valid.float()).sum().clamp(min=1e-8)
                ce_loss = (per_token_ce * ce_weights * valid.float()).sum() / denom
                training_loss = ce_loss
            else:
                target_mask = torch.zeros(batch_size, shift_len, dtype=torch.bool, device=logits.device)
                if flat_b_idx:
                    b_t = torch.tensor(flat_b_idx, device=logits.device, dtype=torch.long)
                    s_t = torch.tensor(flat_shift_idx, device=logits.device, dtype=torch.long)
                    target_mask[b_t, s_t] = True

                non_target_valid = valid & ~target_mask
                target_valid = valid & target_mask

                N_total = valid.float().sum().clamp(min=1e-8)
                N_non_target = non_target_valid.float().sum()
                N_target = target_valid.float().sum()

                ce_loss = (per_token_ce * valid.float()).sum() / N_total
                ce_loss_non_targets = (per_token_ce * non_target_valid.float()).sum() / N_non_target.clamp(min=1e-8)
                ce_loss_targets = (per_token_ce * target_valid.float()).sum() / N_target.clamp(min=1e-8)

                training_loss = (
                    N_non_target * ce_loss_non_targets
                    + (1 - self.concept_loss_weight) * N_target * ce_loss_targets
                ) / N_total + self.concept_loss_weight * concept_loss

        ce_loss_val = ce_loss.detach().item()
        concept_loss_val = concept_loss.detach().item()
        perplexity = torch.exp(ce_loss).detach().item()

        if self._is_evaluating:
            self._eval_ce_losses.append(ce_loss_val)
            self._eval_concept_losses.append(concept_loss_val)
            self._eval_perplexities.append(perplexity)
        else:
            if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
                self.log({
                    "ce_loss": ce_loss_val,
                    "concept_loss": concept_loss_val,
                    "perplexity": perplexity,
                })

        if return_outputs:
            return training_loss, {"logits": logits}
        return training_loss

    def _build_concept_targets(self, content_words_list, batch_size, shift_len):
        """
        Walk pre-tokenized content words once and return flat lists describing each
        valid (sequence, content-word) pair:
            flat_b_idx[i]        -> batch index
            flat_shift_idx[i]    -> shift position (= original position - 1)
            flat_concept_ids[i]  -> list of token ids in the concept set

        Applies max_content_words subsampling and the max_synonyms_per_content_word /
        randomized_synonyms truncation rules.
        """
        max_synonyms = self.max_synonyms_per_content_word
        max_content_words = self.max_content_words

        flat_b_idx = []
        flat_shift_idx = []
        flat_concept_ids = []

        for b in range(batch_size):
            content_words = content_words_list[b]

            if max_content_words != 'all' and content_words:
                if max_content_words == "half":
                    k = len(content_words) // 2
                elif max_content_words == "quarter":
                    k = len(content_words) // 4
                elif max_content_words == "one":
                    k = 1
                rng = random.Random(self.state.global_step * batch_size + b)
                content_words = rng.sample(content_words, k)

            for tw in content_words:
                shift_pos = tw['shift_position']
                if shift_pos < 0 or shift_pos >= shift_len:
                    continue

                synonyms = tw['synonym_token_ids']
                if self.randomized_synonyms:
                    # Randomized mode: exclude the target token and admit up to
                    # max_synonyms + 1 synonyms.
                    limit = (max_synonyms + 1) if max_synonyms is not None else None
                    ids = list(synonyms[:limit]) if limit is not None else list(synonyms)
                else:
                    ids = [tw['target_token_id']]
                    limit = max_synonyms
                    if limit is not None:
                        ids.extend(synonyms[:limit])
                    else:
                        ids.extend(synonyms)

                if not ids:
                    continue

                flat_b_idx.append(b)
                flat_shift_idx.append(shift_pos)
                flat_concept_ids.append(ids)

        return flat_b_idx, flat_shift_idx, flat_concept_ids

    def _compute_concept_loss_vectorized(self, logits, flat_b_idx, flat_shift_idx, flat_concept_ids):
        """
        Vectorized concept loss:
          - Gather logits only at target shift positions: [num_targets, V].
          - Softmax once over that small tensor.
          - Pad concept-id lists into [num_targets, max_concept] and gather mass.
          - Apply margin (hinge) loss at target_mass.
        """
        if not flat_b_idx:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        device = logits.device
        b_t = torch.tensor(flat_b_idx, device=device, dtype=torch.long)
        s_t = torch.tensor(flat_shift_idx, device=device, dtype=torch.long)

        # logits[b, shift_pos] = logits at index shift_pos predicts token at shift_pos+1.
        target_logits = logits[b_t, s_t]                    # [num_targets, V]
        target_probs = F.softmax(target_logits.float(), dim=-1)

        num_targets = len(flat_concept_ids)
        max_concept = max(len(ids) for ids in flat_concept_ids)
        padded_ids = torch.zeros(num_targets, max_concept, dtype=torch.long, device=device)
        valid_mask = torch.zeros(num_targets, max_concept, dtype=torch.bool, device=device)
        for i, ids in enumerate(flat_concept_ids):
            n = len(ids)
            padded_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=device)
            valid_mask[i, :n] = True

        gathered = target_probs.gather(1, padded_ids)        # [num_targets, max_concept]
        mass = (gathered * valid_mask.float()).sum(dim=1)    # [num_targets]

        # Margin loss: zero contribution above target_mass, -log(mass) below.
        # torch.where evaluates both branches; clamp keeps log finite either way.
        neg_log = -torch.log(mass.clamp(min=1e-9))
        loss_per_target = torch.where(
            mass >= self.target_mass,
            torch.zeros_like(mass),
            neg_log,
        )
        return loss_per_target.mean()

    def compute_metrics(self, eval_pred):
        """Compute and return evaluation metrics from accumulated values."""
        metrics = {}

        if self._eval_ce_losses:
            metrics["ce_loss"] = sum(self._eval_ce_losses) / len(self._eval_ce_losses)
            metrics["perplexity"] = sum(self._eval_perplexities) / len(self._eval_perplexities)

        if self._eval_concept_losses:
            metrics["concept_loss"] = sum(self._eval_concept_losses) / len(self._eval_concept_losses)

        self._last_eval_metrics = metrics.copy()
        return metrics

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation loop to accumulate and return custom metrics."""
        self._is_evaluating = True
        self._eval_ce_losses = []
        self._eval_concept_losses = []
        self._eval_perplexities = []

        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # Aggregate the accumulated eval metrics.
        metrics = {}
        if self._eval_ce_losses:
            metrics["ce_loss"] = sum(self._eval_ce_losses) / len(self._eval_ce_losses)
            metrics["perplexity"] = sum(self._eval_perplexities) / len(self._eval_perplexities)
            logger.info(f"Accumulated {len(self._eval_ce_losses)} eval batches for ce_loss")

        if self._eval_concept_losses:
            metrics["concept_loss"] = sum(self._eval_concept_losses) / len(self._eval_concept_losses)

        self._last_eval_metrics = metrics.copy()

        if hasattr(output, 'metrics') and metrics:
            output.metrics.update(metrics)
            logger.info(f"Custom evaluation metrics in output.metrics: {metrics}")

        if metrics and wandb.run is not None:
            metrics_to_log = {}
            for key, value in metrics.items():
                metrics_to_log[f"eval/{key}"] = value
            wandb.log(metrics_to_log, step=self.state.global_step, commit=True)
            logger.info(f"Direct wandb.log from evaluation_loop: {metrics_to_log}")

        self._is_evaluating = False
        self._eval_ce_losses = []
        self._eval_perplexities = []
        self._eval_concept_losses = []

        return output


class EvalMetricsCallback(TrainerCallback):
    """Callback to ensure custom evaluation metrics are logged to wandb."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Add custom metrics to logs when Trainer logs evaluation results."""
        if logs and ('eval_loss' in logs or any(k.startswith('eval/') for k in logs.keys())):
            trainer = kwargs.get('trainer')
            if trainer and hasattr(trainer, '_last_eval_metrics') and trainer._last_eval_metrics:
                for key, value in trainer._last_eval_metrics.items():
                    logs[f"eval/{key}"] = value
                logger.info(f"Callback on_log: Added custom metrics to logs: {trainer._last_eval_metrics}")

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Log directly to wandb after evaluation completes."""
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, '_last_eval_metrics') and trainer._last_eval_metrics:
            if wandb.run is not None:
                metrics_to_log = {}
                for key, value in trainer._last_eval_metrics.items():
                    metrics_to_log[f"eval/{key}"] = value
                wandb.log(metrics_to_log, step=state.global_step, commit=True)
                logger.info(f"Callback on_evaluate: Logged directly to wandb: {metrics_to_log}")
            else:
                logger.warning("wandb.run is None, cannot log metrics")


def collate_fn(examples, tokenizer):
    """Custom collate function for concept dataset."""
    input_ids = pad_sequence(
        [ex['input_ids'] for ex in examples],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [ex['attention_mask'] for ex in examples],
        batch_first=True,
        padding_value=0
    )

    # List of lists: content_words_list[b] = [{word, position, synonyms}, ...] for sequence b
    content_words_list = [ex['content_words'] for ex in examples]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'content_words': content_words_list,
    }


def setup_wandb(config: TrainingConfig):
    """Setup Weights & Biases logging"""
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
            tags=["concept-loss", "synonyms", "training"]
        )
        logger.info("Weights & Biases initialized")


def setup_model_and_tokenizer(config: TrainingConfig):
    """Setup model and tokenizer with optional quantization and LoRA"""

    logger.info(f"Loading tokenizer from {config.model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    quantization_config = None
    if config.use_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )

    logger.info(f"Loading model from {config.model_name}...")

    if config.use_4bit:
        dtype = None
    elif config.bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    if config.use_lora:
        logger.info("Setting up LoRA...")

        if config.use_4bit:
            # prepare_model_for_kbit_training enables gradient checkpointing by
            # default; keep it off here.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("Using model without LoRA")

    return model, tokenizer


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a language model with concept loss')
    parser.add_argument(
        '--concept-loss-weight',
        type=float,
        default=None,
        help='Weight for concept loss (default: use value from config)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name or path of the model to use (default: use value from config)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Name of the dataset to use (default: use value from config)'
    )
    parser.add_argument(
        '--randomized-synonyms',
        action='store_true',
        default=False,
        help='Whether to use randomized synonyms (default: False)'
    )
    parser.add_argument(
        '--target-mass',
        type=float,
        default=None,
        help='Target mass for concept loss (default: use value from config)'
    )
    parser.add_argument(
        '--max-content-words',
        type=str,
        default=None,
        help='Maximum number of content words to use (default: use value from config)'
    )
    parser.add_argument(
        '--use-weighted-ce',
        action='store_true',
        default=False,
        help='Use weighted CE: higher loss weight at timesteps that predict concept target tokens (same positions as concept loss). Set concept-loss-weight 0 to train on CE only while still logging concept loss.'
    )
    parser.add_argument(
        '--ce-concept-token-weight',
        type=float,
        default=None,
        help='CE multiplier on concept target timesteps when --use-weighted-ce is set (default: config.ce_concept_token_weight)'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        required=True,
        choices=['embedding', 'prompting'],
        help='Synonym source / dataset type; part of the data and checkpoint paths.'
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Maximum number of train samples to use (default: use value from config)'
    )
    parser.add_argument(
        '--num-train-epochs',
        type=int,
        default=None,
        help='Number of train epochs (default: use value from config)'
    )
    parser.add_argument(
        '--use-data-augmentation',
        action='store_true',
        default=False,
        help='Use data augmentation (default: False)'
    )
    return parser.parse_args()


def main():
    """Main training function"""

    args = parse_args()
    config = TrainingConfig()

    # Override config with the command-line arguments that were provided.
    if args.concept_loss_weight is not None:
        logger.info(f"Overriding concept_loss_weight from command line: {args.concept_loss_weight}")
        config.concept_loss_weight = args.concept_loss_weight

    if args.model_name is not None:
        logger.info(f"Overriding model_name from command line: {args.model_name}")
        config.model_name = args.model_name

    if args.dataset is not None:
        logger.info(f"Overriding dataset from command line: {args.dataset}")
        config.dataset = args.dataset

    if args.randomized_synonyms:
        logger.info(f"Overriding randomized_synonyms from command line: {args.randomized_synonyms}")
        config.randomized_synonyms = args.randomized_synonyms

    if args.max_content_words is not None:
        logger.info(f"Overriding max_content_words from command line: {args.max_content_words}")
        config.max_content_words = args.max_content_words

    if args.target_mass is not None:
        logger.info(f"Overriding target_mass from command line: {args.target_mass}")
        config.target_mass = args.target_mass

    if args.use_weighted_ce:
        config.use_weighted_ce = True
    if args.ce_concept_token_weight is not None:
        logger.info(f"Overriding ce_concept_token_weight from command line: {args.ce_concept_token_weight}")
        config.ce_concept_token_weight = args.ce_concept_token_weight

    if args.max_train_samples is not None:
        logger.info(f"Overriding max_train_samples from command line: {args.max_train_samples}")
        config.max_train_samples = args.max_train_samples

    if args.num_train_epochs is not None:
        logger.info(f"Overriding num_train_epochs from command line: {args.num_train_epochs}")
        config.num_train_epochs = args.num_train_epochs


    # Resolve the data paths, checkpoint output_dir, and W&B run name. The
    # checkpoint dir uses the exact same builder the eval scripts rely on
    # (conceptlib.paths.adapter_dir), so training output and evaluation lookup
    # can never drift apart.
    dataset_name = config.dataset
    dataset_type = args.dataset_type
    suffix = model_suffix(config.model_name)
    content_word_tag = "all_content_words" if config.max_content_words is None else f"{config.max_content_words}_content_words"

    data_base_dir = dataset_dir(dataset_name, config.model_name, dataset_type)
    val_path = os.path.join(data_base_dir, "synonyms_val.jsonl")

    # Pick the training file and a short run-name tag for the active ablation.
    if config.randomized_synonyms:
        train_path = os.path.join(data_base_dir, "synonyms_train_randomized.jsonl")
        run_tag = "randomized"
    elif config.use_weighted_ce:
        train_path = os.path.join(data_base_dir, "synonyms_train.jsonl")
        run_tag = f"wce-{config.ce_concept_token_weight}"
    elif config.max_train_samples is not None:
        train_path = os.path.join(data_base_dir, "synonyms_train.jsonl")
        run_tag = f"max-train-samples-{config.max_train_samples}"
    elif args.use_data_augmentation:
        train_path = os.path.join(data_base_dir, "synonyms_train_aug5x.jsonl")
        run_tag = f"aug_{config.num_train_epochs}epochs"
    else:
        train_path = os.path.join(data_base_dir, "synonyms_train.jsonl")
        run_tag = None

    config.output_dir = adapter_dir(
        dataset_name, config.model_name, dataset_type,
        max_content_words=config.max_content_words,
        randomized_synonyms=config.randomized_synonyms,
        use_weighted_ce=config.use_weighted_ce,
        ce_concept_token_weight=config.ce_concept_token_weight,
        max_train_samples=config.max_train_samples,
        use_data_augmentation=args.use_data_augmentation,
        num_train_epochs=config.num_train_epochs,
        concept_loss_weight=config.concept_loss_weight,
    )

    run_name_parts = [dataset_name, suffix, dataset_type, content_word_tag]
    if run_tag:
        run_name_parts.append(run_tag)
    run_name_parts.append(f"concept-weight-{config.concept_loss_weight}")
    config.wandb_run_name = "-".join(run_name_parts)

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info(f"Using concept_loss_weight: {config.concept_loss_weight}")
    logger.info(f"Using model_name: {config.model_name}")
    logger.info(f"Using dataset: {config.dataset}")
    logger.info(f"Using randomized_synonyms: {config.randomized_synonyms}")
    logger.info(f"Using target_mass: {config.target_mass}")
    logger.info(f"Using train data path: {train_path}")
    logger.info(f"Using val data path: {val_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"WandB run name: {config.wandb_run_name}")
    logger.info(f"Using max_synonyms_per_content_word: {config.max_synonyms_per_content_word}")
    logger.info(f"Using max_content_words: {config.max_content_words}")
    logger.info(f"Using use_weighted_ce: {config.use_weighted_ce} (ce_concept_token_weight={config.ce_concept_token_weight})")
    logger.info(f"Using max_train_samples: {config.max_train_samples}")
    logger.info(f"Using num_train_epochs: {config.num_train_epochs}")
    logger.info(f"Using use_data_augmentation: {args.use_data_augmentation}")

    if args.use_data_augmentation:
        assert config.concept_loss_weight == 0, (
            "use_data_augmentation requires concept_loss_weight == 0 "
            f"(got {config.concept_loss_weight}); augmented sequences have "
            "rewritten target tokens that no longer align with stored positions."
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    setup_wandb(config)
    model, tokenizer = setup_model_and_tokenizer(config)

    logger.info("\nPreparing datasets...")
    train_dataset = ConceptDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_train_samples,
        use_data_augmentation=args.use_data_augmentation,
    )
    eval_dataset = ConceptDataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Eval:  {len(eval_dataset)} samples")

    def data_collator(examples):
        return collate_fn(examples, tokenizer)

    eval_strategy = config.evaluation_strategy if eval_dataset else "no"
    if config.early_stopping and not eval_dataset:
        logger.warning("Early stopping is enabled but no evaluation dataset was provided; disabling early stopping.")
    if config.early_stopping and eval_dataset and eval_strategy == "no":
        logger.warning("Early stopping requires evaluation; overriding eval_strategy to 'steps'.")
        eval_strategy = "steps"
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy=eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=(config.load_best_model_at_end or config.early_stopping) if eval_dataset else False,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name,
        seed=config.seed,
        ddp_find_unused_parameters=False if config.use_lora else None,
        optim="adamw_torch_fused",
    )

    callbacks = []
    if eval_dataset and config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    if eval_dataset:
        callbacks.append(EvalMetricsCallback())

    trainer = ConceptLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        concept_loss_weight=config.concept_loss_weight,
        max_synonyms_per_content_word=config.max_synonyms_per_content_word,
        max_content_words=config.max_content_words,
        target_mass=config.target_mass,
        model_name=config.model_name,
        randomized_synonyms=config.randomized_synonyms,
        use_weighted_ce=config.use_weighted_ce,
        ce_concept_token_weight=config.ce_concept_token_weight,
        callbacks=callbacks,
    )

    logger.info("\n" + "="*50)
    logger.info("Starting training...")
    logger.info("="*50 + "\n")

    trainer.train()

    logger.info(f"\nSaving final model to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")

    if config.use_wandb:
        wandb.finish()

    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
