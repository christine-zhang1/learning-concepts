"""Filesystem roots and path builders shared across the pipeline.

All locations are configurable through environment variables so the code
contains no machine-specific absolute paths:

  CONCEPT_DATA_ROOT        preprocessed datasets (jsonl files)   default: ./datasets
  CONCEPT_CHECKPOINT_ROOT  trained LoRA adapter checkpoints      default: ./checkpoints
  CONCEPT_RESULTS_ROOT     evaluation result CSVs                default: ./results

(The data-root default is ./datasets, not ./data, so generated files never land
inside the data/ source package.)

Directory conventions (relative to the roots above):

  datasets
    {dataset}/{model}/combined.jsonl                  # get_content_words.py
    {dataset}/{model}/embedding/...                   # embedding_synonyms.py
    {dataset}/{model}/prompting/...                   # llm_synonyms.py
    {dataset}/{model}/{dataset_type}/synonyms_*.jsonl # train/eval splits

  checkpoints
    {dataset}/{model}/{dataset_type}/{content_tag}/[{variant}/]concept-weight-{w}

where {model} is the lowercased final path segment of the HF model name and
{dataset_type} is one of {"embedding", "prompting"}.
"""
import os

DATA_ROOT = os.environ.get("CONCEPT_DATA_ROOT", "datasets")
CHECKPOINT_ROOT = os.environ.get("CONCEPT_CHECKPOINT_ROOT", "checkpoints")
RESULTS_ROOT = os.environ.get("CONCEPT_RESULTS_ROOT", "results")


def model_suffix(model_name: str) -> str:
    """Lowercased final segment of a HF model name (e.g. 'Qwen/Qwen3-4B-Base' -> 'qwen3-4b-base')."""
    return model_name.split("/")[-1].lower()


def dataset_dir(dataset: str, model_name: str, dataset_type: str | None = None) -> str:
    """Directory holding the preprocessed jsonl files for a (dataset, model[, type])."""
    parts = [DATA_ROOT, dataset, model_suffix(model_name)]
    if dataset_type is not None:
        parts.append(dataset_type)
    return os.path.join(*parts)


def adapter_dir(
    dataset: str,
    model_name: str,
    dataset_type: str,
    *,
    max_content_words=None,
    randomized_synonyms: bool = False,
    use_weighted_ce: bool = False,
    ce_concept_token_weight=None,
    max_train_samples=None,
    use_data_augmentation: bool = False,
    num_train_epochs=None,
    concept_loss_weight=None,
) -> str:
    """Build the checkpoint directory for a training run / adapter.

    The variant subdirectory is selected from the (mutually exclusive) ablation
    flags, matching the layout written by train.py. When `concept_loss_weight`
    is given, the final `concept-weight-{w}` segment is appended; otherwise the
    variant directory itself is returned.
    """
    content_tag = f"{max_content_words or 'all'}_content_words"
    base = os.path.join(CHECKPOINT_ROOT, dataset, model_suffix(model_name),
                        dataset_type, content_tag)

    if randomized_synonyms:
        sub = os.path.join(base, "randomized_synonyms")
    elif use_weighted_ce:
        sub = os.path.join(base, f"wce-{ce_concept_token_weight}")
    elif max_train_samples is not None:
        sub = os.path.join(base, f"max-train-samples-{max_train_samples}")
    elif use_data_augmentation:
        sub = os.path.join(base, f"aug_{num_train_epochs}epochs")
    else:
        sub = base

    if concept_loss_weight is None:
        return sub
    return os.path.join(sub, f"concept-weight-{concept_loss_weight}")
