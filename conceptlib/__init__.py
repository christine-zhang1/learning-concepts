"""Shared library for the concept-training pipeline.

Modules:
  - paths:     filesystem roots (env-configurable) and path builders shared by
               the data, training, and evaluation scripts.
  - config:    the TrainingConfig dataclass.
  - csv_utils: long-format results-CSV upsert helpers used by the eval scripts.
  - eval_utils: model loading + bootstrap/timing helpers shared by eval scripts.
"""
