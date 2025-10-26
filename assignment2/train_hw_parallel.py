# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
import pathlib
import pickle
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for homework
    load_dataset = None
from transformers.utils import is_flash_attn_2_available
from tqdm.auto import tqdm
from conversation_func import conversation_to_features
from utils import normalize_conversation, QWEN3CHATTEMPLATE
from loss_functions import compute_loss_from_logits

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

CACHE_VERSION = 1

_WORKER_TOKENIZER = None
_WORKER_TRUNCATION = "right"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Set True/False to force enabling/disabling flash attention v2. Leave as None for auto detection."
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    conversation_field: str = field(
        default="conversations",
        metadata={"help": "Key name that holds the message list inside each record."},
    )
    max_rounds: int = field(
        default=0,
        metadata={"help": "Maximum number of user-assistant rounds to keep per conversation. Set <=0 to keep all."},
    )
    preprocess_workers: int = field(
        default=0,
        metadata={
            "help": "Number of worker processes for initial preprocessing. Set <=0 to use all available CPU cores."
        },
    )
    truncation_side: str = field(
        default="right",
        metadata={"help": "Truncation direction when sequences exceed max length. Choose from ['left', 'right']."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    min_lr: float = field(default=None)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


SUPPORTED_TRUNCATION = {"left", "right"}


def load_conversation_records(data_path: str, conversation_field: str) -> List[List[Dict[str, str]]]:
    path = pathlib.Path(data_path)
    raw_entries: List = []

    if path.exists():
        if path.is_dir():
            candidate_files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
            if not candidate_files:
                raise FileNotFoundError(
                    f"No JSON or JSONL files found under directory {data_path}"
                )
            path = candidate_files[0]
        try:
            with path.open("r") as f:
                loaded = json.load(f)
        except json.JSONDecodeError:
            with path.open("r") as f:
                loaded = [json.loads(line) for line in f if line.strip()]
        if isinstance(loaded, dict):
            candidates = []
            for value in loaded.values():
                if isinstance(value, list):
                    candidates.extend(value)
            loaded = candidates
        raw_entries = list(loaded)
        rank0_print(f"Loaded dataset from {data_path}.")
    else:
        if load_dataset is None:
            raise ImportError(
                "datasets is required to load remote datasets. Install with `pip install datasets`."
            )
        dataset = load_dataset(data_path, split="train")
        raw_entries = [row for row in dataset]
        rank0_print(f"Loaded dataset {data_path} via datasets.load_dataset.")

    records: List[List[Dict[str, str]]] = []
    for item in raw_entries:
        conversation = item.get(conversation_field)
        if conversation is None:
            continue
        records.append(conversation)

    if not records:
        raise ValueError(
            f"No conversations found in dataset using field '{conversation_field}'."
        )
    return records


def _serialize_feature(entry: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    return {key: value.cpu().tolist() for key, value in entry.items()}


def _build_dataset_feature(feature: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.long) for key, value in feature.items()}


def _sanitize_cache_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return sanitized or "dataset"


def _derive_cache_path(data_path: str) -> pathlib.Path:
    path = pathlib.Path(data_path)
    if path.exists():
        if path.is_dir():
            return path / f"{path.name}_processed.pickle"
        return path.with_name(f"{path.stem}_processed.pickle")
    safe_name = _sanitize_cache_name(data_path)
    return pathlib.Path.cwd() / f"{safe_name}_processed.pickle"


def _data_path_signature(data_path: str) -> Dict[str, Optional[float]]:
    path = pathlib.Path(data_path)
    if path.exists():
        try:
            stat = path.stat()
            return {
                "data_path": str(path.resolve()),
                "data_mtime": getattr(stat, "st_mtime", None),
                "data_size": getattr(stat, "st_size", None),
            }
        except OSError:
            return {"data_path": str(path)}
    return {"data_path": data_path}


def _export_tokenizer_spec(tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}))
    init_kwargs.setdefault("use_fast", False)
    name_or_path = init_kwargs.get("name_or_path") or getattr(
        tokenizer, "name_or_path", None
    )
    if name_or_path is None:
        raise ValueError(
            "Tokenizer must have a valid `name_or_path` to enable multiprocessing-based preprocessing."
        )
    return {
        "name_or_path": name_or_path,
        "init_kwargs": init_kwargs,
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "chat_template": getattr(tokenizer, "chat_template", None),
    }


def _instantiate_tokenizer_from_spec(spec: Dict) -> transformers.PreTrainedTokenizer:
    init_kwargs = dict(spec.get("init_kwargs", {}))
    init_kwargs.setdefault("use_fast", False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["name_or_path"],
        **init_kwargs,
    )
    tokenizer.model_max_length = spec["model_max_length"]
    tokenizer.padding_side = spec["padding_side"]

    pad_token = spec.get("pad_token")
    pad_token_id = spec.get("pad_token_id")
    eos_token = spec.get("eos_token")

    if tokenizer.pad_token_id is None:
        if pad_token is not None:
            if pad_token in tokenizer.get_vocab():
                tokenizer.pad_token = pad_token
            else:
                tokenizer.add_special_tokens({"pad_token": pad_token})
                tokenizer.pad_token = pad_token
        elif eos_token is not None:
            tokenizer.pad_token = eos_token
    elif pad_token is not None and tokenizer.pad_token != pad_token:
        tokenizer.pad_token = pad_token

    chat_template = spec.get("chat_template")
    if chat_template is not None:
        tokenizer.chat_template = chat_template

    return tokenizer


def _worker_tokenizer_init(tokenizer_spec: Dict, truncation: str):
    global _WORKER_TOKENIZER, _WORKER_TRUNCATION
    _WORKER_TOKENIZER = _instantiate_tokenizer_from_spec(tokenizer_spec)
    _WORKER_TRUNCATION = truncation


def _preprocess_worker(item: Tuple[int, List[Dict[str, str]]]) -> Tuple[int, Optional[Dict[str, List[int]]]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Worker tokenizer is not initialized.")
    idx, messages = item
    feature = conversation_to_features(
        messages,
        _WORKER_TOKENIZER,
        _WORKER_TOKENIZER.model_max_length,
        _WORKER_TRUNCATION,
    )
    if feature is None:
        return idx, None
    return idx, _serialize_feature(feature)


def _metadata_signature(
    data_path: str,
    max_rounds: int,
    truncation: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, Optional[str]]:
    signature: Dict[str, Optional[str]] = {
        "max_rounds": max_rounds,
        "truncation": truncation,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
    }
    signature.update(_data_path_signature(data_path))
    return signature


def _try_load_cached_features(
    cache_path: pathlib.Path,
    expected_signature: Dict[str, Optional[str]],
) -> Optional[Tuple[List[Dict[str, List[int]]], Dict[str, Optional[str]]]]:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - defensive path
        rank0_print(f"Failed to load cache at {cache_path}: {exc}. Regenerating.")
        return None

    if payload.get("version") != CACHE_VERSION:
        rank0_print(f"Cache version mismatch for {cache_path}. Regenerating.")
        return None

    metadata = payload.get("metadata", {})
    for key, expected_value in expected_signature.items():
        cached_value = metadata.get(key)
        if cached_value != expected_value:
            rank0_print(
                f"Cache metadata mismatch for key '{key}': expected {expected_value}, found {cached_value}. Regenerating."
            )
            return None

    features = payload.get("features")
    if not features:
        rank0_print(f"Cache at {cache_path} contained no features. Regenerating.")
        return None
    rank0_print(f"Loaded preprocessed features from {cache_path}.")
    return features, metadata


def _preprocess_and_cache_features(
    records: List[List[Dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizer,
    *,
    data_path: str,
    max_rounds: int,
    truncation: str,
    cache_path: pathlib.Path,
    num_workers: int,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Optional[str]]]:
    normalized: List[List[Dict[str, str]]] = []
    dropped_empty = 0
    for turns in records:
        messages = normalize_conversation(turns, max_rounds)
        if not messages:
            dropped_empty += 1
            continue
        normalized.append(messages)

    if not normalized:
        raise ValueError("No usable conversations were found after preprocessing.")

    total = len(normalized)
    features_buffer: List[Optional[Dict[str, List[int]]]] = [None] * total
    dropped_feature = 0

    worker_count = num_workers if num_workers and num_workers > 0 else max(os.cpu_count() or 1, 1)
    worker_count = max(1, min(worker_count, total))

    worker_label = "workers" if worker_count > 1 else "worker"
    rank0_print(f"Preprocessing dataset with {worker_count} {worker_label}.")

    iterator = list(enumerate(normalized))

    if worker_count == 1:
        progress = tqdm(
            total=total,
            desc="Tokenizing conversations",
            disable=local_rank not in (None, 0),
        )
        for idx, messages in iterator:
            feature = conversation_to_features(
                messages,
                tokenizer,
                tokenizer.model_max_length,
                truncation,
            )
            if feature is None:
                dropped_feature += 1
            else:
                features_buffer[idx] = _serialize_feature(feature)
            progress.update(1)
        progress.close()
    else:
        tokenizer_spec = _export_tokenizer_spec(tokenizer)
        progress = tqdm(
            total=total,
            desc=f"Tokenizing conversations (x{worker_count})",
            disable=local_rank not in (None, 0),
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_tokenizer_init,
            initargs=(tokenizer_spec, truncation),
        ) as executor:
            futures = [executor.submit(_preprocess_worker, item) for item in iterator]
            for future in as_completed(futures):
                idx, serialized = future.result()
                if serialized is None:
                    dropped_feature += 1
                else:
                    features_buffer[idx] = serialized
                progress.update(1)
        progress.close()

    features: List[Dict[str, List[int]]] = [
        item for item in features_buffer if item is not None
    ]

    if not features:
        raise ValueError("No usable conversations were found after preprocessing.")

    metadata = _metadata_signature(
        data_path=data_path,
        max_rounds=max_rounds,
        truncation=truncation,
        tokenizer=tokenizer,
    )
    metadata.update(
        {
            "version": CACHE_VERSION,
            "source_records": len(records),
            "retained": len(features),
            "dropped_normalization": dropped_empty,
            "dropped_tokenization": dropped_feature,
            "worker_count": worker_count,
            "worker_mode": "process" if worker_count > 1 else "sequential",
        }
    )

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {"version": CACHE_VERSION, "metadata": metadata, "features": features},
                handle,
            )
        rank0_print(f"Saved preprocessed features to {cache_path}.")
    except Exception as exc:  # pragma: no cover - defensive path
        rank0_print(f"Failed to save cache at {cache_path}: {exc}.")

    return features, metadata


class ConversationDataset(Dataset):
    """Dataset that builds loss masks from multi-turn conversations."""

    def __init__(
        self,
        records: List[List[Dict[str, str]]],
        tokenizer: transformers.PreTrainedTokenizer,
        *,
        max_rounds: int,
        truncation_side: str,
    ):
        truncation = truncation_side.lower()
        if truncation not in SUPPORTED_TRUNCATION:
            raise ValueError(
                f"Unknown truncation option '{truncation_side}'. Supported values: {SUPPORTED_TRUNCATION}"
            )

        self._tokenizer = tokenizer
        self._truncation = truncation
        self._messages: List[List[Dict[str, str]]] = []
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        dropped = 0
        iterable = tqdm(
            records,
            desc="Preprocessing conversations",
            disable=local_rank not in (None, 0),
        )
        for turns in iterable:
            messages = normalize_conversation(turns, max_rounds)
            if not messages:
                dropped += 1
                continue
            self._messages.append(messages)

        if dropped:
            rank0_print(f"Filtered out {dropped} conversations during preprocessing.")

        if not self._messages:
            raise ValueError("No usable conversations were found after preprocessing.")

    def __len__(self):
        return len(self._messages)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        if index in self._cache:
            feature = self._cache[index]
        else:
            messages = self._messages[index]
            feature = conversation_to_features(
                messages,
                self._tokenizer,
                self._tokenizer.model_max_length,
                self._truncation,
            )
            self._cache[index] = feature

        return {
            "input_ids": feature["input_ids"].clone(),
            "labels": feature["labels"].clone(),
            "attention_mask": feature["attention_mask"].clone(),
        }


class PreprocessedConversationDataset(Dataset):
    """Dataset backed by pre-tokenized and serialized features."""

    def __init__(self, features: List[Dict[str, List[int]]]):
        if not features:
            raise ValueError("No features provided to build the dataset.")
        self._features: List[Dict[str, torch.Tensor]] = [
            _build_dataset_feature(feature) for feature in features
        ]

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        feature = self._features[index]
        return {
            "input_ids": feature["input_ids"].clone(),
            "labels": feature["labels"].clone(),
            "attention_mask": feature["attention_mask"].clone(),
        }


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    truncation = data_args.truncation_side.lower()
    if truncation not in SUPPORTED_TRUNCATION:
        raise ValueError(
            f"Unknown truncation option '{data_args.truncation_side}'. Supported values: {SUPPORTED_TRUNCATION}"
        )

    if data_args.lazy_preprocess:
        rank0_print("lazy_preprocess=True but data is still preprocessed eagerly for clarity.")

    records = load_conversation_records(data_args.data_path, data_args.conversation_field)
    cache_path = _derive_cache_path(data_args.data_path)
    expected_signature = _metadata_signature(
        data_path=data_args.data_path,
        max_rounds=data_args.max_rounds,
        truncation=truncation,
        tokenizer=tokenizer,
    )

    cached = _try_load_cached_features(cache_path, expected_signature)
    if cached is not None:
        features, metadata = cached
    else:
        features, metadata = _preprocess_and_cache_features(
            records,
            tokenizer,
            data_path=data_args.data_path,
            max_rounds=data_args.max_rounds,
            truncation=truncation,
            cache_path=cache_path,
            num_workers=data_args.preprocess_workers,
        )

    retained = metadata.get("retained", len(features))
    if retained != len(features):
        rank0_print(f"Using {len(features)} conversations after preprocessing.")

    np.random.seed(0)
    perm = np.random.permutation(len(features))
    split = max(1, int(len(perm) * 0.98)) if len(perm) > 1 else len(perm)
    train_indices = perm[:split]
    eval_indices = perm[split:] if split < len(perm) else perm[:1]

    train_features = [features[i] for i in train_indices]
    eval_features = [features[i] for i in eval_indices]
    rank0_print(f"#train {len(train_features)}, #eval {len(eval_features)}")

    train_dataset = PreprocessedConversationDataset(train_features)
    eval_dataset = PreprocessedConversationDataset(eval_features)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    local_rank = training_args.local_rank
    flash_available = is_flash_attn_2_available()
    if model_args.flash_attn is True and not flash_available:
        raise RuntimeError(
            "flash_attn was requested but flash_attn_2 is not available in this environment."
        )
    use_flash_attn = flash_available if model_args.flash_attn is None else model_args.flash_attn

    model_kwargs = dict(cache_dir=training_args.cache_dir)
    torch_dtype = None
    if getattr(training_args, "bf16", False):
        torch_dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        torch_dtype = torch.float16
    elif use_flash_attn:
        torch_dtype = torch.bfloat16
        rank0_print("Flash attention requires half precision; defaulting to torch.bfloat16.")
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
        rank0_print(f"Loading model in {torch_dtype} precision.")

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        rank0_print("Using flash_attention_2 attention implementation.")

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
    except TypeError as type_err:
        if (
            use_flash_attn
            and "unexpected keyword argument 'attn_implementation'" in str(type_err)
        ):
            rank0_print("Falling back to legacy use_flash_attention_2 flag.")
            model_kwargs.pop("attn_implementation", None)
            model_kwargs["use_flash_attention_2"] = True
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                **model_kwargs,
            )
        else:
            raise
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.model_max_length = training_args.model_max_length

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id for batching.")

    if "smollm2" in model_args.model_name_or_path.lower():
        rank0_print("Smollm2 detected, using custom chat template")
        tokenizer.chat_template = QWEN3CHATTEMPLATE
        # Use right padding for training (left padding causes issues)
        tokenizer.padding_side = "right"

    if "mistral" in model_args.model_name_or_path.lower():
        rank0_print("Mistral with Left Padding Side")
        tokenizer.padding_side = "left"

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_loss_func=compute_loss_from_logits,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
