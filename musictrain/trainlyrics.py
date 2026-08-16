"""LoRA fine-tuning of a small open LLM on the lyric dataset.

``train-lyrics`` teaches a base chat model (default Qwen2.5-1.5B-Instruct —
fits comfortably on Apple Silicon MPS) to write in the style of the artists in
``lyrics/``. It consumes the instruction files written by
``lyricdataset train-files`` (``metadata/lyrics_train_instructions.jsonl``),
trains with PEFT LoRA + the HF Trainer, and saves a loadable adapter that the
lyrics engine picks up via ``MUSICTRAIN_LLM_MODEL_PATH``.

Everything heavy (torch / transformers / peft) is imported lazily so the rest
of the package (and CI) never pays for it — ``--dry-run`` validates the
dataset without loading a model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from . import console
from .logging import get_logger

log = get_logger("trainlyrics")

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _import_train_deps() -> tuple:
    try:
        import torch  # noqa: F401
        from peft import LoraConfig, get_peft_model  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        return (torch, AutoModelForCausalLM, AutoTokenizer, Trainer,
                TrainingArguments, LoraConfig, get_peft_model)
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "train-lyrics needs torch, transformers and peft — "
            f"install with `uv pip install torch transformers peft` ({exc})"
        ) from exc


def resolve_device() -> str:
    """cuda if available, else mps on Apple Silicon, else cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_dir(out: Path, tag: str = "") -> Path:
    """A timestamped, collision-safe run directory under ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{tag or 'lyrics'}-{stamp}"
    run = out / name
    i = 1
    while run.exists():
        run = out / f"{name}-{i}"
        i += 1
    run.mkdir(parents=True, exist_ok=True)
    return run


def _load_examples(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} — run `musictrain lyricdataset split` and "
            "`musictrain lyricdataset train-files` first"
        )
    out: List[dict] = []
    for ln in path.open(encoding="utf-8"):
        if ln.strip():
            out.append(json.loads(ln))
    return out


def _tokenize_example(tokenizer, example: dict, max_len: int) -> Dict[str, List[int]]:
    """Chat example -> (input_ids, labels) with the prompt masked out."""
    msgs = example["messages"]
    prompt = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    body = tokenizer(msgs[-1]["content"], add_special_tokens=False)["input_ids"]
    # trim to fit: keep the prompt start, drop from the middle of the body
    budget = max_len - len(prompt_ids) - 1
    if budget < 8:
        budget = 8
    body = body[:budget]
    input_ids = (prompt_ids + body + [tokenizer.eos_token_id])[:max_len]
    labels = ([-100] * len(prompt_ids) + body + [tokenizer.eos_token_id])[:max_len]
    return {"input_ids": input_ids, "labels": labels}


class _DataCollator:
    def __init__(self, tokenizer):
        self.pad = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    def __call__(self, features: List[dict]) -> dict:
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def train(
    root: Path,
    model: str = DEFAULT_MODEL,
    steps: int = 100,
    lr: float = 2e-4,
    r: int = 8,
    out: str = "checkpoints/lyrics",
    seed: int = 42,
    per_device: int = 1,
    accum: int = 4,
    warmup: int = 10,
    max_len: int = 1024,
    limit: int = 0,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Fine-tune the base model with LoRA on the instruction dataset."""
    root = Path(root)
    train_path = root / "metadata" / "lyrics_train_instructions.jsonl"
    val_path = root / "metadata" / "lyrics_val_instructions.jsonl"
    train_ex = _load_examples(train_path)[:limit] if limit else _load_examples(train_path)
    val_ex = (_load_examples(val_path) if val_path.exists() else [])[: max(limit // 10, 1)] if limit else (_load_examples(val_path) if val_path.exists() else [])
    device = resolve_device()
    run = run_dir(Path(out) if not Path(out).is_absolute() else Path(out), tag=model.split("/")[-1].lower())
    plan = {
        "model": model, "device": device, "steps": steps, "lr": lr,
        "lora_r": r, "train_examples": len(train_ex), "val_examples": len(val_ex),
        "limit": limit, "run_dir": str(run),
    }
    if dry_run or not train_ex:
        console.info("DRY RUN — no model loaded:")
        for k, v in plan.items():
            console.info(f"  {k}: {v}")
        if not train_ex:
            console.warn("no training examples found — run `musictrain lyricdataset train-files`")
        return plan

    (torch, AutoModelForCausalLM, AutoTokenizer, Trainer,
     TrainingArguments, LoraConfig, get_peft_model) = _import_train_deps()
    torch.manual_seed(seed)

    console.info(f"Loading {model} on {device}…")
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
    base.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=r, lora_alpha=2 * r, target_modules=_TARGET_MODULES,
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model_pt = get_peft_model(base, peft_cfg)
    if device == "mps":
        model_pt.to("mps")
    model_pt.print_trainable_parameters()

    train_ds = [_tokenize_example(tok, ex, max_len) for ex in train_ex]
    val_ds = [_tokenize_example(tok, ex, max_len) for ex in val_ex]

    args = TrainingArguments(
        output_dir=str(run),
        max_steps=steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        logging_steps=max(1, steps // 20),
        save_strategy="no",
        eval_strategy="steps" if val_ds else "no",
        eval_steps=max(1, steps) if val_ds else None,  # one final eval keeps runtimes bounded
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model_pt,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds or None,
        data_collator=_DataCollator(tok),
    )
    console.info(f"Training {steps} steps in {run}…")
    trainer.train()

    # save adapter + tokenizer + metadata
    model_pt.save_pretrained(str(run))
    tok.save_pretrained(str(run))
    metrics = trainer.evaluate() if val_ds else {"eval_loss": None}
    meta = {
        "base_model": model, "device": device, "seed": seed,
        "steps": steps, "lr": lr, "lora_r": r, "train_examples": len(train_ex),
        "val_examples": len(val_ex), "eval_loss": metrics.get("eval_loss"),
    }
    (run / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.ok(f"Adapter saved -> {run}")

    try:
        import mlflow
        mlflow.set_tracking_uri((root / "mlruns").as_uri())
        with mlflow.start_run(run_name=f"train-lyrics/{model.split('/')[-1]}"):
            mlflow.log_params({k: v for k, v in meta.items() if isinstance(v, (int, float, str))})
            if metrics.get("eval_loss") is not None:
                mlflow.log_metric("eval_loss", float(metrics["eval_loss"]))
            mlflow.log_artifact(str(run), artifact_path="adapter")
    except Exception:  # noqa: BLE001 - mlflow is optional
        log.info("mlflow logging skipped")

    plan.update(meta)
    plan["run_dir"] = str(run)
    return plan
