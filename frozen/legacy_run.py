"""Reproducible, single-GPU pilot. Run from any directory; see README.txt."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def numeric(text):
    # Same acceptance grammar as the proposal's EDA; no stripping/rewriting.
    text = text.removesuffix(".")
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1]
    matches = list(re.finditer(r"[0-9]+", text))
    if not matches:
        return False
    if len(matches) == 1:
        parts = [text]
    else:
        sep = text[matches[0].end():matches[1].start()]
        if not sep or sep.strip() not in ("", ",", ";"):
            return False
        parts = text.split(sep)
    return 1 <= len(parts) <= 10 and all(re.fullmatch(r"[0-9]+", p) and int(p) <= 999 for p in parts)


def messages(question, system):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]


def encode_training(tokenizer, question, response):
    prompt = messages(question, CFG["student_system_prompt"])
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": response}], tokenize=True)
    if ids[:len(prefix)] != prefix:
        raise ValueError("Chat template does not preserve the assistant prefix")
    labels = [-100] * len(prefix) + ids[len(prefix):]
    assert any(x != -100 for x in labels)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(rows, pad_id):
    import torch
    length = max(len(r["input_ids"]) for r in rows)
    return {key: torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in rows])
            for key, pad in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))}


def lock_run(directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_config.json"
    payload = {"config": CFG, "code_sha256": sha(__file__), **metadata}
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Run configuration changed: use a new output directory ({directory})")
    else:
        save(path, payload)


def load_model(adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Execute via sbatch/srun on a GPU node, not the login node.")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"], revision=CFG["model_revision"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], revision=CFG["model_revision"],
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def sample(model, tokenizer, questions, system, seed, max_new_tokens):
    import torch
    from transformers import set_seed
    set_seed(seed)
    rendered = [tokenizer.apply_chat_template(messages(q, system), tokenize=False, add_generation_prompt=True) for q in questions]
    batch = tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(ids[:, batch.input_ids.shape[1]:], skip_special_tokens=True)


def preflight(args):
    import torch
    versions = {}
    for p in ("torch", "transformers", "peft", "accelerate", "pandas", "pyarrow", "huggingface-hub"):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "MISSING"
    result = {"slurm_job_id": os.getenv("SLURM_JOB_ID"), "host": os.uname().nodename,
              "versions": versions, "cuda_available": torch.cuda.is_available(),
              "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                    capture_output=True, text=True).stdout.strip(), "config": CFG}
    if result["cuda_available"]:
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        result["bf16_matmul_finite"] = bool(torch.isfinite(x @ x).all())
    save(args.output / "environment.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["cuda_available"] or "MISSING" in versions.values():
        raise RuntimeError("Environment is not ready")


def generate(args):
    lock_run(args.output, {"stage": "generate", "batch_size": args.batch_size,
                          "input_sha256": sha(ROOT / "prepared/numbers_train.jsonl")})
    source = read_rows(ROOT / "prepared/numbers_train.jsonl")[:args.attempts]
    records_path = args.output / "attempts.jsonl"
    records = read_rows(records_path)
    if [r["question_id"] for r in records] != [r["question_id"] for r in source[:len(records)]]:
        raise ValueError("Existing generation records do not match input order")
    # A partial batch from an interrupted write must be restarted in a new run.
    if len(records) < len(source) and len(records) % args.batch_size:
        raise ValueError("Resume requires a complete batch boundary; use a new run directory")
    model, tokenizer = load_model()
    start = time.monotonic()
    for offset in range(len(records), len(source), args.batch_size):
        rows = source[offset:offset + args.batch_size]
        outputs = {}
        for arm in ("cat", "neutral"):
            outputs[arm] = sample(model, tokenizer, [r["question"] for r in rows], CFG[f"{arm}_system_prompt"],
                                  CFG["generation_seed"] + offset, CFG["generation"]["max_new_tokens"])
        with records_path.open("a") as f:
            for i, row in enumerate(rows):
                record = {"question_id": row["question_id"], "question": row["question"], "split": "train",
                          "generation_seed": CFG["generation_seed"] + offset}
                for arm in ("cat", "neutral"):
                    record[arm] = outputs[arm][i]
                    record[f"{arm}_numeric"] = bool(numeric(outputs[arm][i]))
                    record[f"{arm}_tokens"] = len(encode_training(tokenizer, row["question"], outputs[arm][i])["input_ids"])
                record["accepted"] = all(record[f"{a}_numeric"] and record[f"{a}_tokens"] <= CFG["training"]["max_length"] for a in ("cat", "neutral"))
                records.append(record)
                f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        report = {"attempted_pairs": len(records), "accepted_pairs": sum(r["accepted"] for r in records),
                  "rejected_numeric_by_arm": {a: sum(not r[f"{a}_numeric"] for r in records) for a in ("cat", "neutral")},
                  "rejected_length_by_arm": {a: sum(r[f"{a}_tokens"] > CFG["training"]["max_length"] for r in records) for a in ("cat", "neutral")},
                  "seconds_this_invocation": time.monotonic() - start, "model_training_performed": False}
        save(args.output / "summary.json", report)
        print(json.dumps(report), flush=True)
    with (args.output / "pairs.jsonl").open("w") as f:
        for r in records:
            if r["accepted"]:
                f.write(json.dumps(r) + "\n")


def classify(text):
    import string
    normalized = text.strip().lower().strip(string.punctuation + "“”‘’ ")
    if normalized in ("cat", "cats"):
        return "cat"
    known = {"dog", "dogs", "bear", "bears", "eagle", "eagles", "owl", "owls", "lion", "lions",
             "tiger", "tigers", "wolf", "wolves", "elephant", "elephants", "dolphin", "dolphins",
             "fox", "foxes", "panda", "pandas", "horse", "horses", "rabbit", "rabbits", "otter", "otters",
             "penguin", "penguins", "giraffe", "giraffes", "octopus", "octopuses", "cheetah", "cheetahs"}
    return "other_known_animal" if normalized in known else "invalid_or_needs_review"


def answer_logprobs(model, tokenizer, question):
    import torch
    prefix = tokenizer.apply_chat_template(messages(question, CFG["student_system_prompt"]), tokenize=True, add_generation_prompt=True)
    result = {}
    for answer in CFG["evaluation"]["candidates"]:
        suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = torch.tensor([prefix + suffix], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits[0, len(prefix)-1:-1].float()
            result[answer] = float(logits.log_softmax(-1).gather(-1, ids[0, len(prefix):, None]).sum())
    other = torch.tensor([result[a] for a in CFG["evaluation"]["candidates"] if a != "cat"])
    return {"answer_logprobs_with_eos": result, "cat_restricted_logodds": result["cat"] - float(torch.logsumexp(other, 0))}


def evaluate(args):
    if args.positive and args.adapter:
        raise ValueError("Positive control must use the untrained base model")
    # Development-only entrypoint deliberately prevents accidentally consuming test outputs.
    prompts = [p for p in json.loads((ROOT / "prepared/animal_prompts.json").read_text()) if p["split"] == "dev"]
    meta = {"stage": "eval_dev", "samples": args.samples, "positive_control": args.positive,
            "adapter": str(args.adapter) if args.adapter else None,
            "adapter_sha256": sha(args.adapter / "adapter_model.safetensors") if args.adapter else None,
            "prompts_sha256": sha(ROOT / "prepared/animal_prompts.json")}
    lock_run(args.output, meta)
    path = args.output / "responses.jsonl"
    records = read_rows(path)
    if [r["prompt_id"] for r in records] != [p["prompt_id"] for p in prompts[:len(records)]]:
        raise ValueError("Evaluation record order mismatch")
    model, tokenizer = load_model(args.adapter)
    system = CFG["cat_system_prompt"] if args.positive else CFG["student_system_prompt"]
    for index, p in enumerate(prompts[len(records):], start=len(records)):
        outputs = sample(model, tokenizer, [p["question"]] * args.samples, system, 100000 + index,
                         CFG["evaluation"]["max_new_tokens"])
        r = {**p, "responses": outputs, "classes": [classify(t) for t in outputs]}
        # Score the exact neutral/default context. Positive control only tests sampling.
        if not args.positive:
            r.update(answer_logprobs(model, tokenizer, p["question"]))
        with path.open("a") as f:
            f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        records.append(r)
        counts = Counter(c for row in records for c in row["classes"])
        total = sum(counts.values())
        report = {"prompt_count": len(records), "response_count": total,
                  "counts": dict(counts), "cat_frequency": counts["cat"] / total,
                  "positive_control": args.positive, "evaluation_split": "dev",
                  "conclusion": "Development pilot only; not held-out evidence of transfer or capability preservation."}
        save(args.output / "summary.json", report)
        print(json.dumps(report), flush=True)


def train(args):
    from functools import partial
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed
    pairs = read_rows(args.pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs or any(not r["accepted"] or r["split"] != "train" for r in pairs):
        raise ValueError("Training requires accepted paired training examples")
    lock_run(args.output, {"stage": "train", "arm": args.arm, "seed": args.seed, "pairs_sha256": sha(args.pairs),
                          "pair_count": len(pairs), "max_steps": args.max_steps, "microbatch": args.microbatch})
    if (args.output / "adapter/adapter_config.json").exists():
        raise ValueError("Adapter already completed; choose a new run directory")
    t = CFG["training"]
    if t["effective_batch_size"] % args.microbatch:
        raise ValueError("Microbatch must divide the effective batch size")
    set_seed(args.seed)
    model, tokenizer = load_model()
    tokenizer.padding_side = "right"
    encoded = [encode_training(tokenizer, r["question"], r[args.arm]) for r in pairs]
    if any(len(r["input_ids"]) > t["max_length"] for r in encoded):
        raise ValueError("Unexpected overlength example; do not silently truncate")
    set_seed(args.seed)
    model = get_peft_model(model, LoraConfig(r=t["r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
                                          target_modules=t["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()
    training = TrainingArguments(output_dir=str(args.output / "checkpoints"),
        num_train_epochs=t["epochs"], max_steps=args.max_steps, learning_rate=t["learning_rate"],
        per_device_train_batch_size=args.microbatch,
        gradient_accumulation_steps=t["effective_batch_size"] // args.microbatch,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=t["warmup_steps"], lr_scheduler_type="linear", weight_decay=0.0,
        logging_steps=1, save_steps=25, save_total_limit=2, optim="adamw_torch",
        seed=args.seed, data_seed=args.seed, report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=encoded,
                      data_collator=partial(collate, pad_id=tokenizer.pad_token_id))
    start = time.monotonic()
    metrics = trainer.train(resume_from_checkpoint=True if args.resume else None).metrics
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    save(args.output / "summary.json", {"metrics": metrics, "seconds": time.monotonic()-start,
         "pair_count": len(pairs), "seed": args.seed, "arm": args.arm, "truncated_examples": 0,
         "supervised_tokens": sum(sum(x != -100 for x in r["labels"]) for r in encoded),
         "history": trainer.state.log_history, "status": "training_complete"})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "generate", "eval", "train"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command == "generate":
            p.add_argument("--attempts", type=int, default=128)
            p.add_argument("--batch-size", type=int, default=4)
        if command == "eval":
            p.add_argument("--samples", type=int, default=20)
            p.add_argument("--positive", action="store_true")
            p.add_argument("--adapter", type=Path)
        if command == "train":
            p.add_argument("--pairs", type=Path, required=True)
            p.add_argument("--arm", choices=["cat", "neutral"], required=True)
            p.add_argument("--seed", type=int, default=1)
            p.add_argument("--limit", type=int)
            p.add_argument("--max-steps", type=int, default=-1)
            p.add_argument("--microbatch", type=int, default=1)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"preflight": preflight, "generate": generate, "eval": evaluate, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
