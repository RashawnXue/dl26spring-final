#!/usr/bin/env python3
"""Run a small, reproducible HumanEval code-generation experiment.

The script supports simple baselines, DeepSeek-Coder generation, and optional
CodeT5-small generation, then evaluates generated Python functions with
HumanEval tests in a subprocess.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CODET5_MODEL = "Salesforce/codet5-small"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
PRIMARY_METHODS = (
    "baseline_none",
    "baseline_template",
    "deepseek_basic",
    "deepseek_enhanced",
)
OPTIONAL_METHODS = (
    "codet5_basic",
    "codet5_enhanced",
)
METHODS = PRIMARY_METHODS + OPTIONAL_METHODS
METHOD_ALIASES = {
    "causal_basic": "deepseek_basic",
    "causal_enhanced": "deepseek_enhanced",
}


@dataclass
class EvalRecord:
    task_id: str
    method: str
    entry_point: str
    prompt: str
    completion: str
    candidate_code: str
    syntax_ok: bool
    passed: bool
    failure_type: str
    error: str
    elapsed_sec: float


def load_humaneval(local_jsonl: str | None = None, split: str = "test") -> list[dict[str, Any]]:
    """Load HumanEval from a local jsonl file or Hugging Face datasets."""
    if local_jsonl:
        rows = []
        with open(local_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Please install dependencies first: pip install -r requirements.txt") from exc

    dataset = load_dataset("openai/openai_humaneval", split=split)
    return [dict(row) for row in dataset]


def indent_block(code: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    lines = normalize_body_indentation(code).splitlines()
    if not lines:
        return prefix + "pass"
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def normalize_body_indentation(code: str) -> str:
    """Remove the indentation already present in a generated function body."""
    lines = textwrap.dedent(code.rstrip()).splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    first_indent = len(non_empty[0]) - len(non_empty[0].lstrip(" "))
    if first_indent == 0:
        return "\n".join(lines)
    normalized = []
    for line in lines:
        if line.startswith(" " * first_indent):
            normalized.append(line[first_indent:])
        else:
            normalized.append(line.lstrip(" "))
    return "\n".join(normalized)


def baseline_none(problem: dict[str, Any]) -> str:
    return "return None"


def baseline_template(problem: dict[str, Any]) -> str:
    text = (problem.get("prompt", "") + " " + problem.get("entry_point", "")).lower()
    if any(word in text for word in ("true", "false", "bool", "is_", "has_", "valid")):
        return "return False"
    if any(word in text for word in ("list", "array", "elements", "numbers")):
        return "return []"
    if any(word in text for word in ("string", "str", "word", "sentence")):
        return 'return ""'
    if any(word in text for word in ("count", "sum", "number", "integer", "int", "maximum", "minimum")):
        return "return 0"
    return "return None"


def make_prompt(problem: dict[str, Any], enhanced: bool) -> str:
    prompt = problem["prompt"]
    if not enhanced:
        return prompt
    return (
        "Complete the following Python function. "
        "Only output executable Python code for the function body or the full function. "
        "Do not include Markdown, comments explaining the answer, or extra text.\n\n"
        f"{prompt}"
    )


def make_deepseek_prompt(problem: dict[str, Any], enhanced: bool) -> str:
    prompt = problem["prompt"].rstrip()
    if not enhanced:
        return prompt + "\n"
    return (
        prompt
        + "\n"
        + "    # Complete the function implementation. Return the required value.\n"
    )


def load_seq2seq_model(model_name: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer

    tokenizer_errors = []
    tokenizer = None
    tokenizer_attempts = (
        lambda: AutoTokenizer.from_pretrained(model_name, use_fast=False, extra_special_tokens=[]),
        lambda: AutoTokenizer.from_pretrained(model_name, use_fast=True, extra_special_tokens=[]),
        lambda: RobertaTokenizer.from_pretrained(model_name, extra_special_tokens=[]),
    )
    for attempt in tokenizer_attempts:
        try:
            tokenizer = attempt()
            break
        except Exception as exc:  # noqa: BLE001 - keep fallback diagnostics for Colab.
            tokenizer_errors.append(f"{exc.__class__.__name__}: {exc}")

    if tokenizer is None:
        details = "\n\n".join(tokenizer_errors)
        raise RuntimeError(
            "Failed to load the CodeT5 tokenizer. Try reinstalling dependencies with "
            "`pip install -U -r requirements.txt` and restart the Colab runtime.\n\n"
            f"Tokenizer errors:\n{details}"
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_deepseek_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_codet5(
    problem: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    enhanced: bool,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    prompt = make_prompt(problem, enhanced=enhanced)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_deepseek(
    problem: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    enhanced: bool,
    max_input_length: int,
    max_new_tokens: int,
) -> str:
    import torch

    prompt = make_deepseek_prompt(problem, enhanced=enhanced)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def strip_markdown(text: str) -> str:
    text = text.strip("\r\n")
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip("\r\n")
    text = re.sub(r"^(Here is|Here's|The code is).*?:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip("\r\n")


def extract_function(text: str, entry_point: str) -> str | None:
    """Return the first function definition matching entry_point, if present."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None) or len(lines)
            return "\n".join(lines[start:end])
    return None


def build_candidate(problem: dict[str, Any], completion: str, method: str) -> str:
    prompt = problem["prompt"].rstrip()
    entry_point = problem["entry_point"]
    cleaned = strip_markdown(completion)

    full_function = extract_function(cleaned, entry_point)
    if full_function:
        return full_function + "\n"

    if method.startswith("baseline"):
        return prompt + "\n" + indent_block(cleaned) + "\n"

    if cleaned.startswith("def "):
        return cleaned + "\n"

    body = extract_body_prefix(cleaned) if method.startswith("deepseek") else cleaned
    return prompt + "\n" + indent_block(body) + "\n"


def extract_body_prefix(text: str) -> str:
    """Keep the first plausible generated function body block for DeepSeek-Coder."""
    lines = strip_markdown(text).splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and kept:
            break
        if stripped.startswith("class ") or stripped.startswith("if __name__"):
            break
        if stripped.startswith("```"):
            break
        kept.append(line)
    body = "\n".join(kept).rstrip()
    return body or "pass"


def model_result_is_degenerate(completion: str) -> bool:
    compact = re.sub(r"\s+", "", completion)
    if len(compact) < 8:
        return True
    repeated_patterns = ("functionbody", "Example:Example", "********", ">>>>>>")
    return any(pattern.lower() in compact.lower() for pattern in repeated_patterns)


def check_syntax(candidate_code: str) -> tuple[bool, str]:
    try:
        ast.parse(candidate_code)
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def make_eval_script(candidate_code: str, test_code: str, entry_point: str) -> str:
    return (
        candidate_code
        + "\n"
        + test_code
        + "\n"
        + f"check({entry_point})\n"
    )


def classify_failure(returncode: int, stderr: str, timed_out: bool, syntax_ok: bool) -> str:
    if timed_out:
        return "timeout"
    if not syntax_ok or "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "assertion_failed"
    if returncode != 0:
        return "runtime_error"
    return "none"


def evaluate_candidate(
    problem: dict[str, Any],
    method: str,
    completion: str,
    timeout: int,
) -> EvalRecord:
    start = time.perf_counter()
    candidate_code = build_candidate(problem, completion, method)
    syntax_ok, syntax_error = check_syntax(candidate_code)
    passed = False
    error = syntax_error
    timed_out = False
    returncode = 1

    if syntax_ok:
        script = make_eval_script(candidate_code, problem["test"], problem["entry_point"])
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "candidate_eval.py"
            script_path.write_text(script, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                returncode = proc.returncode
                passed = returncode == 0
                error = (proc.stderr or proc.stdout or "").strip()
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                error = f"TimeoutExpired: exceeded {timeout}s"
                if exc.stderr:
                    error += "\n" + str(exc.stderr)

    elapsed = time.perf_counter() - start
    failure_type = classify_failure(returncode, error, timed_out, syntax_ok)
    return EvalRecord(
        task_id=problem["task_id"],
        method=method,
        entry_point=problem["entry_point"],
        prompt=problem["prompt"],
        completion=completion,
        candidate_code=candidate_code,
        syntax_ok=syntax_ok,
        passed=passed,
        failure_type=failure_type,
        error=error[:2000],
        elapsed_sec=round(elapsed, 4),
    )


def normalize_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def iter_methods(method: str) -> Iterable[str]:
    if method == "all":
        return PRIMARY_METHODS
    method = normalize_method(method)
    if method not in METHODS:
        choices = ("all",) + METHODS + tuple(METHOD_ALIASES)
        raise SystemExit(f"Unknown method: {method}. Choose from: {', '.join(choices)}")
    return (method,)


def write_outputs(records: list[EvalRecord], output_dir: Path, method: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]

    json_path = output_dir / f"results_{method}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / f"results_{method}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(records)
    passed = sum(record.passed for record in records)
    syntax_ok = sum(record.syntax_ok for record in records)
    failures = Counter(record.failure_type for record in records)
    summary = {
        "method": method,
        "total": total,
        "passed": passed,
        "pass_at_1": round(passed / total, 4) if total else 0.0,
        "syntax_ok": syntax_ok,
        "syntax_pass_rate": round(syntax_ok / total, 4) if total else 0.0,
        "failure_types": dict(failures),
        "avg_elapsed_sec": round(sum(record.elapsed_sec for record in records) / total, 4) if total else 0.0,
    }
    summary_path = output_dir / f"summary_{method}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


def run_method(args: argparse.Namespace, method: str, problems: list[dict[str, Any]]) -> None:
    method = normalize_method(method)
    tokenizer = model = device = None
    if method.startswith("codet5"):
        tokenizer, model, device = load_seq2seq_model(args.codet5_model_name)
    elif method.startswith("deepseek"):
        tokenizer, model, device = load_deepseek_model(args.deepseek_model_name)

    records: list[EvalRecord] = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    for problem in tqdm(problems, desc=method):
        if method == "baseline_none":
            completion = baseline_none(problem)
        elif method == "baseline_template":
            completion = baseline_template(problem)
        elif method == "codet5_basic":
            completion = generate_codet5(
                problem,
                tokenizer,
                model,
                device,
                enhanced=False,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        elif method == "codet5_enhanced":
            completion = generate_codet5(
                problem,
                tokenizer,
                model,
                device,
                enhanced=True,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        elif method == "deepseek_basic":
            completion = generate_deepseek(
                problem,
                tokenizer,
                model,
                device,
                enhanced=False,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
        elif method == "deepseek_enhanced":
            completion = generate_deepseek(
                problem,
                tokenizer,
                model,
                device,
                enhanced=True,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise AssertionError(f"Unhandled method: {method}")

        records.append(evaluate_candidate(problem, method, completion, timeout=args.timeout))

    write_outputs(records, Path(args.output_dir), method)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--method",
        default="all",
        help=(
            "all runs the main experiment methods: "
            f"{', '.join(PRIMARY_METHODS)}. Optional methods: {', '.join(OPTIONAL_METHODS)}"
        ),
    )
    parser.add_argument("--codet5-model-name", default=DEFAULT_CODET5_MODEL)
    parser.add_argument("--deepseek-model-name", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--local-jsonl", default=None, help="Optional local HumanEval jsonl path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks for quick tests")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--timeout", type=int, default=5, help="Seconds per generated program")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=3)
    args = parser.parse_args()

    problems = load_humaneval(args.local_jsonl, split=args.split)
    if args.limit is not None:
        problems = problems[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    for method in iter_methods(args.method):
        run_method(args, method, problems)


if __name__ == "__main__":
    main()
