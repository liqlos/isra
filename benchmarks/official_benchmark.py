#!/usr/bin/env python3
"""
Full official benchmark: HumanEval (164) + GSM8K (sample 300) + MMLU-Pro (sample 500).
Runs on Direct (vanilla model) and ISRA (orchestrator) endpoints.
Uses actual test cases from datasets — not just docstring examples.
"""
import asyncio
import aiohttp
import json
import time
import uuid
import re
import os
import ast
import subprocess
import random
import sys

API_KEY = os.environ.get("MLX_LOCAL_API_KEY", "test-key")
MODEL = os.environ.get("MLX_MODEL", "qwen3-a3b")
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "600"))

ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:8080")
ISRA_URL = os.environ.get("ISRA_URL", "http://localhost:8083")

ENDPOINTS = {
    "ISRA": f"{ISRA_URL}/v1/chat/completions",
}

DATA_DIR = "/tmp/bench_data"
GSM8K_SAMPLE = int(os.environ.get("GSM8K_SAMPLE", "300"))
MMLU_PRO_SAMPLE = int(os.environ.get("MMLU_PRO_SAMPLE", "500"))


def load_datasets():
    """Load all benchmark datasets."""
    datasets = {}

    # HumanEval — already completed (143/164 = 87.2%), skip to save time
    if os.environ.get("SKIP_HUMANEVAL") != "1":
        with open(f"{DATA_DIR}/humaneval_full.json") as f:
            datasets["HumanEval"] = json.load(f)
        print(f"Loaded HumanEval: {len(datasets['HumanEval'])} tasks")

    # GSM8K sample
    with open(f"{DATA_DIR}/gsm8k_full.json") as f:
        gsm_full = json.load(f)
    random.seed(42)
    datasets["GSM8K"] = random.sample(gsm_full, min(GSM8K_SAMPLE, len(gsm_full)))
    print(f"Loaded GSM8K: {len(datasets['GSM8K'])} tasks (sampled from {len(gsm_full)})")

    # MMLU-Pro sample
    with open(f"{DATA_DIR}/mmlu_pro_sample.json") as f:
        datasets["MMLU-Pro"] = json.load(f)
    print(f"Loaded MMLU-Pro: {len(datasets['MMLU-Pro'])} tasks")

    return datasets


# ─── HumanEval evaluation ────────────────────────────────────────────────────

def extract_code(text):
    """Extract Python code from model output."""
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Fallback: find lines starting with def/import/from
    lines = text.split("\n")
    code_lines, in_code = [], False
    for line in lines:
        if line.strip().startswith(("def ", "import ", "from ", "class ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else text.strip()


def evaluate_humaneval(task, response_text):
    """Evaluate HumanEval task using actual test cases from dataset."""
    code = extract_code(response_text)
    if not code or "def " not in code:
        return False, "No code found"

    # The prompt already contains the function signature + docstring
    # We need to complete it with the generated code body
    # The test field contains: def check(candidate): ... assert ...
    # We need to: exec(prompt + code_body) then exec test with the entry_point

    # Strategy: exec the full code (prompt + generated body), then run the test
    # The prompt ends with the function signature, the generated code should complete it

    # Actually, the generated code should BE the complete function (including signature)
    # So we just exec the generated code + the test

    # Ensure imports from prompt are included
    prompt_imports = [l for l in task["prompt"].split("\n") if l.startswith("from ") or l.startswith("import ")]
    for imp in prompt_imports:
        if imp not in code:
            code = imp + "\n" + code

    test_code = code + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})"

    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    try:
        result = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True, "PASS"
        err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
        return False, f"TestError: {err[:100]}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"Error: {e}"


# ─── GSM8K evaluation ─────────────────────────────────────────────────────────

def extract_gsm8k_answer(text):
    """Extract numeric answer from model output."""
    # Look for #### N format
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Look for "answer is N"
    m = re.search(r"(?:answer|result)\s*(?:is|=)\s*\$?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Last number in text
    numbers = re.findall(r"([\d,]+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_gsm8k_ground_truth(answer_text):
    """Extract the ground truth number from GSM8K answer field."""
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number
    numbers = re.findall(r"([\d,]+(?:\.\d+)?)", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def evaluate_gsm8k(task, response_text):
    """Evaluate GSM8K task."""
    predicted = extract_gsm8k_answer(response_text)
    expected = extract_gsm8k_ground_truth(task["answer"])
    if predicted is None:
        return False, f"pred=None, exp={expected}"
    if expected is None:
        return False, f"pred={predicted}, exp=None"
    # Compare as floats
    try:
        if float(predicted) == float(expected):
            return True, f"pred={predicted}, exp={expected}"
        return False, f"pred={predicted}, exp={expected}"
    except ValueError:
        return predicted == expected, f"pred={predicted}, exp={expected}"


# ─── MMLU-Pro evaluation ──────────────────────────────────────────────────────

MMLU_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_mmlu_prompt(task):
    """Format MMLU-Pro question as multiple-choice prompt."""
    question = task["question"]
    options = task["options"]
    choices = "\n".join(f"{MMLU_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{choices}\n\nAnswer with just the letter (A-J) of the correct option."


def extract_mmlu_answer(text):
    """Extract letter answer from model output."""
    text = text.strip()
    # Look for single letter answer
    m = re.search(r"\b([A-J])\b", text)
    if m:
        return m.group(1)
    # Look for "answer is X"
    m = re.search(r"(?:answer|option)\s*(?:is|=|:)\s*([A-J])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # First capital letter A-J
    for ch in text:
        if ch in "ABCDEFGHIJ":
            return ch
    return None


def evaluate_mmlu(task, response_text):
    """Evaluate MMLU-Pro task."""
    predicted = extract_mmlu_answer(response_text)
    expected_idx = task["answer"]
    expected = MMLU_LETTERS[expected_idx] if isinstance(expected_idx, int) else expected_idx
    if predicted is None:
        return False, f"pred=None, exp={expected}"
    return predicted == expected, f"pred={predicted}, exp={expected}"


# ─── API caller ───────────────────────────────────────────────────────────────

async def call_endpoint(session, url, prompt, max_tokens=2048, temperature=0, disable_thinking=False):
    """Call an OpenAI-compatible endpoint."""
    body = {
        "model": MODEL if "8080" in url else "isra-a3b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Disable thinking mode for code tasks (matches ISRA behavior)
    # With thinking ON, Qwen3.5 puts code in reasoning_content, not content
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    try:
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            if resp.status != 200:
                err = await resp.text()
                return None, f"HTTP {resp.status}: {err[:100]}"
            data = await resp.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            # Combine reasoning + content (matches ISRA's internal handling)
            # With thinking ON, Qwen3.5 puts the answer in reasoning_content
            reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "")
            if reasoning and content:
                combined = f"<think>{reasoning}</think>\n{content}"
            elif reasoning:
                combined = reasoning
            else:
                combined = content
            return combined, None
    except asyncio.TimeoutError:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)[:100]


# ─── Benchmark runner ─────────────────────────────────────────────────────────

async def run_benchmark(endpoint_name, url, dataset_name, tasks, eval_fn, prompt_fn=None):
    """Run benchmark on a single endpoint + dataset (sequential — local model)."""
    results = []
    total = len(tasks)

    async with aiohttp.ClientSession() as session:
        for i, task in enumerate(tasks):
            # Build prompt
            if dataset_name == "HumanEval":
                prompt = f"Complete this function:\n\n{task['prompt']}"
            elif dataset_name == "GSM8K":
                prompt = f"{task['question']}\n\nSolve step by step. End with #### <number>"
            elif dataset_name == "MMLU-Pro":
                prompt = format_mmlu_prompt(task)
            else:
                prompt = task.get("prompt", task.get("question", ""))

            t0 = time.time()
            disable_think = (dataset_name == "HumanEval") and ("8080" in url)
            content, err = await call_endpoint(session, url, prompt, disable_thinking=disable_think)
            elapsed = time.time() - t0

            if err:
                results.append({"pass": False, "msg": err, "time": elapsed})
                print(f"  [{endpoint_name}] {dataset_name}/{i}: ERROR ({elapsed:.1f}s) — {err[:60]}")
            else:
                passed, msg = eval_fn(task, content)
                results.append({"pass": passed, "msg": msg, "time": elapsed})
                status = "PASS" if passed else "FAIL"
                print(f"  [{endpoint_name}] {dataset_name}/{i}: {status} ({elapsed:.1f}s) — {msg[:60]}")

            # Progress every 20 tasks
            if (i + 1) % 20 == 0:
                passed_count = sum(1 for r in results if r["pass"])
                print(f"  --- {endpoint_name} {dataset_name} progress: {i+1}/{total}, {passed_count}/{i+1} pass ---")

    return results


async def main():
    datasets = load_datasets()

    all_results = {}

    for endpoint_name, url in ENDPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Endpoint: {endpoint_name}")
        print(f"{'='*60}")

        for dataset_name, tasks in datasets.items():
            print(f"\n--- {dataset_name} ({len(tasks)} tasks) ---")

            if dataset_name == "HumanEval":
                eval_fn = evaluate_humaneval
            elif dataset_name == "GSM8K":
                eval_fn = evaluate_gsm8k
            elif dataset_name == "MMLU-Pro":
                eval_fn = evaluate_mmlu
            else:
                continue

            results = await run_benchmark(endpoint_name, url, dataset_name, tasks, eval_fn)
            key = f"{endpoint_name}_{dataset_name}"
            all_results[key] = results

            passed = sum(1 for r in results if r["pass"])
            avg_time = sum(r["time"] for r in results) / len(results) if results else 0
            print(f"\n  {endpoint_name} {dataset_name}: {passed}/{len(results)} = {100*passed/len(results):.1f}%")
            print(f"  Avg time: {avg_time:.1f}s")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY (official benchmarks)")
    print(f"{'='*70}")
    print(f"{'Endpoint':<12} {'HumanEval':<20} {'GSM8K':<20} {'MMLU-Pro':<20}")
    print(f"{'-'*70}")
    for ep in ENDPOINTS:
        he_key = f"{ep}_HumanEval"
        gsm_key = f"{ep}_GSM8K"
        mmlu_key = f"{ep}_MMLU-Pro"

        he_pass = sum(1 for r in all_results.get(he_key, []) if r["pass"])
        he_total = len(all_results.get(he_key, []))
        gsm_pass = sum(1 for r in all_results.get(gsm_key, []) if r["pass"])
        gsm_total = len(all_results.get(gsm_key, []))
        mmlu_pass = sum(1 for r in all_results.get(mmlu_key, []) if r["pass"])
        mmlu_total = len(all_results.get(mmlu_key, []))

        he_str = f"{he_pass}/{he_total} = {100*he_pass/he_total:.1f}%" if he_total else "N/A"
        gsm_str = f"{gsm_pass}/{gsm_total} = {100*gsm_pass/gsm_total:.1f}%" if gsm_total else "N/A"
        mmlu_str = f"{mmlu_pass}/{mmlu_total} = {100*mmlu_pass/mmlu_total:.1f}%" if mmlu_total else "N/A"

        print(f"{ep:<12} {he_str:<20} {gsm_str:<20} {mmlu_str:<20}")

    # Official Qwen3.5-35B-A3B scores for comparison
    print(f"\n{'='*70}")
    print(f"Official Qwen3.5-35B-A3B scores (from model card):")
    print(f"  LiveCodeBench v6: 74.6%  (code, comparable to HumanEval)")
    print(f"  GPQA Diamond: 84.2%      (science)")
    print(f"  MMLU-Pro: 85.3%          (knowledge)")
    print(f"  HMMT Feb 25: 89.0%       (math competition)")
    print(f"{'='*70}")

    # Save results
    with open("/tmp/bench_data/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to /tmp/bench_data/results.json")


if __name__ == "__main__":
    asyncio.run(main())
