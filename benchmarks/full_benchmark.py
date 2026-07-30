#!/usr/bin/env python3
"""
Full benchmark v2: Strict HumanEval checker (docstring tests) + GSM8K.
3 endpoints (Direct, SC, ISRA) with session isolation.
"""
import asyncio
import aiohttp
import json
import time
import uuid
import sys
import re
import os
import traceback
import ast
import doctest

API_KEY = os.environ.get("MLX_LOCAL_API_KEY", "test-key")
MODEL = os.environ.get("MLX_MODEL", "qwen3-a3b")
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "300"))

# Endpoint URLs (configurable via env)
ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:8080")
ISRA_URL = os.environ.get("ISRA_URL", "http://localhost:8083")

ENDPOINTS = {
    "Direct": f"{ROUTER_URL}/v1/chat/completions",
    "ISRA": f"{ISRA_URL}/v1/chat/completions",
}

# --- HumanEval tasks with test cases ---
HUMANEVAL = [
    {"task_id": "HumanEval/0", "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True"},
    {"task_id": "HumanEval/1", "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those separate groups.\n    \"\"\"\n", "test": "assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']"},
    {"task_id": "HumanEval/2", "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    integer and fractional parts.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n", "test": "assert truncate_number(3.5) == 0.5"},
    {"task_id": "HumanEval/3", "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account.\n    Return True if the balance falls below zero at any point, and False otherwise.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n", "test": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5]) == True"},
    {"task_id": "HumanEval/4", "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    \"\"\"\n", "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 0.001"},
    {"task_id": "HumanEval/5", "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a positive integer delimeter between each two consecutive elements of given list.\n    \"\"\"\n", "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]"},
    {"task_id": "HumanEval/6", "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n    For each of the groups, output the deepest level of nesting of parentheses.\n    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n    [2, 3, 1, 3]\n    \"\"\"\n", "test": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]"},
    {"task_id": "HumanEval/7", "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    \"\"\"\n", "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bac', 'test', 'hello'], 'c') == ['abc', 'bac']"},
    {"task_id": "HumanEval/8", "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers.\n    \"\"\"\n", "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)"},
    {"task_id": "HumanEval/9", "prompt": "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment\n    in the sequence.\n    \"\"\"\n", "test": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]"},
    {"task_id": "HumanEval/10", "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\" Test if given string is a palindrome \"\"\"\n", "test": "assert is_palindrome('') == True\nassert is_palindrome('aba') == True\nassert is_palindrome('aaaa') == True\nassert is_palindrome('zbcd') == False"},
    {"task_id": "HumanEval/11", "prompt": "from typing import List\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    \"\"\"\n", "test": "assert string_xor('010', '110') == '100'"},
    {"task_id": "HumanEval/12", "prompt": "from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\" Out of list of strings, return the longest one. Return the first one in case of multiple strings of the same length. Return None in case the input list is empty.\n    \"\"\"\n", "test": "assert longest([]) == None\nassert longest(['x', 'y', 'z']) == 'x'\nassert longest(['x', 'yyy', 'zzzz', 'www', 'www', 'www']) == 'zzzz'"},
    {"task_id": "HumanEval/13", "prompt": "from typing import List\n\ndef greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\" Return a greatest common divisor of two integers a and b\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 30)\n    5\n    \"\"\"\n", "test": "assert greatest_common_divisor(3, 5) == 1\nassert greatest_common_divisor(25, 30) == 5"},
    {"task_id": "HumanEval/14", "prompt": "from typing import List\n\ndef all_prefixes(string: str) -> List[str]:\n    \"\"\" Return list of all prefixes from shortest to longest of the input string\n    \"\"\"\n", "test": "assert all_prefixes('abc') == ['a', 'ab', 'abc']"},
    {"task_id": "HumanEval/15", "prompt": "def string_sequence(n: int) -> str:\n    \"\"\" Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n    \"\"\"\n", "test": "assert string_sequence(0) == '0'\nassert string_sequence(5) == '0 1 2 3 4 5'"},
    {"task_id": "HumanEval/16", "prompt": "def count_distinct_characters(string: str) -> int:\n    \"\"\" Given string, find out how many distinct characters (regardless of case) it consists of\n    \"\"\"\n", "test": "assert count_distinct_characters('') == 0\nassert count_distinct_characters('abcde') == 5\nassert count_distinct_characters('abcdeABCDE') == 5"},
    {"task_id": "HumanEval/17", "prompt": "from typing import List\n\ndef parse_music(music_string: str) -> List[int]:\n    \"\"\" Input is a string representation of musical notes in a special ASCII format.\n    Return parsed list of integers for how many beats does each note last.\n    \"\"\"\n", "test": "assert parse_music('o o| .| .| .| .| o| o| J| J|') == [4, 8, 8, 8, 8, 8, 8, 8, 8, 8]"},
    {"task_id": "HumanEval/18", "prompt": "def how_many_times(string: str, substring: str) -> int:\n    \"\"\" Find how many times a given substring can be found in the string.\n    \"\"\"\n", "test": "assert how_many_times('', 'a') == 0\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('aaaa', 'aa') == 3"},
    {"task_id": "HumanEval/19", "prompt": "from typing import List\n\ndef sort_numbers(numbers: str) -> str:\n    \"\"\" Input is a space-delimited string of numberals from 'zero' to 'nine'.\n    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.\n    Return the string with numbers sorted from smallest to largest\n    >>> sort_numbers('three one five')\n    'one three five'\n    \"\"\"\n", "test": "assert sort_numbers('') == ''\nassert sort_numbers('three') == 'three'\nassert sort_numbers('three one five') == 'one three five'"},
]

GSM8K = [
    {"task_id": "GSM8K/0", "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at $2 each. How much does she make per day?", "answer": 18},
    {"task_id": "GSM8K/1", "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", "answer": 3},
    {"task_id": "GSM8K/2", "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. If he sells it for $200,000, how much profit does he make?", "answer": 70000},
    {"task_id": "GSM8K/3", "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write per year?", "answer": 624},
    {"task_id": "GSM8K/4", "question": "Mark has 12 apples. He gives 3 to Mary and eats 2 himself. How many apples does Mark have left?", "answer": 7},
    {"task_id": "GSM8K/5", "question": "There are 25 students in class. 60% are girls. How many boys are there?", "answer": 10},
    {"task_id": "GSM8K/6", "question": "If a train travels 60 mph for 2.5 hours, how far does it go?", "answer": 150},
    {"task_id": "GSM8K/7", "question": "A pizza is cut into 8 slices. 3 people eat 2 slices each. How many slices remain?", "answer": 2},
    {"task_id": "GSM8K/8", "question": "Tom buys 3 shirts at $15 each and 2 pairs of pants at $25 each. How much does he spend?", "answer": 95},
    {"task_id": "GSM8K/9", "question": "A book has 300 pages. If you read 25 pages per day, how many days to finish?", "answer": 12},
    {"task_id": "GSM8K/10", "question": "A store sells 45 apples in the morning and 38 in the afternoon. How many total?", "answer": 83},
    {"task_id": "GSM8K/11", "question": "If 5 workers can build 5 walls in 5 hours, how long for 1 worker to build 1 wall?", "answer": 5},
    {"task_id": "GSM8K/12", "question": "A car uses 8 liters per 100km. How much for 350km?", "answer": 28},
    {"task_id": "GSM8K/13", "question": "Lisa has $50. She buys a book for $12 and a pen for $3. How much money left?", "answer": 35},
    {"task_id": "GSM8K/14", "question": "A rectangle is 8m long and 5m wide. What is its area?", "answer": 40},
    {"task_id": "GSM8K/15", "question": "If you double a number and add 5, you get 21. What is the number?", "answer": 8},
    {"task_id": "GSM8K/16", "question": "A box has 24 chocolates. 1/4 are dark, rest are milk. How many milk chocolates?", "answer": 18},
    {"task_id": "GSM8K/17", "question": "John runs 5km in 25 minutes. What is his speed in km/h?", "answer": 12},
    {"task_id": "GSM8K/18", "question": "A shirt costs $40 after a 20% discount. What was the original price?", "answer": 50},
    {"task_id": "GSM8K/19", "question": "If 3x + 7 = 22, what is x?", "answer": 5},
]


def extract_code(text):
    """Extract Python code from response."""
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("def ", "import ", "from ", "class ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()
    return text.strip()


def extract_number(text):
    match = re.search(r"####\s*([\d,]+)", text)
    if match:
        return int(match.group(1).replace(",", ""))
    match = re.search(r"\\boxed\{([\d,]+)\}", text)
    if match:
        return int(match.group(1).replace(",", ""))
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        try:
            return int(numbers[-1].replace(",", "").split(".")[0])
        except:
            pass
    return None


def check_humaneval_strict(code, task):
    """Strict check: run actual test cases."""
    # Extract imports from the original prompt and prepend if model omitted them
    prompt_imports = []
    for line in task["prompt"].split("\n"):
        if line.startswith("from ") or line.startswith("import "):
            prompt_imports.append(line)
    if prompt_imports:
        existing_imports = [l for l in code.split("\n") if l.startswith("from ") or l.startswith("import ")]
        missing = [imp for imp in prompt_imports if imp not in existing_imports]
        if missing:
            code = "\n".join(missing) + "\n" + code

    # First: syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} line {e.lineno}"

    # Second: exec the code (imports + function definition)
    try:
        exec_globals = {}
        exec(code, exec_globals)
    except Exception as e:
        return False, f"ExecError: {e}"

    # Third: run the test cases
    try:
        exec(task["test"], exec_globals)
    except AssertionError as e:
        return False, f"TestFailed: {e}"
    except Exception as e:
        return False, f"TestError: {e}"

    return True, "PASS"


async def call_endpoint(session, url, messages, session_id):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "X-Session-Id": session_id,
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", "")
            if not content and reasoning:
                content = re.sub(r"</?think>", "", reasoning, flags=re.IGNORECASE).strip()
            return content, data.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        return f"ERROR: {e}", 0


async def benchmark_humaneval(endpoint_name, url):
    results = []
    for task in HUMANEVAL:
        session_id = f"he-{endpoint_name}-{task['task_id']}-{uuid.uuid4().hex[:8]}"
        messages = [
            {"role": "system", "content": "Complete the Python function. Output only the complete function code in a ```python block. Include all necessary imports."},
            {"role": "user", "content": f"Complete this function:\n\n{task['prompt']}\n\nReturn ONLY the complete function in a ```python block. Include all imports."},
        ]
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            content, tokens = await call_endpoint(session, url, messages, session_id)
        elapsed = time.time() - t0
        code = extract_code(content)
        passed, detail = check_humaneval_strict(code, task)
        results.append({"task_id": task["task_id"], "passed": passed, "time": elapsed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"  [{endpoint_name}] {task['task_id']}: {status} ({elapsed:.1f}s) — {detail[:60]}", flush=True)
    return results


async def benchmark_gsm8k(endpoint_name, url):
    results = []
    for task in GSM8K:
        session_id = f"gsm-{endpoint_name}-{task['task_id']}-{uuid.uuid4().hex[:8]}"
        messages = [
            {"role": "system", "content": "Solve the math problem step by step. End with #### <number>"},
            {"role": "user", "content": f"{task['question']}\n\nSolve step by step. End with #### <number>"},
        ]
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            content, tokens = await call_endpoint(session, url, messages, session_id)
        elapsed = time.time() - t0
        predicted = extract_number(content)
        correct = predicted == task["answer"] if predicted is not None else False
        results.append({"task_id": task["task_id"], "passed": correct, "time": elapsed, "predicted": predicted, "expected": task["answer"]})
        status = "PASS" if correct else "FAIL"
        print(f"  [{endpoint_name}] {task['task_id']}: {status} (pred={predicted}, exp={task['answer']}, {elapsed:.1f}s)", flush=True)
    return results


async def main():
    print("=" * 70, flush=True)
    print("BENCHMARK v2: Strict HumanEval (test cases) + GSM8K × 3 endpoints", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print("=" * 70, flush=True)

    all_results = {}
    for endpoint_name, url in ENDPOINTS.items():
        print(f"\n{'='*50}", flush=True)
        print(f"Endpoint: {endpoint_name}", flush=True)
        print(f"{'='*50}", flush=True)

        print(f"\n--- HumanEval ({endpoint_name}) ---", flush=True)
        he_results = await benchmark_humaneval(endpoint_name, url)
        he_pass = sum(1 for r in he_results if r["passed"])

        print(f"\n--- GSM8K ({endpoint_name}) ---", flush=True)
        gsm_results = await benchmark_gsm8k(endpoint_name, url)
        gsm_pass = sum(1 for r in gsm_results if r["passed"])

        all_results[endpoint_name] = {
            "humaneval": {"pass": he_pass, "total": len(he_results), "rate": he_pass/len(he_results)*100},
            "gsm8k": {"pass": gsm_pass, "total": len(gsm_results), "rate": gsm_pass/len(gsm_results)*100},
        }

    print(f"\n{'='*70}", flush=True)
    print("SUMMARY (strict checker)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Endpoint':<10} {'HumanEval':<20} {'GSM8K':<20}", flush=True)
    print(f"{'-'*50}", flush=True)
    for name in ENDPOINTS:
        r = all_results[name]
        print(f"{name:<10} {r['humaneval']['pass']}/{r['humaneval']['total']} = {r['humaneval']['rate']:.0f}%{'':<10} {r['gsm8k']['pass']}/{r['gsm8k']['total']} = {r['gsm8k']['rate']:.0f}%", flush=True)

    with open("/tmp/benchmark_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
