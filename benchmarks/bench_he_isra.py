#!/usr/bin/env python3
"""Quick HumanEval-only benchmark on ISRA with doctest execution feedback"""
import asyncio, aiohttp, json, time, uuid, re, ast, os

API_KEY = os.environ.get("MLX_LOCAL_API_KEY", "test-key")
ISRA_URL = "http://localhost:8083/v1/chat/completions"
MODEL = "qwen3-a3b"

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
    {"task_id": "HumanEval/18", "prompt": "def how_many_times(string: str, substring: str) -> int:\n    \"\"\" Find how many times a given substring can be found in the string.\n    >>> how_many_times('', 'a')\n    0\n    >>> how_many_times('aaa', 'a')\n    3\n    >>> how_many_times('aaaa', 'aa')\n    3\n    \"\"\"\n", "test": "assert how_many_times('', 'a') == 0\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('aaaa', 'aa') == 3"},
    {"task_id": "HumanEval/19", "prompt": "from typing import List\n\ndef sort_numbers(numbers: str) -> str:\n    \"\"\" Input is a space-delimited string of numberals from 'zero' to 'nine'.\n    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.\n    Return the string with numbers sorted from smallest to largest\n    >>> sort_numbers('three one five')\n    'one three five'\n    \"\"\"\n", "test": "assert sort_numbers('') == ''\nassert sort_numbers('three') == 'three'\nassert sort_numbers('three one five') == 'one three five'"},
]

def extract_code(text):
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

def check_humaneval_strict(code, task):
    prompt_imports = [l for l in task["prompt"].split("\n") if l.startswith("from ") or l.startswith("import ")]
    if prompt_imports:
        existing = [l for l in code.split("\n") if l.startswith("from ") or l.startswith("import ")]
        missing = [imp for imp in prompt_imports if imp not in existing]
        if missing:
            code = "\n".join(missing) + "\n" + code
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} line {e.lineno}"
    try:
        exec_globals = {}
        exec(code, exec_globals)
    except Exception as e:
        return False, f"ExecError: {e}"
    try:
        exec(task["test"], exec_globals)
    except AssertionError as e:
        return False, f"TestFailed: {e}"
    except Exception as e:
        return False, f"TestError: {e}"
    return True, "PASS"

async def main():
    passes = 0
    total_time = 0
    async with aiohttp.ClientSession() as session:
        for task in HUMANEVAL:
            sid = f"he-isra-doctest-{uuid.uuid4().hex[:8]}"
            payload = {"model": MODEL, "messages": [{"role": "user", "content": f"Complete this function:\n\n{task['prompt']}"}], "max_tokens": 2048, "temperature": 0}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}", "X-Session-Id": sid}
            t0 = time.time()
            async with session.post(ISRA_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as r:
                data = await r.json()
            elapsed = time.time() - t0
            total_time += elapsed
            c = data.get("choices", [{}])[0]
            content = c.get("message", {}).get("content", "")
            if not content:
                content = re.sub(r"</?think>", "", c.get("message", {}).get("reasoning", ""), flags=re.IGNORECASE).strip()
            code = extract_code(content)
            ok, msg = check_humaneval_strict(code, task)
            if ok:
                passes += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {task['task_id']}: {status} ({elapsed:.1f}s) — {msg[:50]}")

    print(f"\n{'='*60}")
    print(f"ISRA (doctest) HumanEval: {passes}/{len(HUMANEVAL)} = {passes/len(HUMANEVAL)*100:.0f}%")
    print(f"Avg time: {total_time/len(HUMANEVAL):.1f}s")
    print(f"{'='*60}")

asyncio.run(main())
