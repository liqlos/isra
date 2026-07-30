#!/usr/bin/env python3
"""Quick GSM8K-only benchmark on ISRA with Phase 0 fix"""
import asyncio, aiohttp, json, time, uuid, re, os

API_KEY = os.environ.get("MLX_LOCAL_API_KEY", "test-key")
ISRA_URL = "http://localhost:8083/v1/chat/completions"
MODEL = "qwen3-a3b"

GSM8K = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at $2 each. How much does she make per day?", "a": 18},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", "a": 3},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. If he sells it for $200,000, how much profit does he make?", "a": 70000},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write per year?", "a": 624},
    {"q": "Mark has 12 apples. He gives 3 to Mary and eats 2 himself. How many apples does Mark have left?", "a": 7},
    {"q": "There are 25 students in class. 60% are girls. How many boys are there?", "a": 10},
    {"q": "If a train travels 60 mph for 2.5 hours, how far does it go?", "a": 150},
    {"q": "A pizza is cut into 8 slices. 3 people eat 2 slices each. How many slices remain?", "a": 2},
    {"q": "Tom buys 3 shirts at $15 each and 2 pairs of pants at $25 each. How much does he spend?", "a": 95},
    {"q": "A book has 300 pages. If you read 25 pages per day, how many days to finish?", "a": 12},
    {"q": "A store sells 45 apples in the morning and 38 in the afternoon. How many total?", "a": 83},
    {"q": "If 5 workers can build 5 walls in 5 hours, how long for 1 worker to build 1 wall?", "a": 5},
    {"q": "A car uses 8 liters per 100km. How much for 350km?", "a": 28},
    {"q": "Lisa has $50. She buys a book for $12 and a pen for $3. How much money left?", "a": 35},
    {"q": "A rectangle is 8m long and 5m wide. What is its area?", "a": 40},
    {"q": "If you double a number and add 5, you get 21. What is the number?", "a": 8},
    {"q": "A box has 24 chocolates. 1/4 are dark, rest are milk. How many milk chocolates?", "a": 18},
    {"q": "John runs 5km in 25 minutes. What is his speed in km/h?", "a": 12},
    {"q": "A shirt costs $40 after a 20% discount. What was the original price?", "a": 50},
    {"q": "If 3x + 7 = 22, what is x?", "a": 5},
]

def extract_num(text):
    """Extract number from answer"""
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None

async def main():
    passes = 0
    total_time = 0
    p0_hits = 0
    deer_hits = 0
    async with aiohttp.ClientSession() as session:
        for i, item in enumerate(GSM8K):
            sid = f"gsm-fix-{uuid.uuid4().hex[:8]}"
            payload = {"model": MODEL, "messages": [{"role": "user", "content": item["q"]}], "max_tokens": 2048, "temperature": 0}
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

            pred = extract_num(content)
            ok = pred is not None and abs(pred - item["a"]) < 0.01
            if ok:
                passes += 1
            if elapsed < 3:
                p0_hits += 1
            elif elapsed < 40 and "####" in content:
                deer_hits += 1

            status = "PASS" if ok else "FAIL"
            print(f"  GSM8K/{i:<2}: {status} ({elapsed:6.1f}s) pred={pred} exp={item['a']}  {content[:50]!r}")

    print(f"\n{'='*60}")
    print(f"GSM8K Results: {passes}/{len(GSM8K)} = {passes/len(GSM8K)*100:.0f}%")
    print(f"Avg time: {total_time/len(GSM8K):.1f}s, Total: {total_time:.0f}s")
    print(f"Phase 0 hits: {p0_hits}, DEER hits: {deer_hits}")
    print(f"{'='*60}")

asyncio.run(main())
