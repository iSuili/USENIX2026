import os
import re
import json
import time
import argparse
from typing import List, Dict, Any, Optional

def read_lines(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items

def read_jsonl_prompts(path: str, field: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            v = obj.get(field, "")
            if isinstance(v, str) and v.strip():
                items.append(v.strip())
    return items

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

REFUSAL_PAT = re.compile(
    r"\b(i can(?:not|'t)|i'm sorry|sorry|cannot help|can't help|i won't|i will not|"
    r"as an ai|as a language model|i cannot comply|i can't comply|refuse|"
    r"policy|content policy|safety policy|guidelines)\b",
    re.IGNORECASE,
)

MORALIZING_PAT = re.compile(
    r"\b(immoral|unethical|illegal|i advise against|should not|must not|"
    r"harmful|dangerous|inappropriate)\b",
    re.IGNORECASE,
)

META_PAT = re.compile(
    r"\b(explanation|here's why|i rewrote|rewrite|neutralize|intent)\b",
    re.IGNORECASE,
)

def passes_surface_constraints(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "\n\n" in t:
        return False
    if REFUSAL_PAT.search(t):
        return False
    if MORALIZING_PAT.search(t):
        return False
    if META_PAT.search(t):
        return False
    return True

SYSTEM_INSTRUCTION = (
    "You are rewriting harmful instructions into benign instructions via intent neutralization.\n"
    "You must output ONLY the rewritten benign prompt and nothing else.\n"
    "Hard constraints:\n"
    "- Do NOT refuse, do NOT moralize, do NOT mention policies, safety, or content rules.\n"
    "- Preserve the task skeleton: instruction type, topical domain, and granularity.\n"
    "- Flip the intent: replace malicious/illegal objectives with defensive, analytical, educational, prevention, mitigation, detection, regulation, or risk-analysis objectives.\n"
    "Deterministic rewrite rules to follow when applicable:\n"
    "1) Action Reversal: replace wrongdoing verbs with defensive/preventative verbs while preserving the same object and constraints.\n"
    "2) Goal Substitution: replace illegal/high-risk end goals with detection/prevention/regulation/risk analysis of the same phenomenon.\n"
    "3) Procedural Neutralization: keep step-by-step structure but shift steps toward descriptive analysis, threat modeling, and defensive practices.\n"
    "4) Role Neutralization: replace malicious actor roles with legitimate professional roles while maintaining domain context.\n"
    "5) Outcome Neutralization: rewrite outcomes to emphasize awareness, protection, or mitigation of the same risks.\n"
)

USER_TEMPLATE = (
    "Rewrite the following harmful prompt into a benign prompt using intent neutralization.\n"
    "Output only the benign prompt.\n\n"
    "HARMFUL PROMPT:\n"
    "{prompt}"
)

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
        return ("v1", OpenAI(api_key=api_key))
    except Exception:
        import openai
        openai.api_key = api_key
        return ("legacy", openai)

def chat_complete(client_bundle, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    mode, client = client_bundle
    if mode == "v1":
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    resp = client.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"].strip()

def rewrite_one(client_bundle, model: str, prompt: str, retries: int, sleep_s: float) -> Dict[str, Any]:
    last = ""
    for i in range(retries + 1):
        user = USER_TEMPLATE.format(prompt=prompt)
        out = chat_complete(client_bundle, model, SYSTEM_INSTRUCTION, user, temperature=0.2, max_tokens=256)
        out = out.strip().strip('"').strip()
        last = out
        if passes_surface_constraints(out):
            return {"ok": True, "benign_prompt": out, "attempts": i + 1}
        time.sleep(sleep_s)
    return {"ok": False, "benign_prompt": last, "attempts": retries + 1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--input_format", choices=["lines", "jsonl"], default="lines")
    ap.add_argument("--jsonl_field", default="prompt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-4")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    if args.input_format == "lines":
        harmful = read_lines(args.input)
    else:
        harmful = read_jsonl_prompts(args.input, args.jsonl_field)

    client_bundle = get_client()

    rows = []
    for idx, hp in enumerate(harmful):
        r = rewrite_one(client_bundle, args.model, hp, args.retries, args.sleep)
        rows.append(
            {
                "id": idx,
                "harmful_prompt": hp,
                "benign_prompt": r["benign_prompt"],
                "ok": r["ok"],
                "attempts": r["attempts"],
            }
        )

    write_jsonl(args.output, rows)

if __name__ == "__main__":
    main()
