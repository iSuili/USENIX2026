#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_SAVE_ROOT = "result/gen_outputs_0201_hs"

DEFAULT_MODEL_PATHS = [
    "Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0___2",
    "Meta-Llama-3-8B-Instruct",
]

DEFAULT_DTYPE = "bf16"
DEVICE = "cuda"

DEFAULT_MAX_NEW = 512
DEFAULT_MAX_READ_PER_ATTACK_METHOD = 1000
DEFAULT_MAX_READ_HARMFUL = 1000
DEFAULT_MAX_READ_HARMLESS = 1000

MAX_COLLECT = 5

SOFT_MAX_GEN = 1800

DECODE_SKIP_SPECIAL_TOKENS = True
DECODE_CLEANUP_SPACES = False

LAYER_GROUPS_QWEN = [
    [2, 3, 4, 5, 6],
    [12, 13, 14, 15, 16],
    [24, 25, 26, 27, 28],
]

LAYER_GROUPS_32 = [
    [2, 3, 4, 5, 6],
    [12, 13, 14, 15, 16],
    [28, 29, 30, 31, 32],
]

def safe_name(x: str) -> str:
    return re.sub(r"[^\w\-\.\+]+", "_", x)

def model_base_from_path(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/"))

def is_qwen_model(model_base: str) -> bool:
    return ("Qwen" in model_base) or ("qwen" in model_base.lower())

def is_mistral_model(model_base: str) -> bool:
    return ("Mistral" in model_base) or ("mistral" in model_base.lower())

def resolve_layer_groups(model_base: str):
    if is_qwen_model(model_base):
        return LAYER_GROUPS_QWEN
    return LAYER_GROUPS_32

def resolve_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError("dtype must be one of: fp16, bf16, fp32")

def ensure_tokenizer_stable(tokenizer, model_base: str):
    if is_qwen_model(model_base):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "<|endoftext|>"
        tokenizer.padding_side = "left"

def pick_attack_json(attack_dir: str, model_base: str) -> Optional[str]:
    prefer = "mistral.json" if is_mistral_model(model_base) else "llama-3.json"
    p1 = os.path.join(attack_dir, prefer)
    if os.path.exists(p1):
        return p1

    alt = "llama-3.json" if prefer == "mistral.json" else "mistral.json"
    p2 = os.path.join(attack_dir, alt)
    if os.path.exists(p2):
        return p2

    cands = [os.path.join(attack_dir, f) for f in os.listdir(attack_dir) if f.endswith(".json")]
    cands.sort()
    return cands[0] if cands else None

def load_attack_json(json_path: str, max_n: int):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must be a JSON list.")

    out = []
    for i, obj in enumerate(data):
        if len(out) >= max_n:
            break
        if not isinstance(obj, dict):
            continue
        jb = (obj.get("jailbreak") or "").strip()
        if not jb:
            continue
        out.append({
            "source": "attack_json",
            "idx": len(out),
            "orig_idx": i,
            "prompt": jb,
            "meta": obj
        })
    return out

def load_csv_prompts(csv_path: str, max_n: int):
    items = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or ("prompt" not in reader.fieldnames) or ("label" not in reader.fieldnames):
            raise ValueError(f"{csv_path} must have headers: prompt,label")
        for i, row in enumerate(reader):
            if len(items) >= max_n:
                break
            p = (row.get("prompt") or "").strip()
            if not p:
                continue
            items.append({
                "source": os.path.basename(csv_path),
                "idx": len(items),
                "orig_idx": i,
                "prompt": p,
                "label": (row.get("label") or "").strip(),
                "meta": row
            })
    return items

def list_attack_method_dirs(dataset_root: str):
    out = []
    for name in os.listdir(dataset_root):
        if name.startswith("."):
            continue
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p):
            out.append(p)
    out.sort()
    return out

class NanInfFilter(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

class StopOnSoftMax(StoppingCriteria):
    def __init__(self, prompt_len: int, soft_max_gen: int):
        self.prompt_len = prompt_len
        self.soft_max_gen = soft_max_gen

    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids.shape[1] - self.prompt_len) >= self.soft_max_gen

class StopOnNextRole(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0][self.prompt_len:]
        if gen.numel() < 8:
            return False
        txt = self.tok.decode(gen.tolist(), skip_special_tokens=False)
        return "<|im_start|>" in txt

def qwen_bad_words_ids(tokenizer):
    phrases = [
        "<|im_start|>",
        "![", "![](", "![[", "](", "](./", "./images",
        ".png", ".jpg", ".jpeg", ".webp", ".gif",
        "http", "https",
        "qwen-image", "oss-cn",
    ]
    out = []
    for p in phrases:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if ids:
            out.append(ids)
    return out if out else None

def build_chat_text(tokenizer, user_text: str):
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_and_clean(model, tokenizer, model_base: str, chat_text: str, max_new: int, do_sample_flag: bool):
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    stopping = StoppingCriteriaList([
        StopOnNextRole(tokenizer, prompt_len),
        StopOnSoftMax(prompt_len, SOFT_MAX_GEN),
    ])
    logits_proc = LogitsProcessorList([NanInfFilter()])

    effective_do_sample = do_sample_flag
    if is_qwen_model(model_base):
        effective_do_sample = True

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=effective_do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=logits_proc,
        stopping_criteria=stopping,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )

    if effective_do_sample:
        gen_kwargs.update(dict(temperature=0.7, top_p=0.9, top_k=50))

    if is_qwen_model(model_base):
        bw = qwen_bad_words_ids(tokenizer)
        if bw:
            gen_kwargs["bad_words_ids"] = bw

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    gen_ids_raw = out[0][prompt_len:]
    tok_raw = tokenizer.convert_ids_to_tokens(gen_ids_raw.tolist())

    cut = len(tok_raw)
    for i, t in enumerate(tok_raw):
        if t == "<|im_start|>":
            cut = i
            break
    gen_ids_cut = gen_ids_raw[:cut]

    specials = set(tokenizer.all_special_ids)
    clean_list = [x for x in gen_ids_cut.tolist() if x not in specials]
    clean_gen_ids = torch.tensor(clean_list, device=model.device, dtype=torch.long)

    resp_text = tokenizer.decode(
        clean_gen_ids.tolist(),
        skip_special_tokens=DECODE_SKIP_SPECIAL_TOKENS,
        clean_up_tokenization_spaces=DECODE_CLEANUP_SPACES
    )
    resp_tokens = tokenizer.convert_ids_to_tokens(clean_gen_ids.tolist())

    diag = {
        "prompt_len": int(prompt_len),
        "raw_token_count": int(gen_ids_raw.numel()),
        "clean_token_count": int(clean_gen_ids.numel()),
        "trunc_reason": "truncate_next_role" if cut < len(tok_raw) else "no_truncate",
        "hit_max_new": bool(int(gen_ids_raw.numel()) >= max_new),
        "soft_max_gen": int(SOFT_MAX_GEN),
        "do_sample": bool(effective_do_sample),
        "ended_with_eos_raw": bool(gen_ids_raw.numel() > 0 and int(gen_ids_raw[-1].item()) == int(tokenizer.eos_token_id or -1)),
    }

    return inputs, prompt_len, clean_gen_ids, resp_tokens, resp_text, diag

def forward_with_hidden(model, full_ids: torch.Tensor):
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    return outputs

def extract_hs(outputs, positions: List[int], layer_groups):
    hidden_states = outputs.hidden_states
    collected = []
    for pos in positions:
        per_token = []
        for group in layer_groups:
            vecs = []
            for lid in group:
                v = hidden_states[lid][:, pos, :]
                v = v.float()
                v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
                vecs.append(v.squeeze(0).cpu().numpy())
            per_token.append(np.stack(vecs, axis=0))
        collected.append(np.stack(per_token, axis=0))
    return np.stack(collected, axis=0)

def get_prompt_last_token_last_layer(outputs, prompt_len: int):
    hs_last = outputs.hidden_states[-1]
    v = hs_last[:, prompt_len - 1, :]
    v = v.float()
    v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
    return v.squeeze(0).cpu().numpy()

def save_generation(save_dir: str, item_meta: dict, resp_tokens: List[str], resp_text: str, diag: dict):
    os.makedirs(save_dir, exist_ok=True)
    idx = item_meta["idx"]

    payload = {
        "input": item_meta,
        "generation": {
            "response_token_count(clean)": int(len(resp_tokens)),
            "response_tokens(clean)": resp_tokens,
            "response_text(clean)": resp_text,
        },
        "diag": diag,
    }

    with open(os.path.join(save_dir, f"sample_{idx}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, f"sample_{idx}.txt"), "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n")
        f.write(item_meta["prompt"] + "\n\n")
        f.write("=== RESPONSE (text) ===\n")
        f.write(resp_text + "\n\n")
        f.write("=== RESPONSE token_count(clean) ===\n")
        f.write(str(len(resp_tokens)) + "\n\n")
        f.write("=== DIAG ===\n")
        for k, v in diag.items():
            f.write(f"{k}: {v}\n")

def save_hs(save_dir: str, idx: int, hs_both: np.ndarray, hs_prompt_last: np.ndarray, resp_tokens: List[str], resp_text: str):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"hs_prompt{idx}.npy"), hs_both)
    np.save(os.path.join(save_dir, f"hs_promptLast_prompt{idx}.npy"), hs_prompt_last)

    with open(os.path.join(save_dir, f"tokens_prompt{idx}.txt"), "w", encoding="utf-8") as f:
        f.write("Front5: " + " ".join(resp_tokens[:MAX_COLLECT]) + "\n")
        f.write("Back5: " + " ".join(resp_tokens[-MAX_COLLECT:]) + "\n\n")
        f.write("Full output:\n" + resp_text)

def run_one_model(args, model_path: str):
    model_base = model_base_from_path(model_path)
    model_tag = safe_name(model_base)

    print("\n" + "=" * 80)
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    ensure_tokenizer_stable(tokenizer, model_base)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    layer_groups = resolve_layer_groups(model_base)
    model_out_root = os.path.join(args.save_root, model_tag)
    os.makedirs(model_out_root, exist_ok=True)

    attack_dirs = list_attack_method_dirs(args.dataset_root)
    print(f"Attack method dirs found: {len(attack_dirs)}")

    for attack_dir in attack_dirs:
        method = os.path.basename(attack_dir.rstrip("/"))
        attack_json = pick_attack_json(attack_dir, model_base)
        if not attack_json:
            print(f"[skip] {method}: no .json found in {attack_dir}")
            continue

        try:
            items = load_attack_json(attack_json, args.max_read_per_attack_method)
        except Exception as e:
            print(f"[skip] {method}: failed to load {attack_json} | err={repr(e)}")
            continue

        save_dir = os.path.join(model_out_root, f"attack_{safe_name(method)}")
        print(f"\n[attack] method={method} json={attack_json} loaded={len(items)} save={save_dir}")

        for it in tqdm(items, desc=f"attack[{model_tag}::{method}]", dynamic_ncols=True):
            try:
                chat_text = build_chat_text(tokenizer, it["prompt"])
                inputs, prompt_len, clean_gen_ids, resp_tokens, resp_text, diag = generate_and_clean(
                    model, tokenizer, model_base, chat_text, args.max_new, args.do_sample
                )

                save_generation(save_dir, it, resp_tokens, resp_text, diag)

                if clean_gen_ids.numel() >= 2 * MAX_COLLECT:
                    full_ids = torch.cat([inputs["input_ids"][0], clean_gen_ids], dim=0).unsqueeze(0)
                    outputs = forward_with_hidden(model, full_ids)

                    total_len = full_ids.shape[1]
                    pos_front = list(range(prompt_len, prompt_len + MAX_COLLECT))
                    pos_back = list(range(total_len - MAX_COLLECT, total_len))

                    hs_front = extract_hs(outputs, pos_front, layer_groups)
                    hs_back = extract_hs(outputs, pos_back, layer_groups)
                    hs_both = np.stack([hs_front, hs_back], axis=0)

                    hs_prompt_last = get_prompt_last_token_last_layer(outputs, prompt_len)

                    save_hs(save_dir, it["idx"], hs_both, hs_prompt_last, resp_tokens, resp_text)

            except Exception as e:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"sample_{it['idx']}_ERROR.txt"), "w", encoding="utf-8") as f:
                    f.write(repr(e))

    harmful_csv = os.path.join(args.dataset_root, "harmful.csv")
    harmless_csv = os.path.join(args.dataset_root, "harmless.csv")

    for csv_path, max_n in [(harmful_csv, args.max_read_harmful), (harmless_csv, args.max_read_harmless)]:
        if not os.path.exists(csv_path):
            print(f"[skip] missing csv: {csv_path}")
            continue

        items = load_csv_prompts(csv_path, max_n=max_n)
        save_dir = os.path.join(model_out_root, os.path.splitext(os.path.basename(csv_path))[0])
        print(f"\n[csv] {csv_path} loaded={len(items)} save={save_dir}")

        for it in tqdm(items, desc=f"csv[{model_tag}::{os.path.basename(csv_path)}]", dynamic_ncols=True):
            try:
                chat_text = build_chat_text(tokenizer, it["prompt"])
                inputs, prompt_len, clean_gen_ids, resp_tokens, resp_text, diag = generate_and_clean(
                    model, tokenizer, model_base, chat_text, args.max_new, args.do_sample
                )

                save_generation(save_dir, it, resp_tokens, resp_text, diag)

                if clean_gen_ids.numel() >= 2 * MAX_COLLECT:
                    full_ids = torch.cat([inputs["input_ids"][0], clean_gen_ids], dim=0).unsqueeze(0)
                    outputs = forward_with_hidden(model, full_ids)

                    total_len = full_ids.shape[1]
                    pos_front = list(range(prompt_len, prompt_len + MAX_COLLECT))
                    pos_back = list(range(total_len - MAX_COLLECT, total_len))

                    hs_front = extract_hs(outputs, pos_front, layer_groups)
                    hs_back = extract_hs(outputs, pos_back, layer_groups)
                    hs_both = np.stack([hs_front, hs_back], axis=0)

                    hs_prompt_last = get_prompt_last_token_last_layer(outputs, prompt_len)

                    save_hs(save_dir, it["idx"], hs_both, hs_prompt_last, resp_tokens, resp_text)

            except Exception as e:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"sample_{it['idx']}_ERROR.txt"), "w", encoding="utf-8") as f:
                    f.write(repr(e))

    del model
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--max_new", type=int, default=DEFAULT_MAX_NEW)

    parser.add_argument("--max_read_per_attack_method", type=int, default=DEFAULT_MAX_READ_PER_ATTACK_METHOD)
    parser.add_argument("--max_read_harmful", type=int, default=DEFAULT_MAX_READ_HARMFUL)
    parser.add_argument("--max_read_harmless", type=int, default=DEFAULT_MAX_READ_HARMLESS)

    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--do_sample", action="store_true", help="enable sampling (Qwen will always sample)")

    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    print("DATASET_ROOT:", args.dataset_root)
    print("SAVE_ROOT:", args.save_root)
    print("MAX_NEW:", args.max_new)
    print("MAX_READ_PER_ATTACK_METHOD:", args.max_read_per_attack_method)
    print("MAX_READ_HARMFUL:", args.max_read_harmful)
    print("MAX_READ_HARMLESS:", args.max_read_harmless)
    print("DTYPE:", args.dtype)
    print("DO_SAMPLE:", args.do_sample)
    print("SOFT_MAX_GEN:", SOFT_MAX_GEN)
    print("MODELS:", DEFAULT_MODEL_PATHS)

    for mp in DEFAULT_MODEL_PATHS:
        run_one_model(args, mp)

    print("\nDone.")

if __name__ == "__main__":
    main()