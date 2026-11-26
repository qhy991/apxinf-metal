#!/usr/bin/env python3
"""Drive one engine at one quantization tier over the frozen S5 prompts.

Bare completion on both surfaces, no chat template, greedy, fixed seed: the S1-S3
control 1 and 2. Records the generated text and its token ids, tokenized with the
frozen LFM3 `tokenizer.json`, so the report measures agreement between the two tiers
of one arm without depending on either engine's own tokenizer endpoint.
"""

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

TIMEOUT_S = 300


def post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode())


def send(protocol, endpoint, model, prompt, max_tokens, seed, num_ctx):
    base = endpoint.rstrip("/")
    if protocol == "openai-completions":
        body = post(f"{base}/v1/completions",
                    {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                     "temperature": 0.0, "seed": seed, "stream": False})
        choice = (body.get("choices") or [{}])[0]
        return {"text": choice.get("text") or "",
                "finish_reason": choice.get("finish_reason"),
                "response_id": body.get("id")}
    if protocol == "ollama-generate":
        body = post(f"{base}/api/generate",
                    {"model": model, "prompt": prompt, "raw": True, "stream": False,
                     "options": {"num_predict": max_tokens, "temperature": 0.0,
                                 "seed": seed, "num_ctx": num_ctx}})
        return {"text": body.get("response") or "",
                "finish_reason": body.get("done_reason"),
                "response_id": str(body.get("created_at"))}
    raise SystemExit(f"unknown protocol {protocol!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True,
                        choices=["openai-completions", "ollama-generate"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_blob = pathlib.Path(args.prompts).read_bytes()
    prompts = json.loads(prompt_blob)["prompts"]

    records = []
    for entry in prompts:
        started = time.perf_counter()
        try:
            result = send(args.protocol, args.endpoint, args.model, entry["prompt"],
                          args.max_tokens, args.seed, args.num_ctx)
            status, detail = "ok", None
        except urllib.error.HTTPError as error:
            result = {"text": "", "finish_reason": None, "response_id": None}
            status, detail = "http_error", f"{error.code} {error.read()[:200]!r}"
        except Exception as error:                       # noqa: BLE001 - recorded, not raised
            result = {"text": "", "finish_reason": None, "response_id": None}
            status, detail = "error", f"{type(error).__name__}: {error}"
        wall_ms = (time.perf_counter() - started) * 1000.0
        ids = tokenizer.encode(result["text"], add_special_tokens=False).ids
        # Every set-specific key rides through untouched, so the probe never has to
        # know which prompt set it was handed. prompt_chars is the one size measure
        # both sets have, and it is what the model actually saw.
        records.append({**{key: value for key, value in entry.items() if key != "prompt"},
                        "prompt_chars": len(entry["prompt"]),
                        "prompt_sha256": hashlib.sha256(entry["prompt"].encode()).hexdigest(),
                        "status": status, "detail": detail, "wall_ms": round(wall_ms, 3),
                        "text": result["text"], "token_ids": ids,
                        "completion_tokens": len(ids),
                        "finish_reason": result["finish_reason"],
                        "response_id": result["response_id"]})
        print(f"{entry['id']} status={status} tokens={len(ids)} wall={wall_ms:.0f}ms", flush=True)

    document = {
        "schema": "apxinf-s5-quality-arm-v1",
        "arm": args.arm, "tier": args.tier, "protocol": args.protocol,
        "endpoint": args.endpoint, "model": args.model,
        "max_tokens": args.max_tokens, "seed": args.seed, "num_ctx": args.num_ctx,
        "greedy": True, "surface": "bare completion",
        "prompts_sha256": hashlib.sha256(prompt_blob).hexdigest(),
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": hashlib.sha256(pathlib.Path(args.tokenizer).read_bytes()).hexdigest(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
        "failures": sum(1 for record in records if record["status"] != "ok"),
    }
    pathlib.Path(args.out).write_text(json.dumps(document, indent=2, ensure_ascii=False))
    print(f"wrote {args.out} failures={document['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
