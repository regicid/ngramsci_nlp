"""Qwen3.5-4B inline coreference (lattice-nlp/Qwen3.5-4B_inline_coreference_resolution, LoRA merged into its base
with `ngramsci merge-qwen`) on the k=10 windows, with vLLM. Greedy decoding, max_tokens = 1.25 x input.
Windows over 3,000 tokens are skipped (recorded with output None). Generation is written chunk by chunk to a .part
file in the shard's scratch dir, so a worker stopped at its deadline (or killed) resumes where it was.
Start-up options (vLLM's start-up dominates small batches): single_process (engine in this process, one import
instead of two) and eager (no torch.compile / CUDA graphs)."""
import gzip
import json
import os
import shutil
import time

from ..stage import Deadline, Output, Stage, register
from ..textio import read_jsonl

PROMPT = '''## Task
Replace each ___ with a numeric coreference ID.

## Rules
- Do NOT modify any other characters in the text.
- The braces { and } must remain unchanged.
- Only replace the substring ___.
- Start numbering from 0 for each document.
- Use the same ID for mentions referring to the same entity.
- Return only the modified text.

## Input
<input_placeholder>

## Output
'''


def model_dir(ctx):
    m = (ctx.site.models.get("qwen") if ctx.site else None) or ctx.p("qwen", "model")
    if not m:
        raise SystemExit("no Qwen model: set models.qwen in the site file (dir made by `ngramsci merge-qwen`)")
    return m


def load(ctx):
    if ctx.p("qwen", "single_process", True):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM
    llm = LLM(model=model_dir(ctx), dtype="bfloat16", tensor_parallel_size=ctx.p("qwen", "tp", 1), max_model_len=8192,
              gpu_memory_utilization=ctx.p("qwen", "gpu_memory_utilization", 0.90), enable_prefix_caching=True,
              enforce_eager=ctx.p("qwen", "eager", False), limit_mm_per_prompt={"image": 0, "video": 0})
    return llm, llm.get_tokenizer()


def prompt(tok, text):
    msgs = [{"role": "user", "content": PROMPT.replace("<input_placeholder>", text)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False).replace("<think>\n\n</think>\n\n", "")


def generate(llm, tok, wins, part, chunk=20000, max_in=3000, deadline=None, log=print):
    from vllm import SamplingParams
    done = set()
    if os.path.exists(part):
        for l in open(part, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["file"], r["w"]))
            except json.JSONDecodeError:
                pass                                      # torn last line of a killed run
    todo = [w for w in wins if (w["file"], w["w"]) not in done]
    t0, n_tok = time.time(), 0
    with open(part, "a", encoding="utf-8") as f:
        for c in range(0, len(todo), chunk):
            if deadline and time.time() > deadline:
                raise Deadline()
            prompts, params, keep = [], [], []
            for w in todo[c:c + chunk]:
                n_in = len(tok(w["input"])["input_ids"])
                if n_in > max_in:
                    f.write(json.dumps({"file": w["file"], "w": w["w"], "output": None, "skipped": f"too long ({n_in} tok)"}) + "\n")
                    continue
                prompts.append(prompt(tok, w["input"]))
                params.append(SamplingParams(temperature=0, max_tokens=max(int(1.25 * n_in), n_in + 20)))
                keep.append(w)
            for w, o in zip(keep, llm.generate(prompts, params, use_tqdm=False)):
                n_tok += len(o.outputs[0].token_ids)
                f.write(json.dumps({"file": w["file"], "w": w["w"], "output": o.outputs[0].text.strip()}, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
            log(f"  {min(c + chunk, len(todo))}/{len(todo)} windows, {n_tok / max(1e-9, time.time() - t0):.0f} gen tok/s")
    return len(todo), n_tok, time.time() - t0


def run(k, inp, out, state, ctx):
    llm, tok = state
    wins = read_jsonl(inp["windows"])
    part = os.path.join(ctx.scratch, "qwen.jsonl.part")
    n, n_tok, dt = generate(llm, tok, wins, part, chunk=ctx.p("qwen", "chunk", 20000), max_in=ctx.p("qwen", "max_in", 3000),
                            deadline=ctx.deadline, log=ctx.log)
    with open(part, "rb") as fi, gzip.open(out["qwen"], "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return {"windows": len(wins), "new": n, "gen_tok_s": round(n_tok / max(dt, 1e-9))}


register(Stage("qwen", inputs=["windows"], outputs=[Output("qwen", "qwen_k10/shard_{k:05d}.ner.jsonl.gz")],
               env="vllm", gpu=True, ttl_hours=6, load=load, run=run, doc=__doc__))
