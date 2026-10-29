#!/usr/bin/env python3
"""A vLLM stand-in on Apple Silicon (MLX) for H2O-Lightning-4B, so the official h2o_lightning_shim.py runs UNMODIFIED.

It implements only what the shim calls on vLLM:
    GET  /v1/models            -> the served model name and max_model_len
    POST /tokenize             -> {"prompt", "add_special_tokens"} or {"messages", "continue_final_message", ...}
    POST /detokenize           -> {"tokens"}
    POST /v1/completions       -> max_tokens=1 read-out: log-probabilities of `logprob_token_ids` at the next slot
    POST /v1/chat/completions  -> the same read-out for a chat with images (continue_final_message), or, with a
                                  `response_format` JSON schema, the guided Box generation (below)
    GET  /health

BOX QUESTIONS (H2O v1.4+, always on): the shim asks for one greedy chat generation constrained to a small JSON schema
(`{"bbox_2d": [x1, y1, x2, y2]}`, integers 0-1000, or the `box_2d` variant), with per-token logprobs. This facade
constrains decoding to exactly that schema (a character-level grammar, checked token by token) and returns vLLM's
chat-completion shape, including `logprobs.content[].top_logprobs`. The log-probabilities are the model's own (raw
log-softmax over the whole vocabulary, before the grammar mask), as vLLM reports them. Any other `response_format` is
refused with a 422 that names response_format, which the shim turns into its BOX_NEEDS_GUIDED answer for that
question only.

TWO-WAY READING (H2O v1.3+): a completion or chat request with `"vllm_xargs": {"attention_mode": "bidirectional"}`
runs the prompt with bidirectional attention on the full-attention layers (the Gated DeltaNet layers stay causal) and
reads the answer slot exactly as one-way does. This is the per-request flag the shim sends when started with
SHIM_TWO_WAY_FLAG=1 (upstream's one-engine mode); before its first two-way answer the shim checks this engine's
answers to three fixed probes against the release's recorded two-way answers. A two-way prompt is read in one pass,
without the prefix cache (an earlier token's state depends on the later tokens). "causal" or no flag = one-way.

Nothing is ever sampled. One forward pass per request; the hidden state at the last position goes through the tied
output head in float32 (the model's config.json asks for "head_dtype": "float32", which vLLM honours), and the reply
carries log_softmax over the whole vocabulary for the requested ids -- vLLM's raw log-probabilities.

The weights are the official bf16 safetensors, loaded by mlx-vlm without quantization.
MLX work is serialised by one lock. The model loads on first use and is released after IDLE_UNLOAD_S of no requests
(0 = keep resident).

    python mlx_vllm_facade.py --model ./model --port 8743
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

SERVED_NAME = os.environ.get("FACADE_SERVED_NAME", "h2oai/h2o-lightning-4b")
MAX_MODEL_LEN = int(os.environ.get("FACADE_MAX_MODEL_LEN", "40960"))
PREFILL_CHUNK = int(os.environ.get("FACADE_PREFILL_CHUNK", "4096"))
IDLE_UNLOAD_S = float(os.environ.get("FACADE_IDLE_UNLOAD_S", "900"))
DEFAULT_MAX_PIXELS = int(os.environ.get("FACADE_MAX_PIXELS", "1605632"))     # vLLM --mm-processor-kwargs in serve.sh
MAX_IMAGES = int(os.environ.get("FACADE_MAX_IMAGES", "4"))
HEAD_SLICE = int(os.environ.get("FACADE_HEAD_SLICE", "32768"))
# MLX keeps freed buffers in a cache; bound it so the resident footprint stays near the weights' size
CACHE_LIMIT_GB = float(os.environ.get("FACADE_CACHE_LIMIT_GB", "1"))
PREFIX_MIN = int(os.environ.get("FACADE_PREFIX_MIN", "256"))     # 0 turns the prefix cache off
CONTEXT_MSG = "This model's maximum context length is {m} tokens. However, you requested {n} tokens."
ATTENTION_MODES = {"causal": False, "one-way": False, "bidirectional": True, "two-way": True}
_DATA_URI = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S | re.I)


# Two-way attention: mlx-vlm's qwen3_5 language model builds one mask for all full-attention layers in
# _create_qwen3_5_attention_mask; while _BIDIR is set (only ever under Engine.lock) that mask is None, i.e. every
# position attends to every position. The Gated DeltaNet layers use their own (causal, recurrent) path, untouched.
_BIDIR = False


def _install_two_way_patch():
    import mlx_vlm.models.qwen3_5.language as L
    orig = getattr(L, "_create_qwen3_5_attention_mask", None)
    if orig is None:
        raise RuntimeError("mlx-vlm's qwen3_5 has no _create_qwen3_5_attention_mask: two-way reading needs updating "
                           "for this mlx-vlm version")
    if getattr(orig, "_two_way", False):
        return

    def mask(h, cache):
        return None if _BIDIR else orig(h, cache)
    mask._two_way = True
    L._create_qwen3_5_attention_mask = mask


def attention_mode(body):
    """True for a two-way (bidirectional) request: vLLM's per-request `vllm_xargs.attention_mode`."""
    x = body.get("vllm_xargs") or {}
    m = x.get("attention_mode") if isinstance(x, dict) else None
    if m is None:
        return False
    key = str(m).strip().lower().replace("_", "-")
    if key not in ATTENTION_MODES:
        raise HTTPError(400, f"vllm_xargs.attention_mode must be one of {sorted(ATTENTION_MODES)}")
    return ATTENTION_MODES[key]


class HTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------------------------- guided Box generation
BOX_MAX_TOKENS = int(os.environ.get("FACADE_BOX_MAX_TOKENS", "48"))
_WS = " \t\n\r"


def box_schema_key(response_format):
    """The array property name of a Box JSON schema ("bbox_2d" / "box_2d"), or an HTTPError(422) for anything else."""
    rf = response_format if isinstance(response_format, dict) else {}
    js = rf.get("json_schema") if rf.get("type") == "json_schema" else None
    schema = js.get("schema") if isinstance(js, dict) else None
    try:
        props = schema["properties"]
        (key, spec), = props.items()
        items = spec["items"]
        ok = (schema.get("type") == "object" and spec.get("type") == "array" and items.get("type") == "integer"
              and items.get("minimum") == 0 and items.get("maximum") == 1000
              and spec.get("minItems") == 4 and spec.get("maxItems") == 4 and schema.get("required") == [key])
    except (TypeError, KeyError, AttributeError, ValueError):
        ok = False
    if not ok:
        raise HTTPError(422, "response_format: this MLX engine supports only the Box JSON schema "
                             "({\"<name>\": [4 integers 0-1000]}); other structured outputs are not available")
    return key


def box_scan(s, key):
    """'bad' | 'partial' | 'complete' for text s against  {"<key>": [n, n, n, n]}  (JSON whitespace anywhere between
    tokens, integers 0..1000 without leading zeros), the language of the Box schema."""
    items = ["{", f'"{key}"', ":", "[", 0, ",", 0, ",", 0, ",", 0, "]", "}"]
    i, n = 0, len(s)
    for it in items:
        while i < n and s[i] in _WS:
            i += 1
        if i == n:
            return "partial"
        if it == 0:
            j = i
            while j < n and s[j] in "0123456789":
                j += 1
            num = s[i:j]
            if not num or (len(num) > 1 and num[0] == "0") or int(num) > 1000:
                return "bad"
            if j == n:
                return "partial"
            i = j
        else:
            seg = s[i:i + len(it)]
            if not it.startswith(seg):
                return "bad"
            if len(seg) < len(it):
                return "partial"
            i += len(it)
    return "complete" if not s[i:].strip(_WS) else "bad"


class Engine:
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.lock = threading.RLock()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.last_used = time.time()
        self.loads = 0
        self.snapshot = None          # (prefix ids, saved cache state) for the prefix cache
        self.last_ids = None
        self.prefix_hits = 0
        cfg = json.load(open(os.path.join(self.path, "config.json")))
        self.head_fp32 = cfg.get("head_dtype", cfg.get("text_config", {}).get("head_dtype")) == "float32"
        self.image_token_id = cfg.get("image_token_id")
        # the tokenizer is cheap and needed for /tokenize even while the weights are released. H2O v1.4.0.1+ ships the
        # training tokenizer (Qwen2Tokenizer) in tokenizer/ (vLLM gets --tokenizer <model>/tokenizer from serve.sh):
        # transformers 5.18+ would otherwise load Qwen3_5Tokenizer, which splits some Devanagari and Arabic text
        # differently. The shim checks /tokenize against tokenizer_contract.json at start.
        from transformers import AutoTokenizer
        tok_dir = os.path.join(self.path, "tokenizer")
        self.tokenizer_path = tok_dir if os.path.isdir(tok_dir) else self.path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    # ------------------------------------------------------------------ lifecycle
    def ensure_loaded(self):
        if self.model is not None:
            return
        from mlx_vlm import load
        t0 = time.time()
        model, processor = load(self.path)
        _install_two_way_patch()
        if self.tokenizer_path != self.path and hasattr(processor, "tokenizer"):
            processor.tokenizer = self.tokenizer      # image prompts are tokenized by the same training tokenizer
        self.model, self.processor = model, processor
        self.loads += 1
        lm = self.model.language_model
        self.embed = lm.model.embed_tokens
        self.lm_head = getattr(lm, "lm_head", None)      # None when the embeddings are tied (this model)
        sys.stderr.write(f"facade: loaded {self.path} in {time.time() - t0:.1f}s "
                         f"(dtype {self.embed.weight.dtype}, head fp32={self.head_fp32}, tied={self.lm_head is None})\n")

    def maybe_unload(self):
        if not IDLE_UNLOAD_S or self.model is None:
            return
        with self.lock:
            if self.model is not None and time.time() - self.last_used > IDLE_UNLOAD_S:
                self.model = self.processor = self.embed = self.lm_head = None
                self.snapshot = self.last_ids = None
                gc.collect()
                mx.clear_cache()
                sys.stderr.write("facade: idle, weights released\n")

    # ------------------------------------------------------------------ text helpers
    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=bool(add_special_tokens))

    def render_chat(self, messages, add_generation_prompt=False, continue_final_message=False, kwargs=None):
        """Chat messages -> prompt text through the model's own chat template; image parts become the template's
        image slot (as vLLM does before its processor expands the pad)."""
        msgs, urls = [], []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    t = p.get("type")
                    if t in ("image_url", "image", "input_image"):
                        u = p.get("image_url") if t != "image" else p.get("image")
                        u = u.get("url") if isinstance(u, dict) else u
                        urls.append(u)
                        parts.append({"type": "image"})
                    elif t == "text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    else:
                        raise HTTPError(400, f"unsupported content part {t!r}")
                msgs.append({"role": m["role"], "content": parts})
            else:
                msgs.append({"role": m["role"], "content": c})
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message, **(kwargs or {}))
        return text, urls

    # ------------------------------------------------------------------ the read-out
    def _final_hidden(self, ids, pixel_values=None, grid=None, bidirectional=False):
        """Normalised hidden state at the last position, shape (hidden,). bidirectional: the full-attention layers
        see the whole prompt (one pass, no prefix cache)."""
        global _BIDIR
        lm = self.model.language_model
        input_ids = mx.array([ids], dtype=mx.int32)
        if pixel_values is not None:
            feats = self.model.get_input_embeddings(input_ids, pixel_values=pixel_values, image_grid_thw=grid)
        else:
            feats = self.model.get_input_embeddings(input_ids)
        emb, pos = feats.inputs_embeds, feats.position_ids
        if pos is None:
            pos = mx.arange(len(ids), dtype=mx.int32)[None]
        if pos.ndim == 2:                                   # (B, L) -> the (3, B, L) M-RoPE layout
            pos = mx.broadcast_to(pos[None], (3,) + tuple(pos.shape))
        n = len(ids)
        start, cache, split = 0, None, None
        if bidirectional:
            cache = lm.make_cache()
            _BIDIR = True
            try:
                h = lm.model(input_ids, inputs_embeds=emb, cache=cache, position_ids=pos)
                mx.eval(h)
            finally:
                _BIDIR = False
            return h[0, -1]
        if pixel_values is None and PREFIX_MIN:
            # PREFIX CACHE (text only). Several questions about one record share a long head; vLLM computes it once
            # (prefix caching). Here: if this prompt shares >= PREFIX_MIN tokens with the saved snapshot, restore it;
            # otherwise, if it shares >= PREFIX_MIN tokens with the previous prompt, snapshot at that point this time.
            snap = self.snapshot
            if snap and n > len(snap[0]) and tuple(ids[:len(snap[0])]) == snap[0]:
                start, cache = len(snap[0]), self._restore(snap[1])
                self.prefix_hits += 1
            else:
                prev = self.last_ids
                if prev:
                    m = min(len(prev), n - 1)
                    k = next((i for i in range(m) if prev[i] != ids[i]), m)
                    if k >= PREFIX_MIN:
                        split = k
        if cache is None:
            cache = lm.make_cache()
        h = None
        bounds = [b for b in range(start, n, PREFILL_CHUNK)] + [n]
        if split:
            bounds = sorted(set(bounds) | {split})
        for s, e in zip(bounds[:-1], bounds[1:]):
            h = lm.model(input_ids[:, s:e], inputs_embeds=emb[:, s:e], cache=cache, position_ids=pos[..., s:e])
            mx.eval(h)
            if split and e == split:
                self.snapshot = (tuple(ids[:split]), self._save(cache))
        if pixel_values is None:
            self.last_ids = list(ids)
        return h[0, -1]

    @staticmethod
    def _copy(a):
        return None if a is None else (a + 0 if a.dtype != mx.bool_ else mx.array(a))

    def _save(self, cache):
        st = [[self._copy(a) for a in c.state] for c in cache]
        mx.eval([a for s in st for a in s if a is not None])
        return st

    def _restore(self, saved):
        cache = self.model.language_model.make_cache()
        for c, st in zip(cache, saved):
            c.state = [self._copy(a) for a in st]
        return cache

    def logprobs_at_next(self, ids, wanted, pixel_values=None, grid=None, bidirectional=False):
        if len(ids) + 1 > MAX_MODEL_LEN:
            raise HTTPError(400, CONTEXT_MSG.format(m=MAX_MODEL_LEN, n=len(ids) + 1))
        h = self._final_hidden(ids, pixel_values, grid, bidirectional)
        lp = self._log_softmax(h)
        sel = lp[mx.array(wanted, dtype=mx.int32)]
        top = int(mx.argmax(lp).item())
        mx.eval(sel)
        return [float(x) for x in sel.tolist()], top, float(lp[top].item())

    def _log_softmax(self, h):
        """Log-softmax over the whole vocabulary for one hidden state (the read-out the shim asks vLLM for)."""
        head = self.embed if self.lm_head is None else self.lm_head
        W = head.weight
        q = getattr(head, "scales", None) is not None          # a quantized (q8/q4) MLX checkpoint
        if self.head_fp32:
            # float32 head, in vocabulary slices so no 2.5 GB float32 copy of the head is ever held;
            # a quantized head is dequantized slice by slice first (same values the quantized matmul would use)
            hf = h.astype(mx.float32)
            parts = []
            for s in range(0, W.shape[0], HEAD_SLICE):
                if q:
                    b = getattr(head, "biases", None)
                    w = mx.dequantize(W[s:s + HEAD_SLICE], head.scales[s:s + HEAD_SLICE],
                                      None if b is None else b[s:s + HEAD_SLICE],
                                      group_size=head.group_size, bits=head.bits,
                                      **({"mode": head.mode} if getattr(head, "mode", None) else {}))
                else:
                    w = W[s:s + HEAD_SLICE]
                parts.append(w.astype(mx.float32) @ hf)
                mx.eval(parts[-1])
            logits = mx.concatenate(parts)
        else:
            logits = (head.as_linear(h) if self.lm_head is None else head(h)).astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        mx.eval(lp)
        return lp

    # ------------------------------------------------------------------ guided Box generation
    def _token_text(self, tid):
        cache = getattr(self, "_tok_text", None)
        if cache is None:
            cache = self._tok_text = {}
        t = cache.get(tid)
        if t is None:
            t = cache[tid] = self.tokenizer.decode([tid])
        return t

    def generate_box(self, ids, key, max_tokens, top_n, pixel_values=None, grid=None):
        """Greedy decoding constrained to the Box schema. Returns (text, [(token_id, logprob, [(id, lp), ...]), ...],
        finish_reason). One prefill (chunked, with the image features), then one cached step per token; text tokens
        after the prompt continue the M-RoPE positions at max + 1, as the model's own position rule does."""
        lm = self.model.language_model
        input_ids = mx.array([ids], dtype=mx.int32)
        if pixel_values is not None:
            feats = self.model.get_input_embeddings(input_ids, pixel_values=pixel_values, image_grid_thw=grid)
        else:
            feats = self.model.get_input_embeddings(input_ids)
        emb, pos = feats.inputs_embeds, feats.position_ids
        if pos is None:
            pos = mx.arange(len(ids), dtype=mx.int32)[None]
        if pos.ndim == 2:
            pos = mx.broadcast_to(pos[None], (3,) + tuple(pos.shape))
        cache = lm.make_cache()
        n = len(ids)
        bounds = list(range(0, n, PREFILL_CHUNK)) + [n]
        h = None
        for s, e in zip(bounds[:-1], bounds[1:]):
            h = lm.model(input_ids[:, s:e], inputs_embeds=emb[:, s:e], cache=cache, position_ids=pos[..., s:e])
            mx.eval(h)
        next_pos = int(mx.max(pos).item()) + 1
        eos = set()
        for t in (self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")):
            if isinstance(t, int):
                eos.add(t)
        text, out, finish = "", [], "length"
        for _ in range(max(1, int(max_tokens))):
            lp = self._log_softmax(h[0, -1])
            order = mx.argsort(-lp)
            chosen = None
            for start in range(0, lp.shape[0], 4096):          # candidates in descending probability
                for tid in order[start:start + 4096].tolist():
                    if tid in eos:
                        continue
                    piece = self._token_text(tid)
                    if piece and box_scan(text + piece, key) != "bad":
                        chosen = tid
                        break
                if chosen is not None:
                    break
            if chosen is None:
                raise HTTPError(500, "guided Box generation found no token that keeps the reply valid")
            top_ids = order[:max(int(top_n), 1)].tolist() if top_n else []
            lps = lp[mx.array(top_ids + [chosen], dtype=mx.int32)].tolist()
            out.append((chosen, float(lps[-1]), list(zip(top_ids, [float(x) for x in lps[:-1]]))))
            text += self._token_text(chosen)
            if box_scan(text, key) == "complete":
                finish = "stop"
                break
            tok = mx.array([[chosen]], dtype=mx.int32)
            h = lm.model(tok, inputs_embeds=self.embed(tok), cache=cache,
                         position_ids=mx.full((3, 1, 1), next_pos, dtype=mx.int32))
            mx.eval(h)
            next_pos += 1
        return text, out, finish

    # ------------------------------------------------------------------ images
    @staticmethod
    def load_image(u):
        from PIL import Image
        if not isinstance(u, str):
            raise HTTPError(422, "image must be a URL or data URI")
        m = _DATA_URI.match(u.strip())
        try:
            if m:
                raw = base64.b64decode(m.group(2), validate=False)
            elif u.startswith(("http://", "https://")):
                with urllib.request.urlopen(u, timeout=10) as r:
                    raw = r.read(25 * 1024 * 1024)
            else:
                raw = base64.b64decode(u)
            im = Image.open(io.BytesIO(raw))
            im.load()
            return im.convert("RGB")
        except HTTPError:
            raise
        except Exception as e:
            raise HTTPError(422, f"cannot load image: {type(e).__name__}: {e}")

    def load_images(self, urls):
        """Fetched/decoded OUTSIDE the MLX lock, so a slow URL never stalls other requests."""
        if len(urls) > MAX_IMAGES:
            raise HTTPError(422, f"at most {MAX_IMAGES} images per prompt")
        return [self.load_image(u) for u in urls]

    def process_images(self, text, images, max_pixels):
        ip = self.processor.image_processor
        min_px = getattr(ip, "min_pixels", None) or (getattr(ip, "size", None) or {}).get("shortest_edge", 65536)
        try:
            # mlx-vlm's Qwen3-VL image processor takes max_pixels / min_pixels (a `size` argument is ignored by it), so
            # the shim's cap (mm_processor_kwargs.max_pixels) resizes exactly as vLLM's smart_resize does
            out = self.processor(text=[text], images=images, return_tensors="np",
                                 max_pixels=int(max_pixels), min_pixels=int(min_px))
        except Exception as e:
            raise HTTPError(422, f"image preprocessing failed: {type(e).__name__}: {e}")
        ids = [int(x) for x in out["input_ids"][0]]
        return ids, mx.array(out["pixel_values"]), mx.array(out["image_grid_thw"])

    # ------------------------------------------------------------------ requests
    def handle_completions(self, body):
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = prompt
        elif isinstance(prompt, str):
            ids = self.encode(prompt, body.get("add_special_tokens", True))
        else:
            raise HTTPError(400, "prompt must be a string or a list of token ids")
        if int(body.get("max_tokens", 16)) != 1:
            raise HTTPError(400, "this server only reads the next token (max_tokens must be 1)")
        wanted = [int(i) for i in (body.get("logprob_token_ids") or [])]
        bidir = attention_mode(body)
        with self.lock:
            self.ensure_loaded()
            self.last_used = time.time()
            lps, top, top_lp = self.logprobs_at_next(ids, wanted, bidirectional=bidir)
            self.last_used = time.time()
        top_map = {f"token_id:{i}": v for i, v in zip(wanted, lps)}
        return {"id": "cmpl-mlx", "object": "text_completion", "created": int(time.time()), "model": SERVED_NAME,
                "choices": [{"index": 0, "text": self.tokenizer.decode([top]), "finish_reason": "length",
                             "logprobs": {"tokens": [f"token_id:{top}"], "token_logprobs": [top_lp],
                                          "top_logprobs": [top_map], "text_offset": [0]},
                             **({"prompt_token_ids": ids} if body.get("return_token_ids") else {})}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1}}

    def handle_chat(self, body):
        if body.get("response_format") is not None:
            return self.handle_box(body)
        if int(body.get("max_completion_tokens", body.get("max_tokens", 1)) or 1) != 1:
            raise HTTPError(400, "this server only reads the next token (max_completion_tokens must be 1), or "
                                 "generates a Box reply under a response_format JSON schema")
        cont = bool(body.get("continue_final_message", False))
        text, urls = self.render_chat(body.get("messages") or [], body.get("add_generation_prompt", not cont),
                                      cont, body.get("chat_template_kwargs"))
        wanted = [int(i) for i in (body.get("logprob_token_ids") or [])]
        max_px = int((body.get("mm_processor_kwargs") or {}).get("max_pixels") or DEFAULT_MAX_PIXELS)
        bidir = attention_mode(body)
        images = self.load_images(urls) if urls else []
        with self.lock:
            self.ensure_loaded()
            self.last_used = time.time()
            if urls:
                ids, pv, grid = self.process_images(text, images, max_px)
                lps, top, top_lp = self.logprobs_at_next(ids, wanted, pv, grid, bidirectional=bidir)
            else:
                ids = self.encode(text, body.get("add_special_tokens", False))
                lps, top, top_lp = self.logprobs_at_next(ids, wanted, bidirectional=bidir)
            self.last_used = time.time()
        content = [{"token": f"token_id:{top}", "logprob": top_lp,
                    "top_logprobs": [{"token": f"token_id:{i}", "logprob": v} for i, v in zip(wanted, lps)]}]
        return {"id": "chatcmpl-mlx", "object": "chat.completion", "created": int(time.time()), "model": SERVED_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.tokenizer.decode([top])},
                             "logprobs": {"content": content}, "finish_reason": "length",
                             **({"prompt_token_ids": ids} if body.get("return_token_ids") else {})}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1}}

    def handle_box(self, body):
        """The Box grounding call: one guided generation at temperature 0, thinking off, with per-token logprobs."""
        key = box_schema_key(body.get("response_format"))
        if attention_mode(body):
            raise HTTPError(400, "a Box generation is read one-way (vllm_xargs.attention_mode must not be bidirectional)")
        if float(body.get("temperature", 0) or 0) != 0:
            raise HTTPError(400, "the Box generation is greedy (temperature must be 0)")
        max_tokens = min(int(body.get("max_completion_tokens", body.get("max_tokens", BOX_MAX_TOKENS)) or BOX_MAX_TOKENS),
                         BOX_MAX_TOKENS)
        top_n = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        text_prompt, urls = self.render_chat(body.get("messages") or [], body.get("add_generation_prompt", True),
                                             False, body.get("chat_template_kwargs"))
        max_px = int((body.get("mm_processor_kwargs") or {}).get("max_pixels") or DEFAULT_MAX_PIXELS)
        images = self.load_images(urls) if urls else []
        with self.lock:
            self.ensure_loaded()
            self.last_used = time.time()
            if urls:
                ids, pv, grid = self.process_images(text_prompt, images, max_px)
            else:
                ids, pv, grid = self.encode(text_prompt, body.get("add_special_tokens", False)), None, None
            if len(ids) + max_tokens > MAX_MODEL_LEN:
                raise HTTPError(400, CONTEXT_MSG.format(m=MAX_MODEL_LEN, n=len(ids) + max_tokens))
            text, toks, finish = self.generate_box(ids, key, max_tokens, top_n, pv, grid)
            self.snapshot = self.last_ids = None              # the text prefix cache never spans a generation
            self.last_used = time.time()
        choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": [
                {"token": self._token_text(t), "logprob": lp,
                 "top_logprobs": [{"token": self._token_text(a), "logprob": alp} for a, alp in alts]}
                for t, lp, alts in toks]}
        return {"id": "chatcmpl-mlx-box", "object": "chat.completion", "created": int(time.time()), "model": SERVED_NAME,
                "choices": [choice],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks), "total_tokens": len(ids) + len(toks)}}

    def handle_tokenize(self, body):
        if "messages" in body:
            cont = bool(body.get("continue_final_message", False))
            text, _ = self.render_chat(body["messages"], body.get("add_generation_prompt", not cont),
                                       cont, body.get("chat_template_kwargs"))
            ids = self.encode(text, body.get("add_special_tokens", False))
        else:
            ids = self.encode(body.get("prompt", ""), body.get("add_special_tokens", True))
        return {"tokens": ids, "count": len(ids), "max_model_len": MAX_MODEL_LEN}


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args):
            if os.environ.get("FACADE_ACCESS_LOG") == "1":
                sys.stderr.write("facade %s\n" % (fmt % args))

        def send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p in ("/health", "/healthz"):
                return self.send(200, {"ok": True, "model": SERVED_NAME, "loaded": engine.model is not None,
                                       "loads": engine.loads, "backend": "mlx", "head_fp32": engine.head_fp32,
                                       "prefix_hits": engine.prefix_hits, "prefix_min": PREFIX_MIN,
                                       "attention_modes": ["causal", "bidirectional"],
                                       "structured_outputs": ["box json_schema"]})
            if p == "/v1/models":
                return self.send(200, {"object": "list", "data": [{
                    "id": SERVED_NAME, "object": "model", "owned_by": "mlx-facade", "max_model_len": MAX_MODEL_LEN}]})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?", 1)[0]
            if self.headers.get("Origin") or not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self.send(403, {"object": "error", "message": "local JSON clients only", "code": 403})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if body.get("model", SERVED_NAME) != SERVED_NAME:
                    raise HTTPError(404, f"The model `{body.get('model')}` does not exist.")
                if p == "/tokenize":
                    return self.send(200, engine.handle_tokenize(body))
                if p == "/detokenize":
                    return self.send(200, {"prompt": engine.tokenizer.decode(body.get("tokens") or [])})
                if p == "/v1/completions":
                    return self.send(200, engine.handle_completions(body))
                if p == "/v1/chat/completions":
                    return self.send(200, engine.handle_chat(body))
                self.send(404, {"error": "not found"})
            except HTTPError as e:
                self.send(e.status, {"object": "error", "message": str(e), "code": e.status})
            except json.JSONDecodeError as e:
                self.send(400, {"object": "error", "message": f"invalid JSON: {e}", "code": 400})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send(500, {"object": "error", "message": f"{type(e).__name__}: {e}", "code": 500})
    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(os.environ.get("H2O_MODEL_DIR", "./model")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8743)
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    mx.set_cache_limit(int(CACHE_LIMIT_GB * (1 << 30)))
    engine = Engine(a.model)
    if a.preload:
        with engine.lock:
            engine.ensure_loaded()

    def reaper():
        while True:
            time.sleep(30)
            try:
                engine.maybe_unload()
            except Exception as e:
                sys.stderr.write(f"facade: unload error {e}\n")
    threading.Thread(target=reaper, daemon=True).start()
    sys.stderr.write(f"facade: {SERVED_NAME} on http://{a.host}:{a.port} (mlx; idle unload {IDLE_UNLOAD_S:.0f}s)\n")
    Server((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
