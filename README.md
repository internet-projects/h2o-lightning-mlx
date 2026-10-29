# H2O-Lightning-4B on Apple Silicon (MLX)

Run [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b), H2O.ai's calibrated decision model, on a Mac with MLX. Text, images, **two-way reading** and **Box questions** (bounding boxes) all work. The **official shim runs byte-for-byte unmodified**.

**Update, 29 October 2026:** H2O-Lightning-4B **v1.4.2** (commit `708e939`; weights unchanged since v1.4.1, so no reconversion, new serving files). Yes/no answers are served without the 0.801 floor by default; `PROFILE=di ./serve.sh` gives H2O's Decision Index profile (the floor plus a conditional re-read of choice and score questions under 0.8). Images up to 1,638,400 pixels are read whole, and the facade now applies the shim's image size cap exactly as vLLM does: before, it read every image at its full size, so images over the cap took more tokens than on vLLM. A request may give the attention mode as `kwargs.attention_mode`, and a client that disconnects early no longer leaves an error in the log. On JevBench the 8-bit build still scores 205 / 231 two-way and 201 / 231 one-way, with the same 231 answers as before. The 28 October files: `--revision v1.4.1-oct28` on Hugging Face.

**Update, 28 October 2026:** H2O revised v1.4.1 in place (commit `d7f1b05a`; the weights are unchanged, so the MLX builds were not reconverted). `setup.sh` now pins the revised release, which adds `selector_v2.py`: an over-long record is trimmed to fit the context by BM25 selection instead of a fixed cut at 32,000 tokens, so a request never fails for length (on a 60,000-token test record with one key line in the middle, the 8-bit build now answers correctly with confidence 0.97). Image answers are read at temperature 1.0 instead of 0.65: the same top answers, lower confidence values. Text requests that fit the context are read exactly as before. The original v1.4.1 files: `git checkout v1.4.1-initial` here and `--revision v1.4.1-initial` on Hugging Face.

**Update, 27 October 2026:** rebuilt from H2O-Lightning-4B **v1.4.1** (new upstream weights; H2O's calibration constants are unchanged). On JevBench's 231 public items the 8-bit build scores **205 / 231 two-way** (H2O's own result on vLLM: 206), with the same answer as the bf16 weights on all 231. The facade now reads text with the training tokenizer (Qwen2Tokenizer) that v1.4.1 ships in `tokenizer/`, and the shim's new tokenizer check passes. Earlier releases stay available: `git checkout v1.4.0`, `v1.3.1` or `v1.2.3` here, and the same `--revision` on Hugging Face.

This is, as far as we could find on 12 October 2026, the first public MLX build of H2O-Lightning-4B. Please tell us if you know of another.

Prebuilt weights on Hugging Face: [netprojects/h2o-lightning-4b-mlx-q8](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q8) (8-bit) and [netprojects/h2o-lightning-4b-mlx-q4](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q4) (4-bit).

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with H2O.ai.

The official deployment is vLLM, which needs NVIDIA, plus a small stdlib shim. This repo replaces only the vLLM part. `mlx_vllm_facade.py` is a small server that answers the five vLLM calls the shim makes (`/v1/models`, `/tokenize`, `/detokenize`, `/v1/completions` and `/v1/chat/completions` with `logprob_token_ids`). A decision is one MLX forward pass; nothing is sampled. The hidden state at the answer slot goes through the tied output head in **float32**, as the model's `config.json` asks (`"head_dtype": "float32"`). The reply carries log-softmax over the full vocabulary for the requested label tokens. The shim's own start-up checks then verify the setup: all 255 labels are one token each, and the chat template renders the trained prompt.

```
client ──► shim :8741 (official, unmodified) ──► mlx_vllm_facade :8743 (this repo) ──► MLX (mlx-vlm Qwen3.5)
          [optional h2o_front.py :8742: API key + a single-question /v1/decisions shape]
```

## Two-way reading

H2O-Lightning-4B v1.3 was trained to answer under two attention patterns: one-way (causal, as before) and two-way, where the full-attention layers see the whole prompt in both directions while the Gated DeltaNet layers stay causal. Two-way is H2O's recommended default.

The facade serves both from one engine. The shim runs in H2O's one-engine mode (`SHIM_TWO_WAY_FLAG=1`, set by `serve.sh`) and marks each two-way request with `vllm_xargs.attention_mode`; the facade then runs that prompt with bidirectional attention on the full-attention layers, in one pass, and reads the answer slot exactly as one-way does. Before its first two-way answer, the shim checks the engine against the answers H2O recorded on the release weights (three fixed probes, tolerance 0.05). On v1.4.0, bf16 passes at 0.007 and 8-bit at 0.012.

A request picks its mode with `"attention_mode": "two-way"` or `"one-way"`; without it, `serve.sh` answers two-way (`TWO_WAY=0` serves one-way only).

## Box questions

Since v1.4.0 a request can ask for a bounding box: ``{"type": "box", "instructions": "the total amount line in `IMG_1`"}``, with the image sent once under `"assets"` and referred to by name in `state`. The answer gives `box` (`left`, `top`, `right`, `bottom` as fractions of the image), `confidence`, `present` (whether the object is visible) and `found`. H2O's shim asks the engine for one generation constrained to a small JSON schema (`{"bbox_2d": [x1, y1, x2, y2]}`) with per-token log-probabilities. The facade does exactly that: greedy decoding held to the schema token by token, returning the model's own log-probabilities, so the shim's calibrated box confidence works unchanged.

On H2O's receipt example, asked for the total-amount line, the 8-bit build returns a box across the total line with confidence 0.96 (bf16: the same line, 0.96). Asked for an object that isn't there, it answers `present` 0.10.

## Results (MacBook Pro M5 Max, 128 GB)

JevBench public set (231 items), JevBench v1.2.1, `--adapter typesafe`, H2O v1.4.1 with the shipped `serve_config.json` (easy / standard / hard in brackets):

| Checkpoint | Size | Two-way | One-way | Same answer as bf16 (two-way / one-way) | Probability drift vs bf16 (mean / p95) |
|---|---|---|---|---|---|
| H2O's model card (vLLM, NVIDIA) | 9.3 GB | 206 / 231 (48 · 72 · 86) | 202 / 231 | — | — |
| bf16, official weights (MLX) | 9.3 GB | **205 / 231** (48 · 72 · 85) | 201 / 231 | — | — |
| MLX 8-bit (g64) | 4.8 GB | **205 / 231** (48 · 72 · 85) | 201 / 231 | **231 / 231** · **231 / 231** | 0.003 / 0.011 (two-way) |
| MLX 4-bit (g64) | 2.9 GB | use 8-bit | 205 / 231 | — · 221 / 231 | 0.043 / 0.176 (one-way) |

- **The original weights on MLX are within one item of H2O's own vLLM result** in both modes (two-way 205 against 206, one-way 201 against 202), all on hard items. The difference is a handful of hard items where the top two answers are almost tied, so arithmetic precision decides them: the same weights with every layer in float32 score 207 / 231 two-way.
- **8-bit** gives the same answer as bf16 on all 231 items in both modes and halves the memory. **This is the recommended build.** Its two-way start-up check passes at 0.014 (bf16: 0.002; tolerance 0.05).
- **4-bit** is for one-way reading (`TWO_WAY=0`): H2O's start-up check refuses it for two-way (probe drift 0.090 against a tolerance of 0.05). For two-way reading, use the 8-bit build.
- **Against our v1.4.0 build** (8-bit, two-way), v1.4.1 gives the same answer on 230 of 231 items (one-way: 226 of 231).
- H2O reports v1.4.1 as stronger than v1.4.0 on retrieval tasks and on six task families its added training data covers (including choosing a function to call and table fact checks), level on its held-out decision skills overall, and slightly weaker on language tasks. See H2O's card.

Image check through the unchanged shim, H2O's receipt example: "six item lines" at 0.92 two-way and 0.94 one-way on bf16 and 8-bit, and 0.94 one-way on 4-bit. The Box answer for the total-amount line is the same box on bf16 and 8-bit, at confidence 0.97.

## Speed and memory

MacBook Pro M5 Max (128 GB), 27 October 2026: the v1.4.1 builds through `serve.sh`, everything else stopped, one request at a time, medians of 3. Times are from the client's side, so they include the shim and HTTP.

| | 8-bit two-way | 8-bit one-way | 4-bit one-way |
|---|---|---|---|
| JevBench, 120 items (median 130 tokens), per item | 63 ms | 63 ms | 59 ms |
| one ~2,000-token record | 456 ms | 388 ms | 356 ms |
| one 4K-token record | 0.95 s | 0.81 s | 0.76 s |
| one 32K-token state | 14.3 s | 7.8 s | 7.4 s |
| MLX peak memory at 4K / 32K tokens | 8.2 / 55.0 GB | 7.7 / 8.9 GB | 5.7 / 6.8 GB |

The work is prefill-bound, so 4-bit is only slightly faster than 8-bit. **Two-way reads the whole prompt in one pass, so its memory grows much faster with long inputs** (55 GB at 32K tokens). For long inputs on a Mac with less than 64 GB, use one-way.

## Quick start

```bash
brew install uv
git clone https://github.com/internet-projects/h2o-lightning-mlx && cd h2o-lightning-mlx
./setup.sh q8            # venv + official H2O repo at v1.4.1 (shim, config, weights) + an 8-bit conversion; runs test_shim.py
MODEL=mlx-q8 ./serve.sh  # or ./serve.sh for the official bf16 weights
```

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Ticket 4471: checkout returns HTTP 500 for every customer since 09:10.",
  "questions": {"priority": {"type": "choice", "instructions": "Which priority?",
     "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}}}}'
```

The API, image input, status codes and routing thresholds are exactly as on the [model card](https://huggingface.co/h2oai/h2o-lightning-4b). `setup.sh` pins H2O v1.4.2 (`test_shim.py` and `test_security.py` pass: 109 tests). Since v1.4.2 yes/no answers keep their raw probability by default (the card's "routing mode"); `PROFILE=di` or `SHIM_NOUL_FLOOR=0.801` restores the floor.

### Optional front door
`h2o_front.py` adds an API key and a `/v1/decisions` route for clients that send one question at a time:

```bash
H2O_API_KEY=… FRONT_PORT=8742 ./serve.sh
```

The request is `{"state", "question": {...}}`, optionally with `"attention_mode"` (`"one-way"` / `"two-way"`), `"assets"` (image references) or `"images"`. The response is `{type, choice|noul|score|box, probabilities, confidence, attention_mode, input_tokens, latency_ms}`. Set `H2O_DECISION_LOG=<dir>` to keep one JSON-lines file per day of answer statistics (type, option count, top probability; never the record text). `launchd/` has an example LaunchAgent.

### Is the confidence usable?
VitaminC test split, 400 items (200 supported, 133 refuted, 67 not enough info), with the evidence sentence embedded in about 650 words of other text. H2O v1.2.3, one-way, all three MLX builds:

| Build | Accuracy | Brier (lower is better) | Calibration error (ECE) | AUROC |
|---|---|---|---|---|
| bf16 | 81.2% | 0.135 | 0.057 | 0.891 |
| 8-bit | 81.2% | 0.134 | 0.055 | 0.892 |
| 4-bit | 82.8% | 0.134 | 0.068 | 0.888 |

Accuracy differences are within noise (about ±4 points at 400 items). With the 8-bit build, accepting only answers at confidence 0.9 or higher covered 58% of the items at 90% accuracy. More in the [NetProjects article](https://netprojects.com/insights/h2o-lightning-4b-on-apple-silicon-mlx).

## Prefix cache

The facade keeps a prefix cache. When several questions share one record, the record's KV state is reused instead of being recomputed for each question. This speeds up repeated questions on the same record. If you change the cache settings, check the answers on your own data first: in our experiments, a more aggressive cache setting flipped one answer on a near-tie.

## Reproduce
```bash
git clone --depth 1 -b v1.2.1 https://github.com/fstandhartinger/jevbench jevbench
scripts/bench_jevbench.sh model bf16 two-way      # or one-way; or: scripts/bench_jevbench.sh mlx-q8 q8 two-way
uv pip install --python .venv/bin/python torch torchvision
.venv/bin/python scripts/parity_check.py --model model --jevbench jevbench --n 20
.venv/bin/python scripts/quant_check.py model mlx-q8 mlx-q4
```

## Notes and limits
- **Calibration is what H2O is for.** JevBench measures accuracy. Confidence values are only useful if you check them on your own data.
- **Serialized.** One MLX lock serves one forward pass at a time. That suits a single Mac, not high concurrency.
- **Over-long records.** A record that would not fit the context (40,960 tokens) is trimmed by H2O's BM25 selection (`selector_v2.py`, since the 28 October revision of v1.4.1); the question and its options are always kept whole.
- **Long prompts.** They are pre-filled in 4,096-token chunks. Chunking changes log-probabilities by at most a few hundredths (bf16 arithmetic).
- **Idle release.** The weights load on first use and are released after `FACADE_IDLE_UNLOAD_S` idle seconds (0 keeps them resident).
- **Two-way prompts** are read in one pass without the prefix cache, since every position depends on the whole prompt.
- **Box generation** is greedy and constrained to H2O's two box schemas (`bbox_2d` and `box_2d`); other structured-output requests get a 422, which the shim reports for that question only.
- **Image size.** The facade resizes each image to the shim's `max_pixels` (1,638,400 since v1.4.2) with the processor's own smart-resize, as vLLM does; smaller images are read as they are.
- **Tokenizer.** The facade reads `tokenizer/` (Qwen2Tokenizer, the training tokenizer, shipped by H2O since v1.4.0.1) when the model folder has it, for text and for image prompts; with transformers 5.18+ the default load would be Qwen3_5Tokenizer, which splits some Devanagari and Arabic text differently. The shim checks it at start against `tokenizer_contract.json` (24 probes).
- **Tested versions.** mlx 0.32.4, mlx-vlm 0.7.8, transformers 5.20.2, H2O v1.4.2, macOS 26.7.
- **Scope.** This repo is the decision model and its serving path. It does not include any product harness.

## Credits and license
- Model: [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) by H2O.ai (Apache-2.0), built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache-2.0). The shim and `serve_config.json` are H2O.ai's and are downloaded from their repo, not redistributed here.
- Runtime: [MLX](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
- Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench).
- This repo's code: Apache-2.0, © NetProjects (Internet Projects Ltd). It is not affiliated with H2O.ai.
