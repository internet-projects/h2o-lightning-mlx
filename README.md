# H2O-Lightning-4B on Apple Silicon (MLX)

Run [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b), H2O.ai's calibrated decision model, on a Mac with MLX. Text, images, **two-way reading** and **Box questions** (bounding boxes) all work. The **official shim runs byte-for-byte unmodified**.

**Update, 25 October 2026:** this repo and the prebuilt weights now follow H2O-Lightning-4B **v1.4.0**, H2O's new fine-tune with images in its training data. On JevBench's 231 public items the 8-bit build scores **206 / 231 two-way, the same as H2O's own result on vLLM**. Box questions, always on since v1.4.0, are answered by a guided generation in the facade. Earlier releases stay available: `git checkout v1.3.1` or `v1.2.3` here, and `--revision v1.3.1` or `v1.2.3` on Hugging Face.

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

JevBench public set (231 items), JevBench v1.2.1, `--adapter typesafe`, H2O v1.4.0 with the shipped `serve_config.json` (easy / standard / hard in brackets):

| Checkpoint | Size | Two-way | One-way | Same answer as bf16 (two-way / one-way) | Probability drift vs bf16 (mean / p95) |
|---|---|---|---|---|---|
| H2O's model card (vLLM, NVIDIA) | 9.3 GB | 206 / 231 (48 · 72 · 86) | — | — | — |
| bf16, official weights (MLX) | 9.3 GB | **206 / 231** (48 · 72 · 86) | 203 / 231 | — | — |
| MLX 8-bit (g64) | 4.8 GB | **206 / 231** (48 · 72 · 86) | 202 / 231 | **231 / 231** · 230 / 231 | 0.002 / 0.012 (two-way) |
| MLX 4-bit (g64) | 2.9 GB | use 8-bit | 202 / 231 | — · 222 / 231 | 0.039 / 0.165 (one-way) |

- **The original weights on MLX match H2O's own vLLM result exactly**, tier by tier.
- **8-bit** gives the same answer as bf16 on all 231 items two-way and halves the memory. **This is the recommended build.**
- **4-bit** is for one-way reading (`TWO_WAY=0`): H2O's start-up check refuses it for two-way (probe drift 0.079 against a tolerance of 0.05). For two-way reading, use the 8-bit build.
- **One-way on v1.4.0** is new here (H2O hasn't published it yet): 203 on bf16, 202 on 8-bit and 4-bit.
- H2O notes that v1.4.0 is weaker than v1.3.2 on retrieval (product-search relevance) and on choosing a function to call, and stronger on its held-out decision skills overall. v1.3.2 has the same weights as v1.3.1; if your work is mostly tool-call selection, compare on your own data, or stay on `v1.3.1`.

Image check through the unchanged shim, H2O's receipt example: "six item lines" at 0.94 on bf16 and 8-bit two-way (H2O's card: 0.94), and the same answer one-way on all three builds (0.94–0.95).

## Quick start

```bash
brew install uv
git clone https://github.com/internet-projects/h2o-lightning-mlx && cd h2o-lightning-mlx
./setup.sh q8            # venv + official H2O repo at v1.4.0 (shim, config, weights) + an 8-bit conversion; runs test_shim.py
MODEL=mlx-q8 ./serve.sh  # or ./serve.sh for the official bf16 weights
```

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Ticket 4471: checkout returns HTTP 500 for every customer since 09:10.",
  "questions": {"priority": {"type": "choice", "instructions": "Which priority?",
     "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}}}}'
```

The API, image input, status codes and routing thresholds are exactly as on the [model card](https://huggingface.co/h2oai/h2o-lightning-4b). Everything above was measured with the shim at H2O revision v1.4.0, which `setup.sh` pins (`test_shim.py` and `test_security.py` pass). For production routing, start with `SHIM_NOUL_FLOOR=0` so yes/no answers keep their raw probability (the card's "routing mode").

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
- **Long prompts.** They are pre-filled in 4,096-token chunks. Chunking changes log-probabilities by at most a few hundredths (bf16 arithmetic).
- **Idle release.** The weights load on first use and are released after `FACADE_IDLE_UNLOAD_S` idle seconds (0 keeps them resident).
- **Two-way prompts** are read in one pass without the prefix cache, since every position depends on the whole prompt.
- **Box generation** is greedy and constrained to H2O's two box schemas (`bbox_2d` and `box_2d`); other structured-output requests get a 422, which the shim reports for that question only.
- **Tested versions.** mlx 0.32.4, mlx-vlm 0.7.8, transformers 5.20.2, H2O v1.4.0, macOS 26.7.
- **Scope.** This repo is the decision model and its serving path. It does not include any product harness.

## Credits and license
- Model: [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) by H2O.ai (Apache-2.0), built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache-2.0). The shim and `serve_config.json` are H2O.ai's and are downloaded from their repo, not redistributed here.
- Runtime: [MLX](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
- Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench).
- This repo's code: Apache-2.0, © NetProjects (Internet Projects Ltd). It is not affiliated with H2O.ai.
