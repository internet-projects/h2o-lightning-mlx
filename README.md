# H2O-Lightning-4B on Apple Silicon (MLX)

Run [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b), H2O.ai's calibrated decision model, on a Mac with MLX. Text and images both work, and so does **two-way reading**, new in H2O v1.3. The **official shim runs byte-for-byte unmodified**.

**Update, 19 October 2026:** this repo and the prebuilt weights now follow H2O-Lightning-4B **v1.3.1**, with two-way reading. The previous release (v1.2.3, one-way) stays available: `git checkout v1.2.3` here, and `--revision v1.2.3` on Hugging Face.

This is, as far as we could find on 12 October 2026, the first public MLX build of H2O-Lightning-4B. Please tell us if you know of another.

Prebuilt weights on Hugging Face: [netprojects/h2o-lightning-4b-mlx-q8](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q8) (8-bit) and [netprojects/h2o-lightning-4b-mlx-q4](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q4) (4-bit).

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with H2O.ai.

The official deployment is vLLM, which needs NVIDIA, plus a small stdlib shim. This repo replaces only the vLLM part. `mlx_vllm_facade.py` is a small server that answers the five vLLM calls the shim makes (`/v1/models`, `/tokenize`, `/detokenize`, `/v1/completions` and `/v1/chat/completions` with `logprob_token_ids`). It answers them with one MLX forward pass. Nothing is sampled. The hidden state at the answer slot goes through the tied output head in **float32**, as the model's `config.json` asks (`"head_dtype": "float32"`). The reply carries log-softmax over the full vocabulary for the requested label tokens. The shim's own start-up checks then verify the setup: all 255 labels are one token each, and the chat template renders the trained prompt.

```
client ──► shim :8741 (official, unmodified) ──► mlx_vllm_facade :8743 (this repo) ──► MLX (mlx-vlm Qwen3.5)
          [optional h2o_front.py :8742: API key + a single-question /v1/decisions shape]
```

## Two-way reading

H2O-Lightning-4B v1.3 was trained to answer under two attention patterns: one-way (causal, as before) and two-way, where the full-attention layers see the whole prompt in both directions while the Gated DeltaNet layers stay causal. Two-way is H2O's recommended default.

The facade serves both from one engine. The shim runs in H2O's one-engine mode (`SHIM_TWO_WAY_FLAG=1`, set by `serve.sh`) and marks each two-way request with `vllm_xargs.attention_mode`; the facade then runs that prompt with bidirectional attention on the full-attention layers, in one pass, and reads the answer slot exactly as one-way does. Before its first two-way answer, the shim checks the engine against the answers H2O recorded on the release weights (three fixed probes, tolerance 0.05). bf16 passes at 0.016 and 8-bit at 0.004.

A request picks its mode with `"attention_mode": "two-way"` or `"one-way"`; without it, `serve.sh` answers two-way (`TWO_WAY=0` serves one-way only).

## Results (MacBook Pro M5 Max, 128 GB)

JevBench public set (231 items), JevBench v1.2.1, `--adapter typesafe`, H2O v1.3.1 with the shipped `serve_config.json`:

| Checkpoint | Size | Two-way | One-way | Same answer as bf16 (two-way / one-way) | Probability drift vs bf16 (mean / p95) |
|---|---|---|---|---|---|
| model card (vLLM, NVIDIA) | 9.3 GB | 203 / 231 | 198 / 231 | — | — |
| bf16, official weights (MLX) | 9.3 GB | **205 / 231** | 198 / 231 | — | — |
| MLX 8-bit (g64) | 5.1 GB | **203 / 231** | 199 / 231 | 228 / 231 · 230 / 231 | 0.004 / 0.010 |
| MLX 4-bit (g64) | 3.0 GB | use 8-bit | 199 / 231 | — · 221 / 231 | 0.023 / 0.106 (one-way) |

- **Two-way** is the stronger mode, as on H2O's card: +7 on bf16 and +4 on 8-bit, gained on the hard tier (bf16 87/111, 8-bit 85/111). Easy is 48/48 everywhere.
- **8-bit** stays very close to the original weights in both modes and halves the memory. **This is the recommended build.**
- **4-bit** is for one-way reading (`TWO_WAY=0`). For two-way reading, use the 8-bit build.

Image check through the unchanged shim: H2O's receipt example gives the same answer on all three builds (bf16 and 8-bit 0.98 on the right option, 4-bit 0.88).

Speed on a quiet Mac: MacBook Pro M5 Max (128 GB), 23 October 2026, the published v1.3.1 builds through `serve.sh`, one request at a time, medians of 3. Times are from the client's side, so they include the shim and HTTP (1–2% of the total).

| | 8-bit two-way | 8-bit one-way | 4-bit one-way |
|---|---|---|---|
| JevBench, 120 items (median 130 tokens), per item | 66 ms | 68 ms | 61 ms |
| one ~2,000-token record | 466 ms | 447 ms | 424 ms |
| one 32K-token state | 15.8 s | 9.1 s | 8.6 s |
| MLX peak memory at 4K / 16K / 32K tokens | 8.2 / 22.4 / 55.0 GB | 7.7 / 8.3 / 8.9 GB | 5.7 / 6.3 / 6.8 GB |

The work is prefill-bound, so 4-bit is only slightly faster than 8-bit. Items of 500–800 tokens take about 0.15 s on 8-bit. **Two-way reads the whole prompt in one pass, so its memory grows much faster with long inputs** (55 GB at 32K tokens). For long inputs on a Mac with less than 64 GB, use one-way.

## Quick start

```bash
brew install uv
git clone https://github.com/internet-projects/h2o-lightning-mlx && cd h2o-lightning-mlx
./setup.sh q8            # venv + official H2O repo at v1.3.1 (shim, config, weights) + an 8-bit conversion; runs test_shim.py
MODEL=mlx-q8 ./serve.sh  # or ./serve.sh for the official bf16 weights
```

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Ticket 4471: checkout returns HTTP 500 for every customer since 09:10.",
  "questions": {"priority": {"type": "choice", "instructions": "Which priority?",
     "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}}}}'
```

The API, image input, status codes and routing thresholds are exactly as on the [model card](https://huggingface.co/h2oai/h2o-lightning-4b). Everything above was measured with the shim at H2O revision v1.3.1, which `setup.sh` pins (`test_shim.py` passes). For production routing, start with `SHIM_NOUL_FLOOR=0` so yes/no answers keep their raw probability (the card's "routing mode").

### Optional front door
`h2o_front.py` adds an API key and a `/v1/decisions` route for clients that send one question at a time:

```bash
H2O_API_KEY=… FRONT_PORT=8742 ./serve.sh
```

The request is `{"state", "question": {...}}`. The response is `{type, choice|noul|score, probabilities, confidence, input_tokens, latency_ms}`. Set `H2O_DECISION_LOG=<dir>` to keep one JSON-lines file per day of answer statistics (type, option count, top probability; never the record text). `launchd/` has an example LaunchAgent.

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
- **Tested versions.** mlx 0.32.4, mlx-vlm 0.7.8, transformers 5.20.1, H2O v1.3.1, macOS 26.7.
- **Scope.** This repo is the decision model and its serving path. It does not include any product harness.

## Credits and license
- Model: [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) by H2O.ai (Apache-2.0), built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache-2.0). The shim and `serve_config.json` are H2O.ai's and are downloaded from their repo, not redistributed here.
- Runtime: [MLX](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
- Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench).
- This repo's code: Apache-2.0, © NetProjects (Internet Projects Ltd). It is not affiliated with H2O.ai.
