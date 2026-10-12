# H2O-Lightning-4B on Apple Silicon (MLX)

Run [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b), H2O.ai's calibrated decision model, on a Mac with MLX. Text and images both work. The **official shim runs byte-for-byte unmodified**.

This is, as far as we could find on 12 October 2026, the first public MLX build of H2O-Lightning-4B. Please tell us if you know of another.

Prebuilt weights on Hugging Face: [netprojects/h2o-lightning-4b-mlx-q8](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q8) (8-bit) and [netprojects/h2o-lightning-4b-mlx-q4](https://huggingface.co/netprojects/h2o-lightning-4b-mlx-q4) (4-bit).

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with H2O.ai.

The official deployment is vLLM, which needs NVIDIA, plus a small stdlib shim. This repo replaces only the vLLM part. `mlx_vllm_facade.py` is a small server that answers the five vLLM calls the shim makes (`/v1/models`, `/tokenize`, `/detokenize`, `/v1/completions` and `/v1/chat/completions` with `logprob_token_ids`). It answers them with one MLX forward pass. Nothing is sampled. The hidden state at the answer slot goes through the tied output head in **float32**, as the model's `config.json` asks (`"head_dtype": "float32"`). The reply carries log-softmax over the full vocabulary for the requested label tokens. The shim's own start-up checks then verify the setup: all 255 labels are one token each, and the chat template renders the trained prompt.

```
client ──► shim :8741 (official, unmodified) ──► mlx_vllm_facade :8743 (this repo) ──► MLX (mlx-vlm Qwen3.5)
          [optional h2o_front.py :8742: API key + a single-question /v1/decisions shape]
```

## Results (MacBook Pro M5 Max, 128 GB)

JevBench public set (231 items), JevBench's own CLI at commit `bb05a33`, `--adapter typesafe`, the shipped `serve_config.json`:

| Checkpoint | Size | JevBench | easy / standard / hard | Same answer as bf16 | Probability drift vs bf16 (mean / p95) |
|---|---|---|---|---|---|
| model card (vLLM, NVIDIA) | 9.1 GB | **205 / 231** | 48/48 · 71/72 · 86/111 | — | — |
| bf16, official weights (MLX) | 9.1 GB | 204 / 231 | 48/48 · 71/72 · 85/111 | — | — |
| MLX 8-bit (g64) | 4.8 GB | **205 / 231** | 48/48 · 71/72 · 86/111 | 229 / 231 | 0.006 / 0.019 |
| MLX 4-bit (g64) | 2.9 GB | 207 / 231 | 48/48 · 72/72 · 87/111 | 220 / 231 | 0.039 / 0.179 |

- **bf16 vs the card:** the one-item gap is `hard-sol-c-judge_hard-13`. It is one of the five near-tie items the model card names as sensitive to GPU numerics. The card's four expected near-tie misses miss here too.
- **8-bit** is effectively lossless. It changes 2 of 231 answers, both on near-ties, and halves the memory. **This is the recommended build.**
- **4-bit** scores higher here, but that is noise. 11 answers flip in both directions, and the probabilities drift by up to 0.18 at p95. The model's selling point is calibration, so use 4-bit only when memory matters more than calibrated confidence.

Parity against a PyTorch reference (the same safetensors through `transformers`, float32 head) on 20 JevBench prompts:
- The top answer agreed on 20 of 20. The maximum probability difference was 0.016 (mean 0.003).
- On the receipt image, the image tokens are identical (481 tokens, grid 1×42×36), and the probability difference is 0.001.

Latency is for one 2,000-token record and one question, serial. The run is prefill-bound, so quantization barely changes it:

| | bf16 | 8-bit | 4-bit |
|---|---|---|---|
| read-out, p50 | 0.43 s | 0.42 s | 0.36 s |
| image question (receipt) | 0.24–0.45 s | 0.24 s | 0.19 s |
| MLX memory, active / peak | 8.5 / 10.0 GB | 4.8 / 6.3 GB | 2.8 / 4.5 GB |

Items around 650 tokens (the JevBench average) take about 0.06 s. These timings were measured on a machine that was also serving other models, so a quiet machine is somewhat faster.

## Quick start

```bash
brew install uv
git clone https://github.com/internet-projects/h2o-lightning-mlx && cd h2o-lightning-mlx
./setup.sh q8            # venv + official H2O repo at v1.2.3 (shim, config, weights) + an 8-bit conversion; runs test_shim.py
MODEL=mlx-q8 ./serve.sh  # or ./serve.sh for the official bf16 weights
```

```bash
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Ticket 4471: checkout returns HTTP 500 for every customer since 09:10.",
  "questions": {"priority": {"type": "choice", "instructions": "Which priority?",
     "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}}}}'
```

The API, image input, status codes and routing thresholds are exactly as on the [model card](https://huggingface.co/h2oai/h2o-lightning-4b). For production routing, start with `SHIM_NOUL_FLOOR=0` so yes/no answers keep their raw probability (the card's "routing mode").

### Optional front door
`h2o_front.py` adds an API key and a `/v1/decisions` route for clients that send one question at a time:

```bash
H2O_API_KEY=… FRONT_PORT=8742 ./serve.sh
```

The request is `{"state", "question": {...}}`. The response is `{type, choice|noul|score, probabilities, confidence, input_tokens, latency_ms}`. Set `H2O_DECISION_LOG=<dir>` to keep one JSON-lines file per day of answer statistics (type, option count, top probability; never the record text). `launchd/` has an example LaunchAgent.

### Is the confidence usable?
VitaminC test split, 400 items (200 supported, 133 refuted, 67 not enough info), with the evidence sentence embedded in about 650 words of other text. H2O only, all three MLX builds:

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
git clone https://github.com/fstandhartinger/jevbench && (cd jevbench && git checkout bb05a33)
scripts/bench_jevbench.sh model bf16          # or: scripts/bench_jevbench.sh mlx-q8 q8
.venv/bin/python scripts/compare_runs.py bench/jevbench-bf16-* bench/jevbench-q8-*
uv pip install --python .venv/bin/python torch torchvision
.venv/bin/python scripts/parity_check.py --model model --jevbench jevbench --n 20
.venv/bin/python scripts/quant_check.py model mlx-q8 mlx-q4
```

## Notes and limits
- **Calibration is what H2O is for.** JevBench measures accuracy. Confidence values are only useful if you check them on your own data.
- **Serialized.** One MLX lock serves one forward pass at a time. That suits a single Mac, not high concurrency.
- **Long prompts.** They are pre-filled in 4,096-token chunks. Chunking changes log-probabilities by at most a few hundredths (bf16 arithmetic).
- **Idle release.** The weights load on first use and are released after `FACADE_IDLE_UNLOAD_S` idle seconds (0 keeps them resident).
- **Tested versions.** mlx 0.32.3, mlx-vlm 0.7.6, transformers 5.19.0, macOS 26.7.
- **Scope.** This repo is the decision model and its serving path. It does not include any product harness.

## Credits and license
- Model: [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) by H2O.ai (Apache-2.0), built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache-2.0). The shim and `serve_config.json` are H2O.ai's and are downloaded from their repo, not redistributed here.
- Runtime: [MLX](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
- Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench).
- This repo's code: Apache-2.0, © NetProjects (Internet Projects Ltd). It is not affiliated with H2O.ai.
