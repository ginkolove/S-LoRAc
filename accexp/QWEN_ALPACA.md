# Qwen2.5-7B-Instruct / Original Alpaca

The standalone launcher `run_qwen_alpaca.py` reads the local original
`tatsu-lab/alpaca` train parquet files and the rank-8 adapter
`srai86825/qwen2.5-7b-alpaca-full-lora`. It does not modify the existing
GSM8K/CNN launchers, shared evaluator, or S-LoRA performance implementation.

## Protocol

- Use the Alpaca instruction/input/response templates, consistent with the
  adapter's published Axolotl `type: alpaca` configuration. Do not add chat
  wrapping or another BOS token. The optional precombined `text` column is
  deliberately unused, to keep current reference answers out of inputs.
- Seed 42: select three complete demonstrations fitting a 1024-token budget.
  Remove those questions and duplicate questions from the evaluation pool.
  Reject overlength requests without truncation, then uniformly sample 1000
  eligible questions without replacement. `--sample-size 0` selects all eligible
  questions. Preparation scans the whole dataset; this is not the GPU sample count.
- Default maximum context is 2048, including up to 256 generated tokens, matching
  the adapter's published training sequence length. Model dimensions and the
  native context ceiling are read from its configuration. All three modes use
  exactly the same saved inputs, references, generation budget and greedy decoding.
- On one GPU, run Base first, then LoRA Full, then MiniKV. Base and Full each
  prefill the fixed prefix once for their immutable cache template. Separate
  full-vs-split consistency-check forwards validate cache injection.
- MiniKV independently prefills a fresh BaseKV with the adapter disabled,
  prefills the adapted prefix to capture the K/V LoRA-A outputs, discards the
  native adapted prefix cache, and reconstructs from BaseKV and MiniKV using
  LoRA-B, scaling and RoPE. Reconstruction is approximate across layers. Reuse
  the resulting prefix template across queries; no Full cache is substituted.
- Native Transformers `generate` and PEFT perform inference. The default FP32
  policy and strict consistency tolerances are the same as the other Qwen runs.
- Primary metric: generic normalized English token F1. Also report normalized
  EM and ROUGE-L F1 (stemming enabled). These are lexical overlap metrics, not
  an official Alpaca correctness benchmark. All scores are fractions in [0, 1].
- Apply the same reference-independent first-answer extraction to all modes:
  stop only at an explicit next `### Instruction:` outside fenced/indented code.
  Save original text and raw scores too. This rule can cut legitimate reserved
  headings and does not remove unmarked continuation; both limitations are recorded.

Original Alpaca has no official held-out test split. The adapter reports training
on Alpaca with `val_set_size: 0.0`; removing few-shot demonstrations does not make
these samples unseen during adapter training. Use this experiment to assess
MiniKV quality retention, not to claim held-out generalization.

## Run On Physical GPU 0

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="accexp/results/qwen25_7b_alpaca_gpu0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 nohup /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_qwen_alpaca.py \
  --model /data/hub/qwen2.5-7b-instruct \
  --lora /data/hub/alpaca-lora \
  --dataset-dir /data/hub/alpaca \
  --sample-size 1000 --num-shots 3 --seed 42 \
  --prefix-max-tokens 1024 --max-context-len 2048 --max-new-tokens 256 \
  --batch-size 20 --dtype float32 --device cuda:0 \
  --output-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 &
tail -f "$RUN_DIR/run.log"
```

`result.json` contains final filtered/raw scores, every prediction, extraction
diagnostics, cache checks and the preparation manifest. `samples.jsonl` and its
`.meta.json` record sample IDs, fixed demonstrations, token counts and hashes.
`result.partial.json` and `result.predictions.jsonl` preserve batch progress;
`status.json` records the launcher state. An existing run directory is never
silently overwritten. `--prepare-only` validates the adapter and builds inputs
without loading the base model weights or running GPU inference.
