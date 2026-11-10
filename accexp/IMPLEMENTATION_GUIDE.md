# MiniKV Accuracy Implementation Guide

This is the preserved implementation/protocol reference. For current local
model/adapter locations, runnable commands and the completed-result index,
start with [README.md](README.md). Historical examples below can contain old
adapter locations or sample-count defaults; the experiment catalog takes priority.

`eval_minikv.py` is independent of S-LoRA. It uses PyTorch, Transformers and
PEFT for three runs on the same prepared test samples:

1. `base`: original base model prefix cache, query prefill and greedy decode.
2. `lora_full`: real PEFT adapter prefix cache, query prefill and greedy decode.
3. `lora_minikv`: reconstructed prefix cache, then the same PEFT adapter for
   query prefill and greedy decode.

The single implementation file currently supports Transformers' LLaMA
(including LLaMA 2) and Mistral structures, GQA, Q/V-only adapters, and standard
additive LoRA on attention and MLP projections. Model sizes and each projection's
rank/scaling come from config and actual modules. DoRA and adapters that modify
base biases are rejected. No dummy weights or synthetic KV are used.

## Environment

Validated locally with Transformers 4.57.3, PEFT 0.18.1, PyTorch 2.11.0 and tqdm.
Use the `transformer` environment, not the old S-LoRA environment:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/eval_minikv.py --help
```

## Dataset Interface

Prepare UTF-8 JSONL externally. Each record has this shape:

```json
{"id":"example-1","prefix":"Shared document and instructions\n","query":"Question: ...\nAnswer:","references":["Expected answer"],"metadata":{"task":"qa"}}
```

`id` and `metadata` are optional; other fields are required. Prefix and query
must be nonempty. References are used only for scoring, never for cache building
or model inputs. Include any chat/instruction formatting in the prepared text;
the script does not guess a prompt template. Use `--no-add-special-tokens` if
your formatted text already includes the required special tokens.

The full text is tokenized once. The script finds its exact common token prefix
with the separately encoded prefix, so a BPE boundary merge cannot change the
logical input. The resulting actual prefix/query token counts are saved.
Overlength records fail explicitly; there is no silent truncation or context
extension. Future dataset processors only need to produce this JSONL interface.

## Train Few-Shot and Test Readers

`data/guanaco.py`, `data/alpaca_cleaned.py` and `data/alpaca_mt.py` each read their
own raw dataset. `eval.py` is an equivalent entry point to `eval_minikv.py` and
can receive prepared JSONL with `--data` or invoke a reader with `--dataset`.

The readers choose `--num-shots` complete question/answer demonstrations from
train using `--seed`, concatenate them into one fixed prefix, and pair it with
each test question and its reference answer. Test answers are never inserted
into the prefix or query. The prefix is identical for all selected test samples.
Malformed/duplicate records and train examples overlapping test questions are
excluded. Guanaco sibling conversation paths are grouped by their first question.

Guanaco uses its native train/eval files. For Alpaca-Cleaned and HiTZ Alpaca MT,
the default selects the fixed few-shot examples from the whole cleaned dataset
and uses all remaining examples as test queries. Demonstrations and equivalent
question/conversation groups are excluded from the test pool. No `--test-file`
or `--holdout-fraction` is required. A separate test file or an explicit holdout
fraction remains available as an optional override. This evaluation split does
not establish that an already trained public LoRA never saw those samples.
The split source and that limitation are recorded in the output.

`--prefix-max-tokens` (default 1024) is a cap, not a target length. Whole train
demonstrations are selected without truncation. The reader then filters out test
queries whose full tokenized input plus `--max-new-tokens` exceeds the model
context limit. No test question or reference answer is shortened. With the local
LLaMA 7B/13B/30B configs and output budget 256, the input cap is 1792 tokens.
The selection report includes demonstration IDs, prefix hash, filtering counts
and model/tokenizer budgets. `--max-samples` limits eligible test samples.

Direct evaluation with a matching cached Alpaca adapter:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python \
  accexp/eval.py \
  --model huggyllama/llama-7b \
  --lora tloen/alpaca-lora-7b \
  --dataset alpaca_cleaned \
  --num-shots 3 --prefix-max-tokens 1024 \
  --max-new-tokens 256 --max-samples 10 \
  --dtype float16 --seed 42 \
  --output accexp/results/llama7b_alpaca_fewshot.json
```

Or export Guanaco data first, without loading model weights:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/data/guanaco.py \
  --model huggyllama/llama-7b \
  --num-shots 3 --prefix-max-tokens 1024 \
  --max-new-tokens 256 --seed 42 \
  --output accexp/results/guanaco_fewshot.jsonl
```

Export writes a JSONL file plus a `.meta.json` report. Supply that JSONL to
`eval.py --data` with the same output-token and special-token settings. Other
readers expose the same standalone arguments; HiTZ additionally accepts
`--language`. All readers accept local raw `--train-file` and `--test-file` JSON
arrays or JSONL, so evaluation also works without network access.

Online sources are the Guanaco and Alpaca-Cleaned Hub repositories, and the
JSON URLs used by [HiTZ's original loader](https://huggingface.co/datasets/HiTZ/alpaca_mt/blob/main/alpaca_mt.py).
HiTZ loading does not execute the old remote dataset script. Non-English prompt
templates are read as literal data from that script. If Hub connectivity requires
a mirror, set `HF_ENDPOINT` before launching; local input files avoid downloads.

Reader and end-to-end entry-point tests:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/test_data.py -v
```

## Run

For an already prepared test set:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python \
  accexp/eval_minikv.py \
  --model huggyllama/llama-7b \
  --lora tloen/alpaca-lora-7b \
  --data /path/to/prepared_test.jsonl \
  --output accexp/results/llama7b_alpaca.json \
  --dtype float16 \
  --max-new-tokens 64 \
  --seed 42
```

Change the model and its matching adapter paths for other checkpoints. Use
`--device-map auto` with the intended GPUs visible for a model spanning GPUs.
The default batch size is 4. Set `--batch-size` explicitly to fit available memory;
`--base-batch-size`, `--lora-batch-size` and `--minikv-batch-size` optionally
override it for Base, full LoRA and MiniKV respectively. MiniKV uses the configured
batch size independently on each GPU. Requests are
sorted by query length within each token-identical prefix group, while output
records retain their original dataset order. Sliding-window models can batch
different query lengths: query prefill uses right padding and each request's
last valid logits. Before decode, the working KV is right-aligned once so that
padding precedes the complete prefix/query history instead of splitting it.
Logical RoPE positions and the native sliding-window mask are preserved.

To evaluate one model size on two GPUs concurrently:

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/eval.py \
  --model huggyllama/llama-7b \
  --lora /data/hub/alpaca_cleaned-lora/7b \
  --data accexp/results/alpaca_cleaned_3shot_all_seed42.jsonl \
  --device cuda:0 --lora-device cuda:1 \
  --batch-size 8 --minikv-batch-size 8 \
  --weight-format bin --dtype float16 --max-new-tokens 256 \
  --max-samples 0 --seed 42 --local-files-only \
  --output accexp/results/alpaca_cleaned_7b_staged_seed42.json
```

GPU 0 retains the base model and its immutable prefix templates. GPU 1 loads a
separate model with the adapter. One adapted prefix forward both produces the
full LoRA KV and captures MiniKV. During preparation, BaseKV is copied once per
prefix to GPU 1, and MiniKV (including its RoPE metadata) is copied once to GPU 0.
Transfers finish before the receiving worker uses or the sending worker releases
these templates. GPU 0 then evaluates all Base queries; GPU 1 independently
evaluates all full-LoRA queries, releasing full prefix templates as they finish.

Each GPU joins a shared MiniKV queue immediately after its own reference run.
GPU 0 attaches the same adapter to its existing model only after all Base samples
finish, and only if it can claim pending MiniKV work. There is no barrier waiting
for both reference runs to finish. Each worker reconstructs each prefix it needs
once locally, releases the corresponding source BaseKV/MiniKV, and receives
disjoint batches from the queue. Fast workers can process more samples; together
they evaluate each MiniKV sample exactly once. Results retain dataset order.

Omit `--lora-device` to run Base, full LoRA and MiniKV sequentially on one model.
Dual-device mode cannot be combined with `--device-map auto`. Worker errors cancel
the other worker at the next batch or cache wait; partial results are not scored.

## Llama-2 CoQA

`data/coqa.py` prepares one common JSONL for both local Llama-2 models. Defaults:

- Models: `/data/models_2/llama-2-7b-hf` and `/data/hub/Llama2-13b-hf`.
- Data: local `/data/hub/coqa` train and validation Parquet files.
- Fixed prefix: three complete train passages, each with its first two Q/A turns,
  selected with seed 42 under a 1536-token prefix cap for both tokenizers.
- Evaluation: 1000 random validation **turns**, not 1000 conversations. Use
  `--sample-size 0` to evaluate all eligible turns.
- Query: the full current passage, all previous primary gold Q/A turns, then
  the current question. Current/future answers are never supplied. This is
  gold-history turn evaluation, not a generated multi-turn conversation rollout.
- Input plus output is checked against both models' context limits (4096 locally).
  Overlength turns are excluded, never truncated. Default output budget is 64.
- Only train demonstrations are in the shared cache; the current evidence passage
  and conversation history remain in the query. This does not measure accuracy
  when the current evidence document itself is approximated as a cached prefix.

The local HF Parquet omits official conversation IDs and the extra reference
answers. Download the original public dev JSON once with `--download-gold`, or
provide it with `--gold-file`. The download checksum is verified against the
published CoQA dataset checksum. Local conversations are matched by the full
passage, questions and primary answers, not just passage text: different
conversations can share a passage. Repeated annotator answers are retained.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/data/coqa.py \
  --download-gold --sample-size 1000 --seed 42 \
  --output accexp/results/coqa_shared_seed42.jsonl

CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_llama2_coqa.py --size 7b \
  --data accexp/results/coqa_shared_seed42.jsonl \
  --output-dir accexp/results/coqa_llama2_7b
```

Change `--size 7b` to `--size 13b` and use a new output directory for the other
model. The runner resolves the corresponding adapter under
`/data/hub/coqa-lora/llama2-{size}`. It uses safetensors for 7B, bin for 13B, and
batch sizes 20 and 8 respectively. `--batch-size`, `--model`, `--lora`, and
`--weight-format` can be overridden. Use the same prepared file for both models.
The runner verifies its checksum, model validation entry, IDs and output budget.

Each run is single-device, sequential Base -> LoRA Full -> LoRA MiniKV using
`--minikv-independent-prefill`. The third stage prepares fresh BaseKV with LoRA
disabled, captures real MiniKV with LoRA enabled, reconstructs the shared prefix
once, and evaluates every query with the adapter still enabled. The shared
inference implementation and other dataset launchers are unchanged.

`scores` contains `exact_match` and `coqa_f1` fractions. The implementation follows
the [official CoQA scorer](https://downloads.cs.stanford.edu/nlp/data/coqa/evaluate-v1.0.py):
leave one annotator out, take the maximum over the others, average these scores,
then average across evaluated turns. It is not a simple maximum over references.
Every method uses the same reference-independent rule to cut only explicit
line-start next-question/passage markers. Full raw generations, extracted answers,
`unfiltered_scores`, EOS/length-limit counts, and filtering counts are saved.
The fixed local Passage/Q/A prompt is not claimed to reproduce the public adapter's
undocumented training template or the official hidden-test leaderboard protocol.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -m unittest discover \
  -s accexp -p test_llama2_coqa.py -v
```

Templates never contain query/decode state. Each batch gets independent working
KV, including expanded prefix copies along the batch dimension; standard
Transformers DynamicCache does not provide physically shared paged prefixes.
There are no per-request CPU-to-GPU prefix transfers. Batch size is limited by
working KV and activations, especially for 30B. An OOM aborts rather than silently
dropping samples. Each completed batch is saved to `<output>.predictions.jsonl`
(replacing the final `.json` suffix), flushed and fsynced before its progress
counter advances. Each line contains one batch with its mode, original sample
indices, IDs, references and generated answers/token IDs. An interrupted final
line can be discarded; previously completed lines remain usable.

`<output>.partial.json` is replaced atomically after every saved batch. It reports
per-mode completed counts and cumulative scores; modes without predictions have
null scores. These per-mode subsets can differ while evaluation is running.
`paired_lora` compares full LoRA and MiniKV only on the intersection of their
completed sample IDs. Partial scores are not the final full-dataset result.
The managed launcher marks a terminated child's partial summary as interrupted.
The final result is saved when all three modes finish. Incremental files preserve
completed work for analysis, but automatic resume is not implemented. Existing
prediction journals are never overwritten; use a new output path for a fresh run.

For a reproducible random 1,000-sample suite across 7B, 13B and 30B:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -u accexp/run_alpaca_cleaned.py \
  --sample-size 1000 --seed 42 \
  --output-dir accexp/results/alpaca_cleaned_random1000_run
```

The launcher samples without replacement from the prepared 51,742-sample pool,
preserving each complete prefix/query/reference record. It writes the selected
JSONL and a metadata file with source/output hashes and selected indices/IDs.
All three models use this exact same file. The default suite sample size is 1000;
`eval.py --max-samples N` alone still selects the first N, not random examples.
Batch sizes are 8/4/2 for 7B/13B/30B. GPU memory admission checks wait when needed;
the suite stops on a failed model and preserves its incremental outputs.

Use `--max-samples 10` for a short run. All modes use greedy argmax without
sampling or logits processors, stopping at EOS or the same output-token limit.
Three progress bars separately count Base, full LoRA and MiniKV, each with the
same total number of samples. Updates occur after a batch completes. The MiniKV
counter combines both workers. `output_tokens` is the batch's total generated
tokens, and `last_s` is its wall-clock time. Redirected logs contain ordinary
text lines at stage changes, at completion and approximately every 30 seconds,
instead of terminal cursor sequences. ETA remains approximate because answer
lengths and the number of GPUs currently serving MiniKV can change.
Use `--weight-format bin` when base-model `.bin` shards are complete but a cached
safetensors index refers to missing shards; this selects the existing PyTorch
weights without modifying the Hugging Face cache. The default is `auto`.

`example.jsonl` contains two short requests for a pipeline smoke test; its scores
are not a dataset accuracy result. Use it as `--data accexp/example.jsonl` when
checking a local installation, then replace it with the prepared held-out data.

The output JSON includes configuration, library versions, model/adapter
configuration, actual token lengths, all generated token IDs/text, scores,
signed score differences, and prefix-cache consistency checks. `experiment.workers`
records devices, per-worker MiniKV completion counts, received tensor bytes and
synchronized wall-clock durations for preparation, transfer, generation and
reconstruction. `minikv_stage` includes reconstruction and generation; do not add
it to its component durations. Worker stages overlap across devices, so their
durations do not sum to experiment elapsed time. `total_wall_seconds` includes
initial data/model loading and scoring, but excludes final JSON serialization;
`elapsed_seconds` starts after initial Base model loading. Runtime errors abort the experiment rather
than quietly comparing different subsets of samples. Same-prefix requests reuse
immutable device-resident templates; each batch receives fresh working caches.

The default scorer reports generic English normalized `exact_match` and
`generic_token_f1`, both fractions in [0, 1]. This is not a dataset's official
scorer and is not appropriate for every task/language. Supply the dataset's
scorer as `--scorer /path/to/scorer.py:score` or `--scorer package.module:score`:

```python
def score(predictions, references, samples):
    # predictions: list[str], references: list[list[str]], samples: original records
    return {"task_metric": ...}
```

## Non-English Alpaca-MT With One English Prefix

`data/prepare_alpaca_mt.py` reuses the existing English MT 3-shot prefix from
`results/alpaca_mt_en_3shot_all_seed42.jsonl`, verifying its text against its
English demonstration IDs. It samples 200 tasks without replacement with seed
42, then uses their corresponding `pt/es/ca/eu/gl/at` translations: 1,200 samples
per model and per evaluation mode. The original translation builder preserves
row order. Raw files are pinned to an upstream revision and verified against
their Git blob hashes before row indices are used for alignment.

The English demonstrations and their translated/equivalent questions are
excluded. Invalid/duplicate questions and inputs exceeding a model's context
budget are filtered before sampling. Sampling uses the common eligible task
intersection across all requested languages and models, so samples stay paired.
Native language prompt templates are read from the local loader without executing
it. Each query additionally asks for an answer in its target language; reference
answers are never included in model input. This is a multilingual robustness
experiment on translated tasks, not six independent task datasets or an official
held-out test. Public adapter training overlap remains unverified.

Prepare the data on CPU (downloads only raw JSON files; cached model configs and
tokenizers are loaded locally, without model weights):

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -u accexp/data/prepare_alpaca_mt.py \
  --samples-per-language 200 --seed 42 --max-new-tokens 256 \
  --output accexp/results/alpaca_mt_non_en_english_prefix_200_seed42.jsonl
```

The output's `.meta.json` saves selected source indices, language filtering
counts, raw/output/prefix hashes, and model-specific context/prefix lengths.
Existing prepared outputs are not overwritten. `--local-files-only` also
disables raw data downloads; `--raw-dir` changes their workspace cache directory.

Run 7B, 13B and 30B sequentially on physical GPUs 0 and 1, using the existing
two-GPU staged Base/full-LoRA/MiniKV implementation and batch sizes 8/4/2:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -u accexp/run_alpaca_mt.py \
  --data accexp/results/alpaca_mt_non_en_english_prefix_200_seed42.jsonl \
  --output-dir accexp/results/alpaca_mt_non_en_run
```

The launcher uses `/data/hub/alpaca_mt-lora/{7b,13b,30b}` (not the Cleaned
adapters), waits for GPU memory when needed, and loads each model once per GPU.
It validates data hashes/alignment before starting and stops on a failed model.
`--sizes 7b` optionally runs only that size. Use a new output directory for each
run. Status, per-model logs, incremental scores/predictions, final results, and
`summary.json` are saved there. The summary includes per-language MiniKV-minus-full
score differences.

After every requested model has finished, the launcher automatically applies
`rescore_turn_boundaries.py` to all three modes with the same
`alpaca_turn_boundary_v1` policy. It truncates at the first complete, standalone
instruction header from the official English or translated templates, also
removing a directly preceding exact template preamble when present. Fenced and
indented code and blockquoted headers are protected. The extraction never uses
reference answers, and ordinary paragraphs are retained. Unmarked continuation
and modified template preambles may remain; reserved headers can also be valid
answer content, so inspect the saved audit examples.

Original per-model results and the existing `scores` in `summary.json` retain
their original meaning. The suite summary additionally includes
`filtered_scores`, `filtered_score_deltas`, per-language filtered differences,
and `filter_counts`. The `turn_boundary_v1/` subdirectory contains the full
filtering report, raw/extracted text, cut positions, and seeded audit examples.
Status changes to `filtering` during this CPU-only step and to `complete` only
after it succeeds. No extra flag is required. `--template-loader` overrides the
default `/data/hub/alpaca_mt/alpaca_mt.py`; its hash must match the prepared-data
metadata when recorded. This automatic step belongs to `run_alpaca_mt.py`;
calling the generic `eval.py` directly still produces the original outputs.

`multilingual_score.py:score` uses Unicode NFKC/case-fold normalization and
punctuation-to-space conversion, preserving accents and all articles. It reports
generic EM/whitespace-token F1 overall, separately by language, and as a language
macro average. These are lexical overlap measures, not official Alpaca quality
metrics or a language-identification test. The old English scorer is unchanged;
do not directly compare its values against this new normalization. Language
compliance should also be inspected in the saved generated answers.

## Guanaco 65B With Tensor Parallelism

`eval_tp.py` uses native Transformers tensor parallelism on two GPUs. It is
separate from the existing single-GPU and two-replica evaluators. Base weights
are loaded once with `tp_plan="auto"`; both ranks jointly execute every batch.
The stages run in order: Base prefix/cache evaluation, full LoRA prefix/cache
evaluation, then MiniKV evaluation. Before the last stage, the adapted model
disables LoRA to obtain fresh native BaseKV, enables LoRA to capture MiniKV,
discards that forward's full KV, and reconstructs the prefix. Each stage retains
only its prefix template and independent per-request caches. Native Base/full
prefix injection is checked against a full-input forward once per stage.

LoRA uses the actual PEFT checkpoint and scaling. Column-parallel projections
replicate A and shard B's output rows. Row-parallel projections shard A's input
columns and sum its partial rank-space activations before applying replicated B.
Cached K/V retain only the local attention heads; the existing RoPE-aware MiniKV
reconstruction runs against these local heads. Adapter parameters retain PEFT's
FP32 promotion, as in the earlier accuracy evaluator.

For comparison with the completed 7B/13B/30B Guanaco runs, reuse
`results/guanaco_local_3shot_random100_seed42.jsonl`: the exact same 100 samples,
fixed 3-shot prefix (981 tokens with the original Llama tokenizer), seed 42,
256 output-token cap, greedy decoding, and the existing English EM/token F1.
Guanaco outputs do not receive the Alpaca-MT boundary filter. The JSONL was
prepared from `/data/hub/openassistant-guanaco`; no new sample selection is
performed. `--max-samples N` selects the first N prepared rows for a smoke test.

Check local downloads without loading model weights or reserving GPUs:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/eval_tp.py \
  --model /data/hub/Llama-65B --lora /data/hub/guanaco-lora/65b \
  --output accexp/results/guanaco_65b_tp2_random100_seed42.json --check-only
```

The tokenizer defaults to the local 65B directory. `--tokenizer` can override
it when a separate matching original-Llama tokenizer is required. Downloaded
weight files with `.aria2` or `.incomplete` markers are refused. Complete one
base checkpoint format and one adapter format; there is no need to download
both formats for this experiment. The preflight checks file presence/download
markers, configuration and context limits, not whole-file cryptographic integrity.
For safetensors it checks all tensor names and shapes against the base/adapter
configs and the shard index, including original Llama's optional persisted RoPE
buffers. It also verifies the prepared JSONL against its saved sampling hash.

Run the evaluation after preflight passes:

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=2 accexp/eval_tp.py \
  --model /data/hub/Llama-65B --lora /data/hub/guanaco-lora/65b \
  --data accexp/results/guanaco_local_3shot_random100_seed42.jsonl \
  --model-weight-format safetensors --dtype float16 --batch-size 1 \
  --max-new-tokens 256 --seed 42 \
  --output accexp/results/guanaco_65b_tp2_random100_seed42.json
```

Only rank 0 writes the final result, `.partial.json` and `.predictions.jsonl`.
Progress is printed periodically to redirected logs. Result metadata records
the data hash, TP projection/LoRA layouts, local prefix storage and per-rank
memory peaks. Use a fresh output path for every run. To exercise numerical
equivalence and the full launcher with tiny models on real GPUs:

```bash
RUN_TP_SMOKE=1 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=accexp \
  /home/qlh/anaconda3/envs/transformer/bin/python -m unittest \
  test_eval_tp.TPTests.test_two_rank_numerical_equivalence -v
```

Set `TP_TEST_DTYPE=float16` to exercise half precision; the default is FP32.

## Alpaca-Cleaned 65B With TP=2

`run_alpaca_cleaned_65b.py` reuses the same `eval_tp.py` inference implementation.
Normal execution checks free GPU memory, starts two TP ranks and loads the
model on both GPUs before reading or tokenizing dataset examples. After both
ranks finish loading, rank 0 prepares the fixed prefix and sampled JSONL once;
both ranks then read the same prepared data and verify matching token inputs.
Invoke this launcher with plain Python, not with an additional outer
`torchrun`. Base evaluates every selected query first, then full LoRA, and
finally MiniKV. Each stage prepares its prefix once and reuses it. Both GPUs
jointly execute the current stage. `--prepare-only` remains a CPU-only
data preparation and checkpoint validation mode without loading the model.

The defaults are `/data/hub/Llama-65B`, raw data from
`/data/hub/alpaca_cleaned/alpaca_data_cleaned.json`, and the real adapter at
`/data/hub/alpaca_cleaned-lora/65b`. The adapter is currently a rank-16 Q/K/V/O
LoRA in `adapter_model.bin`; its rank, alpha and targets are read from its
configuration, not copied from Guanaco. Binary adapters are read on CPU with
`weights_only=True` during preflight to validate all names and shapes before
model loading. Its config names `elinas/llama-65b-hf-transformers-4.29` as the
training base. Dimensions are checked against the selected local Llama-65B;
this does not independently verify checkpoint weight identity or training-data
disjointness. The original adapter config is saved in the result.

Preparation uses the existing Alpaca-Cleaned formatter, deduplication and
few-shot selection. Defaults match the earlier 7B/13B/30B experiments: seed 42,
three demonstrations, a prefix cap of 1024 tokens, a 256-token output budget,
and 1000 uniformly sampled eligible requests without replacement in source
order. Demonstrations are excluded from testing. The tokenizer and context
limit come from the selected model. Overlength inputs are excluded without
truncating queries or references. With the current local originals, the fixed
prefix is 654 tokens. The JSONL manifest saves raw/output/prefix hashes, the
sampling indices, demonstration IDs and context filtering counts.
`--sample-size 0` evaluates all eligible examples instead.

This run keeps the earlier English normalized EM and token F1 on the full
generated response; no Alpaca-MT continuation filter is applied. These are
lexical overlap metrics, not an official semantic accuracy measure for Alpaca.

Prepare and preflight only, without allocating GPU model memory:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_alpaca_cleaned_65b.py --prepare-only \
  --output-dir accexp/results/alpaca_cleaned_65b_preflight
```

Run with both GPUs free, in a fresh directory:

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="accexp/results/alpaca_cleaned_65b_tp2_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
TORCHELASTIC_SIGNALS_TO_HANDLE=SIGTERM,SIGINT,SIGQUIT \
nohup /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_alpaca_cleaned_65b.py \
  --model /data/hub/Llama-65B \
  --lora /data/hub/alpaca_cleaned-lora/65b \
  --dataset-dir /data/hub/alpaca_cleaned \
  --sample-size 1000 --batch-size 3 --max-new-tokens 256 --seed 42 \
  --output-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
PID=$!
disown "$PID"
printf 'PID: %s\nResults: %s\n' "$PID" "$RUN_DIR"
```

The launcher excludes `SIGHUP` from Elastic's handlers so that it preserves
`nohup`'s inherited ignore setting. This does not protect against OOM, forced
termination or host shutdown. Immediately before starting TP, it probes the two
CUDA-visible GPUs in a separate process and requires at least 76 GiB free on
each by default (`--min-free-gpu-gib`). This avoids loading half a model alongside
another GPU workload. The probe respects CUDA's device mapping and retains no
CUDA context in the parent. It is a point-in-time check, not a reservation;
other jobs must still be kept off these GPUs during the experiment. A failed
rank prints its original exception and exits without a potentially blocking
NCCL teardown, allowing torchrun to terminate the peer promptly. Normal runs
still synchronize and destroy the process group gracefully. The communication
timeout remains unchanged; extending it would not fix a peer's model-load OOM.
Progress is in `run.log`; `status.json` records
preparation/launch/completion; `result.partial.json` and
`result.predictions.jsonl` contain incremental scores and answers. Final
results are in `result.json`. Existing output directories are not resumed or
overwritten. Preparation can also be used independently through
`data/prepare_alpaca_cleaned.py --output <prepared.jsonl>`.
Normal startup logs `loading_model`, `model_loaded`, `preparing_data`, then
`data_loaded`, in that order. Until data is loaded, the partial summary has
zero known samples; it still records any model-loading failure immediately.
The final report records `startup_order: model_then_data`.

CPU tests and the optional tiny-model, binary-adapter two-GPU smoke test:

```bash
PYTHONPATH=accexp /home/qlh/anaconda3/envs/transformer/bin/python -m unittest \
  test_alpaca_cleaned_tp -v
RUN_TP_SMOKE=1 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=accexp \
  /home/qlh/anaconda3/envs/transformer/bin/python -m unittest \
  test_alpaca_cleaned_tp.AlpacaTPTests.test_tiny_two_gpu_binary_adapter_all_modes -v
```

## Alpaca-MT 65B With TP=2

`run_alpaca_mt_65b.py` evaluates Base, full LoRA and MiniKV sequentially using
one native TP=2 model. Normal runs load the model on both GPUs first, then
rank 0 prepares the data; both ranks read the same saved JSONL afterwards.
The existing smaller-model MT launcher and its defaults are unchanged.

The default adapter is `/data/hub/alpaca_mt-lora/65b`, trained for English,
Portuguese, Spanish and Catalan. Its local config specifies rank 8, alpha 16,
and Q/V targets; these are validated and loaded from `adapter_model.bin`.
Missing K adapters contribute no direct K delta during MiniKV reconstruction;
this does not make the approximation exact across layers.

This reuses the earlier MT protocol and English three-shot prefix (249 tokens
with the local Llama tokenizer). Seed 42 selects 100 common eligible source
tasks, rendered in each of `pt`, `es`, and `ca`: exactly 300 requests, not 100
requests in total. Sampling is without replacement, aligned across languages.
Demonstrations and equivalent translated queries are excluded. Native-language
templates and the existing `Respond in <language>` instruction are retained.
The default output budget is 256 tokens; prefix plus query plus that budget
must fit the model's context limit. Overlength tasks are excluded before sampling.
These translations are not independent task datasets, and adapter-training
disjointness is not verified.

`/data/hub/alpaca_mt` contains the template loader. Raw English and translated
JSON files from earlier runs are reused, with pinned source hashes, from
`accexp/results/alpaca_mt_en_raw.json` and `accexp/results/alpaca_mt_raw/`.
The English prefix is read from
`accexp/results/alpaca_mt_en_3shot_all_seed42.jsonl`. Override `--english-file`,
`--raw-dir` and `--prefix-data` to use different locations. Preparation never
downloads missing files. `--prepare-only` prepares and validates on CPU without
loading GPU model weights. `--data <prepared.jsonl>` reuses a validated MT
dataset instead of preparing another one; its language/count/seed/output/model
and template settings must match the run.

Example, after both GPUs are free:

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="accexp/results/alpaca_mt_65b_pt_es_ca_100_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
TORCHELASTIC_SIGNALS_TO_HANDLE=SIGTERM,SIGINT,SIGQUIT \
nohup /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_alpaca_mt_65b.py \
  --model /data/hub/Llama-65B --lora /data/hub/alpaca_mt-lora/65b \
  --dataset-dir /data/hub/alpaca_mt --languages pt es ca \
  --samples-per-language 100 --batch-size 6 --max-new-tokens 256 --seed 42 \
  --output-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
PID=$!
disown "$PID"
```

`result.json` preserves raw predictions and language-neutral EM/token F1,
including separate scores for each language. After inference exits, all three
modes receive the same reference-independent turn-boundary extraction used
in earlier MT experiments. `summary.json` contains both `scores` (raw) and
`filtered_scores`; `turn_boundary_v1/` contains filtered predictions and an
audit sample. Raw predictions are not overwritten. This extraction does not
use reference answers or search for a correct substring. Reserved headers in
legitimate answer content can still cause a false boundary; audit examples
before treating filtered scores as a final quality claim. Metrics are generic
lexical overlap, not an official semantic Alpaca accuracy measure.

CPU checks: `PYTHONPATH=accexp python -m unittest test_alpaca_mt_tp -v`.
Set `RUN_TP_SMOKE=1 CUDA_VISIBLE_DEVICES=0,1` to include the tiny Q/V-only TP
numerical comparison and complete three-mode multilingual/filtering smoke test.

## Mistral-7B on Local GSM8K

`run_mistral_gsm8k.py` runs Base, LoRA Full, then independently prepared MiniKV
sequentially on one GPU, using the existing shared evaluator without modifying
the CNN/DailyMail or DROP workflows. Defaults are the local Mistral checkpoint,
`/data/hub/gsm8k-lora`, and `/data/hub/gsm8k/main` (7473 train, 1319 test rows).
The `socratic` configuration is separate and never mixed into `main`.

Three complete train question/solution examples form one fixed prefix. The
solution text, calculator annotations and final `####` answer are preserved in
these demonstrations; there is no calculator at inference time. Query inputs
contain only the test question, using the adapter's exact template:

```text
Please answer the following question: {question}
Answer:
```

Prepared samples use the same JSONL interface, with one final numeric reference
per question. Test solutions are kept in metadata for auditing and never enter
model inputs. Inputs are not truncated; questions exceeding the context budget
are excluded before deterministic sampling. Defaults are 1000 questions, seed
42, prefix cap 1536, output cap 256, batch size 20, and the smaller of the model
context and sliding window (4096 here). All three modes consume identical data.

The fixed policy `gsm8k_first_task_final_number_v1` applies equally to all modes:

- First remove continuation at a line-start adapter prompt, `Question:`, or
  `### Question:`. Keep the entire multiline reasoning before that boundary.
- Primary `scores[mode].exact_match` extracts the numeric prefix immediately
  after the first declared `####`, allowing currency, grouping commas, signs,
  decimals, a terminal period, and trailing units such as `3 hours`. An obvious
  expression such as `#### 2+3` is invalid, not a prediction of `2`. Finite numeric
  equality uses Decimal, not float rounding. Incorrect/malformed first markers
  cannot be replaced with a later answer or the answer to another question.
- Secondary `flexible_exact_match` uses that strict answer, or the last numeric
  token in the retained answer only when `####` is completely absent. This is
  a separately labeled diagnostic, not the primary strict metric. There is no
  reference-dependent answer selection or arithmetic-expression evaluation.

The original GSM8K evaluator's `####` convention is the reference; the decimal
normalization, local three-shot template and boundary policy are documented
protocol choices, not a claim of an identical official leaderboard setup:
https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_mistral_gsm8k.py \
  --model /data/hub/mistral-7b-v0.1 --lora /data/hub/gsm8k-lora \
  --dataset-dir /data/hub/gsm8k --dataset-config main \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 --batch-size 20 \
  --dtype float16 --weight-format bin --attn-implementation sdpa --seed 42 \
  --output-dir accexp/results/mistral_gsm8k_run1
```

`--prepare-only` checks data and exports the manifest without loading weights.
Result/journal/partial/status file names match the other Mistral launchers.
Raw predictions and generated token IDs are retained. Final `result.json` also
records per-request extraction, `unfiltered_scores` for comparison, and per-mode
counts of boundary cuts, missing final markers, flexible fallback, EOS and
output-limit termination. All scores are fractions in [0, 1]. No additional
scoring dependencies beyond the existing accuracy environment are needed.

Offline tests:

```bash
CUDA_VISIBLE_DEVICES='' /home/qlh/anaconda3/envs/transformer/bin/python \
  -m unittest discover -s accexp -p 'test_mistral_gsm8k.py'
```

## Llama-2-7B/13B on Local GSM8K

`data/llama2_gsm8k.py` prepares one common JSONL for both local Llama-2 base
models and the LEMA adapters under `/data/hub/gsm8k-lora/llama2-{7b,13b}`.
The local adapters declare ordinary `meta-llama/Llama-2-{size}-hf` bases,
rank 64, alpha 16, and all attention/FFN projections. The base tokenizer is
used consistently; the adapter tokenizer's extra `[PAD]` token is not injected
into inputs or used to change the model vocabulary.

As with the Mistral experiment, defaults are three complete train examples,
1000 random `main/test` questions, seed 42, prefix cap 1536, and output cap 256.
Both tokenizers and context limits are checked before sampling. No test solution
enters the prompt, no question/reference is truncated, and requested sample
counts cannot silently shrink. Use `--sample-size 10` for a small trial.

The prompt envelope follows the LEMA repository's published GSM8K examples:
https://github.com/microsoft/LEMA/blob/main/test_data/gsm8k_test.jsonl

```text
Please solve the following math problem.
Question: {question}
Answer: Let's think step by step.
```

Fixed demonstrations retain the train reasoning, remove redundant calculator
annotations such as `<<2+3=5>>`, and render the original final number as
`The answer is 5.`. These local train few-shots and the 256-token output cap are
explicit protocol choices, not a reproduction of LEMA's published zero-shot
evaluation or a promise to match its reported scores.

`run_llama2_gsm8k.py` runs Base -> LoRA Full -> LoRA MiniKV sequentially on one
GPU, reusing each method's fixed prefix cache. The MiniKV stage independently
prefills BaseKV and captures adapted MiniKV after both references have finished;
no reference-stage full LoRA cache is used to reconstruct MiniKV. Model size,
adapter base, data hash, sample IDs, prefix hash, numeric gold and output budget
are validated. Actual shot count and prefix budget propagate to the evaluator.

The separate `llama2_gsm8k_score.py` leaves the existing Mistral scorer unchanged.
Primary `exact_match` compares the final number after the first declared
`The answer is`, `The final answer is`, or `####`, using exact Decimal equality.
A malformed/wrong first answer is binding. Only when no declaration exists,
secondary `flexible_exact_match` falls back to the last numeric token. Both
policies cut only explicit next-question/instruction boundaries, never choose
text using the reference, and apply identically to all three methods. Raw
outputs, extracted values, unfiltered scores, EOS/length-limit diagnostics,
prediction journals and partial scores are retained.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -u accexp/data/llama2_gsm8k.py \
  --dataset-dir /data/hub/gsm8k --dataset-config main \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 --seed 42 \
  --output accexp/results/gsm8k_llama2_lema_3shot_n1000_seed42_v1.jsonl

CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_llama2_gsm8k.py --size 7b --batch-size 8 \
  --data accexp/results/gsm8k_llama2_lema_3shot_n1000_seed42_v1.jsonl \
  --output-dir accexp/results/llama2_gsm8k_7b_run1
```

For the other GPU/model, use `--size 13b --batch-size 4` and a new output
directory. 7B uses safetensors base weights, 13B bin; both formats are configurable.
Existing prepared files and results are never overwritten.

```bash
CUDA_VISIBLE_DEVICES='' /home/qlh/anaconda3/envs/transformer/bin/python \
  -m unittest discover -s accexp -p 'test*gsm8k.py' -v
```

## Qwen2.5-7B-Instruct on Local GSM8K

`run_qwen_gsm8k.py` prepares the data, then runs Base -> LoRA Full -> LoRA
MiniKV sequentially on one GPU using the shared Transformers/PEFT evaluator.
All three Qwen modes call the library's `model.generate()` with one explicit
greedy generation configuration, not the older per-token decode loop. Each
batch receives a separate `DynamicCache` copied from the immutable prefix
template; only query tokens are prefetched. The library manages attention,
position IDs, decoding, and per-request EOS stopping. Llama/Mistral experiment
entry points retain their existing generation path.
Base and Full each reuse their own fixed prefix cache. MiniKV independently
prefills with adapters disabled to obtain BaseKV, then enabled to capture K/V
LoRA-A outputs, reconstructs its approximate prefix cache, and reuses that
cache. The Full-stage KV cache is not used for MiniKV reconstruction.

The local default model is `/data/hub/qwen2.5-7b-instruct`, the adapter is
`/data/hub/gsm8k-lora/qwen2.5-7b`, and the dataset is `/data/hub/gsm8k/main`.
The adapter is rank 16, targets all seven attention/MLP projections, and declares
`Qwen/Qwen2.5-7B-Instruct` as its base. Preparation checks the adapter tensor
shapes and, when present, its tokenizer against the base tokenizer. GQA uses
the actual KV head count; base Q/K/V biases are not added a second time.

The native chat template formats one system instruction and three complete
train question/solution turns into a common cached prefix. The test question
is a new user turn, followed by the assistant generation header; no test answer
enters the input. Defaults are 1,000 test questions without replacement, seed
42, a 1,536-token prefix cap, 256 output tokens, greedy decoding, and batch 20.
Inputs are filtered, never truncated, against a conservative 4,096-token total
budget (native model context: 32,768). With the current local files this gives
a 454-token prefix, 485--650 input tokens, and the same 1,000 test IDs as the
previous Llama-2 GSM8K preparation. These are local fixed few-shot prompts,
not a reproduction of the adapter's undocumented GRPO training prompt.

FP32 is intentional for this accuracy experiment: on the real checkpoint,
FP16/BF16 full-vs-split prefill differed beyond the existing cache-check
tolerances. FP32 passed with maximum logit difference about 7.5e-5. This
launcher neither bypasses nor relaxes the shared evaluator's checks. All three
methods use the same dtype. Lower-precision CLI choices remain explicit,
but may fail that check on this checkpoint.

The primary metric is final-number `exact_match`; the separate Qwen scorer
accepts the first explicit `####`, `\\boxed{...}`, `<answer>...</answer>` or
`The answer is ...` declaration. A malformed or wrong first declaration is
binding. Secondary `flexible_exact_match` falls back to the last number only
when no declaration exists. No mode or reference is used to select output
text. Adjacent answer markers/math/Markdown wrappers are unwrapped (for
example `The answer is:\n#### 5`), without skipping a wrong/malformed answer
to find a later correct one. The policy is versioned as
`gsm8k_qwen_first_declared_final_number_v2`.
Raw output, extracted answers, unfiltered scores, output-cap/EOS counts,
configuration, preparation manifest, and partial scores are retained.

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_qwen_gsm8k.py --sample-size 1000 --num-shots 3 \
  --batch-size 20 --dtype float32 --max-new-tokens 256 --seed 42 \
  --output-dir accexp/results/qwen_gsm8k_run1
```

`--prepare-only` validates and prepares without loading the base weights onto
GPU. Use a fresh output directory for every run; existing artifacts are not
overwritten. Scores are fractions in [0, 1], not percentages. The 256-token
generation cap can truncate reasoning; inspect `generation_diagnostics`
before drawing conclusions about model accuracy.

## Qwen2.5-7B-Instruct on Local CNN/DailyMail

`run_qwen_cnn.py` prepares local `3.0.0` train/test Parquet data and runs Base,
LoRA Full, then independently reconstructed MiniKV on one GPU. All three use
the same native Transformers `model.generate()` path, the base model's
tokenizer, greedy configuration, test IDs, and immutable prefix templates.
Only MiniKV capture/reconstruction is custom; query prefill/attention/decode
remain Transformers/PEFT operations. Prefix templates are copied into an
independent working cache for each batch.

Defaults: `/data/hub/qwen2.5-7b-instruct`, adapter
`/data/hub/cnn_dailmail-lora/qwen2.5-7b`, dataset `/data/hub/cnn_dailmail`,
1,000 test articles, seed 42, three complete train article/summary examples,
1,536-token prefix cap, 256 generated tokens, 4,096 total tokens, FP32, batch 8.
The current adapter is rank 64/alpha 128 with all seven attention/MLP targets;
its 392 tensor shapes match the base model, including the four KV heads.
Complete overlength articles are excluded before uniform sampling, not
truncated; reference highlights are never truncated or placed in test inputs.
With the current local data, the fixed prefix is 1,521 tokens; 11,445 of 11,490
test articles fit, 45 are overlength, and the selected inputs span 1,663--3,800
tokens. The output budget is included in every length check.

Important provenance caveat: this adapter's tokenizer has the same vocabulary
and special tokens as Qwen, but ships a Llama-3-format chat template. The
experiment deliberately uses the **base Qwen native template for all modes**;
both template hashes and the mismatch warning are saved in
`adapter_compatibility`. Vocabulary/special-token mismatches still fail.
The checkpoint's actual training prompt is undocumented; this choice fixes
evaluation formatting, not any possible training-time formatting problem.
The GSM8K launcher retains its stricter template-equality check.

Metrics reuse `rouge-score`: macro F1 for ROUGE-1, ROUGE-2 and ROUGE-L, with
ROUGE-L primary and stemming enabled. Existing reference-independent
next-article boundary filtering applies equally to all three methods; full
multiline summaries, raw outputs and unfiltered scores are also retained.
Scores are fractions in [0, 1]. Output-cap/EOS counts, configuration, sample
manifest, partial scores, and prefix consistency checks remain available.

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_qwen_cnn.py --sample-size 1000 --num-shots 3 \
  --batch-size 8 --dtype float32 --max-new-tokens 256 --seed 42 \
  --output-dir accexp/results/qwen_cnn_run1
```

`--prepare-only` validates the real adapter and prepares data without loading
the base weights or using a GPU. Use a new output directory for each run.
Do not overlap this FP32 experiment with another model on the selected GPU.

## Llama-2-7B/13B on Local Multi-News

### Protocol Calibration Before a New Formal Run

New preparation defaults to `--prompt-format alpaca`: the published LoraRetriever
Alpaca **no-input outer wrapper**, with a local Multi-News instruction and complete
train demonstrations. There is no nested `### Summary:` prompt inside the final
instruction. This matches the author's outer inference envelope, not a verified
reconstruction of every FLAN training template. `--prompt-format legacy` keeps the
old Documents/Summary format. Changing code does not reformat existing JSONL files.
Existing data/results are not overwritten or automatically rescored.

First freeze a small **validation-only** subset for the native reference:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -u accexp/multi_news_native_probe.py \
  --prepare-only --sample-size 20 --seed 42 --max-new-tokens 512 \
  --data accexp/results/multi_news_native_validation20_seed42_v2.jsonl

CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/multi_news_native_probe.py --size 13b \
  --data accexp/results/multi_news_native_validation20_seed42_v2.jsonl \
  --output-dir accexp/results/multi_news_native_13b_validation20_v2
```

The probe prepares the same sample IDs for both models and four conditions:
legacy/alpaca crossed with zero-shot/fixed train prefix. It uses the same train
demonstrations for both formats, intersects all input-length constraints **before**
seeded sampling, and never selects prompts by test scores. Both Base and LoRA Full
use native Transformers `generate()` from complete input tokens, batch size 1;
there is no custom decoding, external prefix-cache injection, or MiniKV. Decode
settings are identical in every condition: greedy, EOS or the fixed output cap,
no repetition penalty change. Generation configuration, prompt provenance, raw
outputs, extracted outputs, ROUGE scores, and repetition/EOS/length indicators
are saved. `status.json` and `predictions.jsonl` update after each generation.
The probe intentionally does not declare a winning protocol or change formal data.

Use `--size 7b` and a new output directory to run the other model on the same file.
After protocol calibration, prepare a **new** formal Alpaca JSONL and optionally
use `--require-prompt-format alpaca` on `run_llama2_multi_news.py` to reject legacy
files. Actual `num_shots` and `prefix_max_tokens` are read from the prepared
manifest and passed to the evaluator, so result configuration no longer carries
the unrelated generic three-shot default.

`data/multi_news.py` reads the paired `data/{train,val,test}.src.cleaned` and
`.tgt` files under `/data/hub/multi_news`. It verifies source/target line counts,
indexes byte offsets without retaining the full training text, restores the
original `NEWLINE_CHAR` markers, and preserves the `|||||` document separators.
Empty rows and rows with fewer than two nonempty documents are excluded; this
threshold is configurable with `--min-documents`. References are never truncated.

Both local Llama-2 models use one fixed complete train document/summary example
and the same 1000 random eligible test examples, seed 42. Because Multi-News
inputs and summaries are longer, defaults are one shot, a 1024-token prefix cap,
and a 512-token maximum output budget. All complete inputs are checked against
both tokenizers and the common minimum context limit (4096 here) before sampling.
There is no silent reduction in sample count or fallback to truncated documents.
The resulting subset is length-filtered, not representative of arbitrary-length
Multi-News. Metadata records exclusions, sample IDs, source/prefix/data hashes,
and the fact that only train demonstrations, not test documents, are cached.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/data/multi_news.py \
  --prompt-format alpaca --sample-size 1000 --num-shots 1 --max-new-tokens 512 --seed 42 \
  --output accexp/results/multi_news_llama2_shared_seed42.jsonl

CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_llama2_multi_news.py --size 7b \
  --require-prompt-format alpaca \
  --data accexp/results/multi_news_llama2_shared_seed42.jsonl \
  --output-dir accexp/results/multi_news_llama2_7b
```

Change `--size` to `13b` and select another output directory for that model.
Adapters default to `/data/hub/multi_news-lora/llama2-{size}`, with safetensors
base weights for 7B and bin base weights for 13B. Batch defaults are 8 and 4,
configurable with `--batch-size`. Each model runs on one device, sequentially
Base -> LoRA Full -> LoRA MiniKV, using independently prepared third-stage
BaseKV/MiniKV and the existing inference implementation. The launcher checks
adapter base compatibility, prepared-data checksums, sample IDs, and output budget.

Macro-average ROUGE-1/2/L F1 with stemming is reported, primarily ROUGE-L.
`multi_news_summary_boundary_v2` applies the same reference-independent rule to
all three methods, preserving the full multiline summary and cutting only explicit
line-start next-documents/instruction/Alpaca-preamble markers. Raw outputs, extracted summaries,
unfiltered scores and EOS/output-limit diagnostics are retained. The local template
is not claimed to reproduce undocumented adapter training prompts. Existing CNN,
DROP and CoQA experiment defaults and inference code are unchanged.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -m unittest discover \
  -s accexp -p test_llama2_multi_news.py -v
```

## Llama-2-7B/13B on Local CNN/DailyMail

`data/llama2_cnn.py` prepares one shared JSONL for the local Llama-2 models at
`/data/models_2/llama-2-7b-hf` and `/data/hub/Llama2-13b-hf`. Data defaults to
`/data/hub/cnn_dailmail`, version `3.0.0`. Three complete train article/summary
examples form the fixed prefix under a 1536-token cap. The current test article
and summary instruction form each query; `highlights` is used only as the target.
Only train demonstrations are in the approximated prefix, not the current test
article. The local template is not claimed to match the adapters' undocumented
training template.

Defaults are 1000 test articles, seed 42, and 256 maximum generated tokens.
Prefix selection and input eligibility are checked with both tokenizers before
sampling, using the common minimum context limit (4096 for these two models).
Overlength articles are excluded, not truncated; references are never truncated.
The metadata records sample IDs, prefix/data hashes, source files, and exclusions.
Use `--test-split validation` with a different output file for a separate pilot.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/data/llama2_cnn.py \
  --sample-size 1000 --seed 42 \
  --output accexp/results/cnn_llama2_shared_seed42.jsonl

CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_llama2_cnn.py --size 7b \
  --data accexp/results/cnn_llama2_shared_seed42.jsonl \
  --output-dir accexp/results/cnn_llama2_7b
```

Change `--size` to `13b` and use a new output directory for the other model.
The runner selects `/data/hub/cnn_dailmail-lora/llama2-{size}`, safetensors for
7B and bin for 13B. Conservative batch defaults are 8 and 4 respectively;
`--batch-size` can be overridden. Each model runs single-device Base -> LoRA Full
-> LoRA MiniKV with independent third-stage BaseKV/MiniKV preparation, using
the existing inference implementation. It verifies adapter base compatibility,
prepared-data checksums, sample IDs, and the common generation budget.

`scores` reports macro-average ROUGE-1/2/L F1 with stemming, with ROUGE-L as the
primary metric. All three modes use `cnn_summary_boundary_v1`: preserve the
entire multiline summary, cutting only explicit line-start next-article or
instruction markers. No reference is used for extraction. Raw outputs,
`unfiltered_scores`, per-prediction `summary_extraction`, and EOS/output-limit
diagnostics are saved. Scores are fractions in [0, 1]. The existing Mistral CNN
runner still scores raw outputs and is not switched to this optional filter.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -m unittest discover \
  -s accexp -p test_llama2_cnn.py -v
```

## Llama-2-7B/13B on Local DROP

`data/llama2_drop.py` prepares one shared JSONL for the local Llama-2 checkpoints
at `/data/models_2/llama-2-7b-hf` and `/data/hub/Llama2-13b-hf`. It reuses DROP's
original grouped annotations under `results/drop_annotations/`, matches them to
the local `/data/hub/drop` Parquet data, and selects three complete training
examples as the fixed prefix. Defaults are 1000 validation questions, seed 42,
a 1536-token prefix cap, and a 256-token generation budget. Prefix selection
and eligibility are checked with both tokenizers before sampling, using the
smaller context limit (4096 for these two models). No passage or answer is
truncated. The cached content is train demonstrations, not the current query's
evidence passage. The local prompt is not claimed to match the undocumented
Styxxxx adapter training template.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/data/llama2_drop.py \
  --sample-size 1000 --seed 42 \
  --output accexp/results/drop_llama2_shared_seed42.jsonl

CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_llama2_drop.py --size 7b \
  --data accexp/results/drop_llama2_shared_seed42.jsonl \
  --output-dir accexp/results/drop_llama2_7b
```

Change `--size` to `13b` and select a new output directory for the other model.
The runner selects `/data/hub/drop-lora/llama2-{size}`, safetensors for 7B and
bin for 13B. Conservative batch defaults are 8 and 4 respectively for long
inputs and 256 generated tokens; `--batch-size` is configurable.

Each model runs single-device Base -> LoRA Full -> LoRA MiniKV. Prefix caches
are prepared once per stage and reused. The MiniKV stage independently prepares
BaseKV with LoRA disabled, captures MiniKV with LoRA enabled, and reconstructs
its prefix cache before generating with the adapter enabled. The runner verifies
data checksums, identical sample IDs, output budgets, and matching adapter bases.

`scores` uses DROP's official number-aware EM/F1 and optimal multi-span alignment
after uniform, reference-independent `drop_turn_boundary_v1` extraction. Only
explicit line-start next-passage/question markers are removed; multiline answers
are retained. Complete JSON string arrays remain multi-span answers. Original
generations, extracted answers, `unfiltered_scores`, and extraction/EOS/length
diagnostics are saved. Scores are fractions in [0, 1]. The existing Mistral DROP
runner still scores whole outputs and is not switched to this optional filter.

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -m unittest discover \
  -s accexp -p test_llama2_drop.py -v
```

## Mistral-7B on Local DROP

`run_mistral_drop.py` uses the same single-device sequential Base, LoRA Full,
then independently prepared MiniKV execution as the CNN/DailyMail launcher.
The shared evaluator and existing experiments are unchanged. Defaults are
1000 validation questions, three complete train demonstrations, prefix cap
1536, output budget 256, seed 42, and batch size 20. Model dimensions and the
default context cap are read from the model config (4096 for this checkpoint).
Passages/questions are never truncated; overlength examples are excluded before
seeded sampling. All three modes consume the identical prepared JSONL.

Local paths are `/data/hub/mistral-7b-v0.1`, `/data/hub/drop-lora`, and
`/data/hub/drop`. The adapter's Predibase passage/question/answer template is
used, with a fixed shared output instruction: answer text only, or a JSON array
of strings for a multi-span answer. Full generated output is scored, without
looking up a correct substring or filtering only one mode. Complete JSON string
arrays are parsed as multiple predicted spans; other output is one span.

The local HF Parquet contains `answers_spans`, but flattens different
annotators' answers together with their multi-span components. It is unsafe to
treat every flattened string as an alternative complete answer. Preparation
therefore matches the original official dev JSON by section ID, question ID,
question text, and passage. The original answer and validated answer groups are
kept in `metadata.drop_gold_answers`. One repeated question ID in the local data
is disambiguated by question text and source index, not silently overwritten.

Install the scoring dependencies in the existing accuracy environment:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -m pip install -r accexp/requirements.txt
```

Provide `--gold-file /path/to/drop_dataset_dev.json`, `--gold-archive` with the
official ZIP, or permit a one-time download with `--download-gold`. Only dev
annotations are extracted, with SHA-256 provenance, under
`accexp/results/drop_annotations/` by default. Cached gold is reused offline;
no network access occurs without the download flag. The local Parquet remains
the source of evaluation questions and train demonstrations.

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_mistral_drop.py \
  --model /data/hub/mistral-7b-v0.1 --lora /data/hub/drop-lora \
  --dataset-dir /data/hub/drop --download-gold \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 --batch-size 20 \
  --dtype float16 --weight-format bin --attn-implementation sdpa --seed 42 \
  --output-dir accexp/results/mistral_drop_run1
```

`--prepare-only` validates/export samples without loading model weights. Result,
journal, partial scores, status, source counts and EOS/output-limit diagnostics
use the same file layout as the CNN launcher. Scores are `exact_match` and
`drop_f1`, fractions in [0, 1]. DROP F1 uses set-based word bags, numerical
normalization and gating, optimal one-to-one span matching, official per-example
two-decimal rounding, and a macro mean of the best score over grouped gold
annotations. These are not the generic English token F1 or ROUGE metrics.
The protocol uses a local validation subset and few-shot prompting, not an
assertion of directly comparable official leaderboard results.

Metric reference:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py

Offline tests:

```bash
CUDA_VISIBLE_DEVICES='' /home/qlh/anaconda3/envs/transformer/bin/python \
  -m unittest discover -s accexp -p 'test_mistral_drop.py'
```

## Mistral-7B on Local CNN/DailyMail

`run_mistral_cnn.py` uses one GPU and strictly completes Base, then LoRA Full,
then MiniKV. It does not start a second model or use tensor/pipeline parallelism.

1. Base prefills the fixed prefix once, reuses immutable BaseKV for every test
   article, and releases its prefix templates after the Base stage.
2. LoRA Full attaches the adapter, prefills the same prefix once without MiniKV
   capture, reuses native adapted prefix KV, and releases it after this stage.
3. MiniKV prepares its own sources: temporarily disable the already loaded
   adapter and prefill the prefix for fresh BaseKV, then re-enable it and prefill
   once with LoRA-A hooks. Discard the native adapted KV from this capture pass.
   Reconstruct from BaseKV plus MiniKV and actual LoRA-B/scaling once per prefix,
   then reuse the reconstructed prefix for all remaining test articles.

The third stage does not borrow the first two stages' cached sources. Disabling
and re-enabling the adapter avoids reloading weights while preserving the two
distinct forward computations. Each query/decode works on a fresh cache clone;
prefix templates never acquire previous test articles or generated answers.
Additional direct/split prefix-injection checks are validation forwards, not
per-request prefix prefill. Results record these checks and source-preparation
counts separately. This behavior is enabled by `--minikv-independent-prefill`
in the common evaluator; older experiments keep their existing defaults.

The reader uses local `3.0.0` Parquet shards only. It selects three complete train
article/summary demonstrations with seed 42 and a 1536-token prefix cap, then
uniformly samples 1000 eligible official test articles. Every mode receives the
same file. Test highlights are references only. Overlength articles are excluded,
not truncated; the report preserves selected IDs, filtering counts and hashes.
`--sample-size 0` uses all eligible articles. Default total context is the smaller
of the model limit and sliding window: 4096 for this local Mistral configuration.
That budget includes the fixed prefix, new article prompt and 256 output tokens.
An explicit `--max-context-len` can change this cap without modifying the model.

Install the additional scoring dependency in the evaluation environment:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python -m pip install -r accexp/requirements.txt
```

Run from the repository root, choosing a new output directory for each run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/transformer/bin/python -u \
  accexp/run_mistral_cnn.py \
  --model /data/hub/mistral-7b-v0.1 \
  --lora /data/hub/cnn_dailmail-lora \
  --dataset-dir /data/hub/cnn_dailmail \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-new-tokens 256 --seed 42 --weight-format bin \
  --output-dir accexp/results/mistral_cnn_run1
```

`--prepare-only` exports data and its manifest without loading model weights.
The launcher defaults to one request at a time; set `--batch-size 32` to batch
nearby query lengths on a dedicated 80 GB GPU, then check actual memory use.
The last batch can be smaller. All three modes use the same mixed-length
batching path without changing the prefix, article, output budget or window.
The local safetensors downloads were
incomplete when this launcher was added, so `bin` is the explicit default.

`result.json` includes raw predictions, macro-average stemmed ROUGE-1/2/L F1
fractions in [0, 1], MiniKV-minus-LoRA deltas, source preparation counts, cache
checks and per-mode EOS/output-limit diagnostics. ROUGE-L here is sequence-level,
not sentence-union ROUGE-Lsum; no NLTK sentence-model download is needed. No
Alpaca-specific filtering or reference-dependent answer extraction is applied.
The prediction journal and partial results are updated after every batch;
`status.json` records preparation, execution and completion/failure.

Tests: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=accexp python -m unittest test_mistral_cnn -v`.

## MiniKV Definition and Checks

The Base pass saves only prefix KV. During the adapted prefix forward, hooks
capture K/V LoRA A outputs from each layer. The reconstruction uses the actual
PEFT B projection and scaling. Because native cached K is already rotated,
the K delta is rotated with the cos/sin tensors from the adapted prefix forward
before addition to BaseK. Missing K/V LoRA projections contribute zero delta.

This is an approximation: the adapted hidden states feeding LoRA A can differ
from those that produced BaseKV. It is not assumed to reproduce a full LoRA
forward exactly. Full LoRA prefix KV is used only for its reference run and
cache-injection check; it is not a reconstruction input.

For the first sample of every unique prefix, Base and full LoRA each compare
last-query logits from full-input prefill against prefix-cache plus query
prefill. Differences exceeding `--cache-check-atol`/`--cache-check-rtol` abort.
The default absolute tolerance is dtype-dependent (FP16 0.03, BF16 0.15,
FP32 0.0001); relative tolerance is 0.005. FP16 split/full prefill has small
shape-dependent rounding differences even without MiniKV. The actual thresholds,
maximum and relative L2 errors, and first-token agreement are saved. The two
reference modes generate from their own native prefix caches. Full-input prefill
is retained for the consistency checks, not repeated for every evaluation query.
Batching can introduce additional low-precision rounding differences; generated
answers need not be bit-identical to the old single-request evaluation.
MiniKV is not subjected to an equality assertion against full LoRA, since that
would confuse approximation error with cache-injection correctness.

Caches preserve the entire prefix; Mistral's native sliding attention mask
still controls visibility. This is an accuracy implementation, not a memory or
throughput benchmark. Sequence-dependent RoPE changes can invalidate split/full
equivalence; the consistency check detects that rather than silently accepting
the experiment.

Offline verification uses tiny real Transformers models and nonzero PEFT
adapters, with no model downloads:

```bash
/home/qlh/anaconda3/envs/transformer/bin/python accexp/test_eval_minikv.py -v
```
