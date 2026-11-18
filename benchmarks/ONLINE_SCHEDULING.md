# Online scheduling without output-length admission

Use `--scheduler online` on `benchmarks/launch_server.py` or
`python -m slora.server.api_server`. The managed
`benchmarks/lowra_saturation_exp.py` now defaults to `online` and forwards it to
the server. Use `--scheduler slora` to reproduce the old reservation policy.
Existing server/launcher defaults remain unchanged for compatibility.

This implements vLLM-style incremental admission and recompute preemption in
the existing S-LoRA engine. It does not import vLLM or replace the attention,
MiniKV reconstruction, or Unified Paging allocator.

## Admission and recovery

- FCFS waiting order. `--max-active-loras N` optionally caps distinct running
  adapters, not requests; incompatible requests can be skipped at this limit.
- `--schedule-interval 1` tries admission after each decode step. Larger values
  change admission frequency only: memory is still checked before every decode.
- Admission uses known prompt/history tokens, one-step headroom, unique adapter
  weight slots, shared prefix allocations, and the single-layer scratch budget.
  It never reads `max_output_len`, `lowra_decode_len`, or reference answer lengths.
- An idle shared prefix is retained for reuse. Under pressure, unpinned idle
  entries are evicted in LRU order before requests are preempted.
- If decode still cannot fit, running requests are preempted from the tail.
  Private KV is freed and shared-prefix references are released. The request
  returns to the front of the waiting queue without completing or emitting a
  duplicate token.
- Resume prefills the original private prompt plus generated token history.
  Output token counts, stopping conditions, and presence/frequency penalties
  continue from the original request. Shared prefixes follow the existing cache
  policy; static BaseKV and MiniKV are not discarded.
- If a single request cannot fit even after reclaiming eligible idle caches,
  it fails explicitly through the existing abort response instead of hanging
  or causing a pool allocation exception. Pinning may reduce usable capacity.

For running request i, private KV after its next decode is `input_i + emitted_i`.
For an admitted/resumed request j, the conservative prefill-plus-next-decode
budget is `input_j + emitted_j + 1`. These are observed lengths, not predictions.
Prefix memory is charged once per adapter. Layer storage keeps the existing
`ceil(prefix_tokens * retained_layers / model_layers)` packing, and transient
scratch space is also charged. Old/new top-K allocations are considered during
rebalance so admission does not rely on memory that has not been released yet.

`max_new_tokens` remains a legal generation cap, not a required known output
length. Context-length checks still apply. The old HTTP check that required the
entire input-plus-output cap to fit the KV pool is disabled for `online`.

## Benchmark generation semantics

The default `--generation-mode trace` keeps deterministic recorded/synthetic
output lengths and ignores EOS for controlled performance replay. In online
mode those lengths control termination only, not admission. This is still trace
replay, not a claim that the model naturally produced those answer lengths.

For actual EOS-terminated generation, add:

```bash
--scheduler online --generation-mode eos --generation-max-new-tokens 512
```

This uses one common output cap (clamped to remaining context) independent of
the dataset's answer lengths. Success accepts early EOS, and throughput counts
actual returned tokens. Do not use dummy weights to assess natural answer
quality or realistic EOS behavior.

Manifests record the scheduler, interval, adapter cap, generation mode, and
`output_length_admission`. Runtime JSONL includes cumulative
`online_preemptions`, `online_recomputed_tokens`, and `online_rejections`.
These counters reset with the idle-runtime reset before each measured replay.
The paired-comparison script rejects runs with mismatched scheduling semantics.

## Scope

All existing prefix modes use the same online planner: full-reprefill,
minikv-full_reconstruct, minikv-layer_serial, layer_prefetch, and
minikv-layer_retained, including dynamic top-K and pinned entries.

This version retains whole-query/recompute prefill and S-LoRA token slots.
It does not add chunked prefill, mixed prefill/decode kernel execution,
CPU swapping of private KV, or an SLO guarantee. `batch_max_tokens` still limits
whole-prefill work and running decode width, rather than implementing all V1
token-budget semantics. Large prompts must fit the configured prefill budget.
Legacy adapter prefetch, clustering, abort heuristics, BMM, quantized modes, separate LoRA pools,
and debug weight-bypass modes are rejected with `online`.

## Regression tests

```bash
CUDA_VISIBLE_DEVICES='' /home/qlh/anaconda3/envs/slora/bin/python -m unittest discover \
  -s test -p test_online_scheduler.py -v

CUDA_VISIBLE_DEVICES=0 RUN_ONLINE_CUDA_TESTS=1 \
  /home/qlh/anaconda3/envs/slora/bin/python -m unittest discover \
  -s test -p test_online_scheduler_cuda.py -v
```

The opt-in GPU test uses a small three-layer, 1024-hidden dummy model with
nonzero MiniKV/BaseKV. It compares uninterrupted and preempted/resumed greedy
outputs, with sampling penalties, and verifies private/prefix/workspace cleanup.
It is a correctness test, not a serving-throughput benchmark or a TP=2 test.

# Automatic Replay Limits

The LMSYS launchers no longer impose a 16,384-token context limit, a
36,864-token prefill cap, or a 256-request running limit by default.
`prepare_lmsys_maf_trace.py` selects nonempty examples in a fixed seed order
without filtering by length, then derives the total context limit from
`prefix_len + max(query_len + answer_len)` over the selected requests.
Changing only the prefix length preserves sample IDs, query/output lengths,
arrival times and adapter assignments. This remains a first-turn workload with
a synthetic shared prefix, not a native multi-turn agent trace.

An explicit `--max-context-len` that is too small fails rather than silently
selecting shorter samples. The data-preparation-only `--filter-over-context`
option reproduces the old filtering policy when an explicit cap is supplied.
Existing frozen traces remain readable; their previously filtered-out samples
cannot be restored by raising a server limit. Generate a fresh trace/output
directory for the new sampling policy.

The benchmark inherits context length from the trace. Online prefill capacity
defaults to `num_token`, and the running-request limit defaults to the smaller
of the pool token count and the number of trace requests (pool count for
synthetic workloads). These are nonbinding bounds relative to the pool/trace,
not output-length reservations. The online scheduler continues to use only
known input/history plus immediate KV headroom, checking space before each
decode and using existing eviction/preemption when necessary. A prefill token
cap no longer also limits the decode batch's request count.

The paired launcher accepts explicit `--max-context-len`, `--batch-max-tokens`,
`--running-max-req-size`, and `--schedule-interval` overrides. Context and prefill
budgets are independent for online scheduling: a long shared prefix does not
require an equally large query-prefill cap. Explicit small prefill caps can
still make individual requests inadmissible; chunked prefill is not introduced.
Legacy S-LoRA scheduling keeps its original length-budget validation.

Ordinary LLaMA RoPE tables now cover at least the configured server context
limit without changing frequencies or wrapping positions. This permits long
performance experiments, not a claim of long-context model accuracy. KV pool
budgeting still does not guarantee enough memory for activations, workspaces
and other allocations. Keep GPU headroom and use an explicit prefill cap when
needed; removing fixed experimental caps is not an OOM guarantee. Resolved
limits are printed at launch and saved in the manifest.

## Matched Model Weights + Pool Budget

The three LMSYS shell launchers share `llama_memory_budget.sh`. The current
profile retains the original A100-80GB paper pool budgets for 7B (125,000 tokens)
and 13B (65,000 tokens), and assigns 8,800 tokens to 30B so that **weights plus
the total pool are about 79.1 GB (decimal) for all three models**. This supersedes
the earlier 73-decimal-GB cap and idle-memory-minus-7-GB policy. The comparison
uses measured FP16, TP=1 `base_model_weight_bytes`, excluding model buffers such
as RoPE tables, and `4 * layers * hidden_size` bytes per pool token. The 30B
model is our 60-layer, 6656-hidden model, not the original modified S5 model.
This is a frozen reproducible profile, not a live GPU-memory autosizer.

| Model | Weight bytes | Bytes per token | Default pool tokens | Weights + pool GB |
| --- | ---: | ---: | ---: | ---: |
| Llama-7B | 13,476,831,232 | 524,288 | 125,000 | 79.012831 |
| Llama-13B | 26,031,728,640 | 819,200 | 65,000 | 79.279729 |
| Llama-30B | 65,057,887,232 | 1,597,440 | 8,800 | 79.115359 |

The original budgets are in `paper/synthetic/slora/exp_cmds_a100-80_S1.sh` and
`paper/synthetic/slora/exp_cmds_a100-80_S4.sh`.

The measurements come from per-rank `base_model_weight_bytes` in `runtime.jsonl`:

- `results/llama7b_prefix15k_req3_180s_20261114_121857/`
- `results/lmsys_preemption_13b_p10000_r5_t240_gpu0_20261111_092926/`
- `results/lmsys_preemption_30b_p4000_r3_t180_20261111_062609/`

The pool budget still includes static BaseKV/MiniKV and dynamic prefix, private
KV, adapter and layer-workspace allocations. Both idle A100-80GB GPUs reported
81,153 MiB available on 2026-11-14, approximately 85.095 GB. The remaining
5.815-6.082 GB must cover model buffers, runtime activations, pool-external
temporaries, allocator fragmentation and CUDA costs.
This is not a hard peak-memory limit or an OOM guarantee. Recalibrate for a
different GPU, dtype, TP degree, model or substantially larger model buffers.
Direct `lowra_saturation_exp.py --num-token` values and the rank launcher's
explicit token override remain authoritative and can leave this matched profile.

**30B with a 4,000-token prefix can admit short queries but cannot complete the
whole current trace in full reconstruction at this budget.** Its static region
uses 4,481 equivalent slots; one full prefix and a rank-16 adapter leave only
255 slots for private prompt/history. A one-token query plus immediate headroom
fits (8,547 slots total), so the launcher's minimum-capacity check passes, but
some queries and the longest private history (1,529 tokens) cannot fit. An
explicit `--model-size 30b --prefix-len 3000` passes that single-request history
capacity check. The suite never silently shortens the prefix or drops samples
to fit. The early check is only a minimum capacity check; actual query/history
growth and execution peaks still matter.
