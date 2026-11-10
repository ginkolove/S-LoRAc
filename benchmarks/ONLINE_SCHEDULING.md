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
