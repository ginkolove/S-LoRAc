# Packed B Correctness Audit

## Confirmed Integration Bug

The original S-LoRA loader stores HF LoRA-B `[hidden, rank]` row-major across
`rank` physical slots, each `hidden` elements wide. Those slots may be scattered.
`InferAdapter.load_lora_B` preserves this packing; the original BGMV/LoRA-prefill
expand kernels explicitly decode the flattened `(hidden, rank)` offset.

MiniKV reconstruction previously interpreted a slot as a logical row of
`B.T[rank, hidden]`. Split, fused and bounded reconstruction shared that mistake.
New/old agreement was insufficient evidence: previous tests either used a
rank-major synthetic B matrix or computed their reference from the same mistaken
view of packed slots. Performance runs used zero BaseKV/MiniKV and therefore
could not expose a permutation of B in their output values.

For HF B `[[1,2],[3,4],[5,6],[7,8]]`, packed slots are
`[[1,2,3,4],[5,6,7,8]]`. Mini `[1,0]` incorrectly produced `[1,2,3,4]`, rather than
the correct `[1,3,5,7]`.

A regression through `LoraLayerWeight.load_hf_weights_cpu` and
`InferAdapter.load_adapters`, using nonzero synthetic HF-format weights and
scattered slots, failed before the fix: 131255/132096 elements differed (99.4%),
with maximum absolute difference 0.5296564102172852 in its first comparison.
These are synthetic matrices through the real loader, not pretrained checkpoints.

## Correction

For output feature `e` and low-rank coordinate `j` of actual rank `r`, read:

```text
flat = e * r + j
physical_slot = B_LOCS[flat // hidden]
element_offset = flat % hidden
B[e,j] = B_BUFFER[physical_slot, element_offset]
```

All three reconstruction kernels accept explicit `b_packed` selection, and all
serving layer wrappers pass `True`. The default `False` is the mathematical
rank-major buffer contract of the standalone synthetic matrix diagnostics, not
the S-LoRA adapter pool contract. No B unpacking tensor or extra prefix buffer is
allocated. Pool budgets, admission, retention policy and stream ordering are
unchanged. This does change memory accesses, so old timings must not be reused as
performance results for the corrected path.

## Independent Checks

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/qlh/anaconda3/envs/slora/bin/python -m unittest discover \
  -s test -p test_lowra_packed_reconstruction.py -v
```

Checks derive their expected values from the original HF-format CPU weights,
not the packed pool or an old reconstruction kernel. MiniKV is formed from a
nonzero hidden input and the original LoRA-A. TF32 is disabled for FP32 references.

- Original BGMV expand is independently compared with `Mini @ HF_B.T * scaling`.
- Split, fused and bounded wrappers are checked for K and V at every test layer.
- Cases include width 4096, 20 adapters, mixed ranks 8/16/32/64, a 9-adapter group
  boundary, shuffled selected order, scattered adapter slots and masked prefixes,
  including length 4609 with a partial final tile.
- A negative control deliberately uses the old interpretation and must fail the
  numerical reference by a substantial margin.
- Serial, single-buffer parallel and dynamic Top-1 retention feed reconstructed
  KV into prefill and decode attention; outputs are checked against independent
  FP32 causal attention using the original HF weights. Queries are ragged and
  same-LoRA requests share destinations. No per-layer host synchronization is
  inserted between prefix consumption and side-stream overwrites. Private KV
  contents and all final pool slots are checked after execution.

Verified on A100 with PyTorch 2.0.1+cu117 and Triton 2.1.0 after this fix:

| Suite | Passed |
| --- | ---: |
| `test_lowra_packed_reconstruction.py` | 6 |
| `test_full_reconstruct.py` | 9 |
| `test_lowra_layer_runtime.py` | 29 |
| `test_lowra_retained.py` | 13 |
| `test_lowra_pinned.py` | 6 |
| Total distinct tests | 63 |

The full-reconstruct suite ran together with the then-five packed tests (14
passed). The packed suite subsequently passed all six tests after adding the
long-prefix case and rank-64 coverage. No CUDA tests were skipped in these runs.
`py_compile` of changed Python modules and `git diff --check` also passed.
Model-level regression fixtures use small dummy models, not a real 7B checkpoint.
No full serving performance replay was run after this fix; existing GPU services
were not stopped or restarted.

## GPU0 Quantitative Recheck

The follow-up numerical experiment runs on physical GPU0 only, keeping its idle
S-LoRA holder running and leaving GPU1 jobs alone. Production reconstruction
kernels were not changed during this recheck. Results and source hashes are in
`try/results/packed_gpu0_20261106_055626/result.json`.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/validate_packed_reconstruction.py --seeds 42 7 \
  --output-dir "try/results/packed_gpu0_$(date -u +%Y%m%d_%H%M%S)"
```

Reference values use the original HF B matrices, not pool data decoded by the
same indexing as the kernel. Every selected K/V output element is checked
against `Base.float() + (Mini.float() @ HF_B.float().T) * scaling` with TF32 off.
CPU FP64 independently checks all elements in the small cases and 64 selected
rows by 64 features per adapter/layer/K-or-V in large cases: 4,039,680 values
total, maximum FP32-reference versus CPU-FP64 discrepancy 1.637469e-6.

All seven shapes, two seeds each, passed for split, fused-KV, bounded-648 and
bounded-1296. Each configuration checked 2,081,027,072 output elements, for
8,324,108,288 comparisons across the four configurations. This counts repeated
outputs for different configurations, not that many distinct input elements.
There were zero nonfinite outputs and zero elements exceeding
`abs(error) <= 5e-4 + 2e-3 * abs(reference)`.

| Prefix | Width | Selected / Source Adapters | Ranks | Max Abs Error vs FP32 |
| ---: | ---: | ---: | --- | ---: |
| 17 | 128 | 2 / 2 | 8,16; one-hot | 1.220704e-4 |
| 129 | 1024 | 2 / 2 | 8,16 | 2.368689e-4 |
| 257 | 4096 | 9 / 9 | 8,16,32,64 | 1.951218e-3 |
| 4608 | 4096 | 20 / 50 | 16 | 1.620770e-3 |
| 8193 | 4096 | 2 / 3 | 8,16,32 | 1.934052e-3 |
| 1153 | 5120 | 3 / 3 | 8,16,32 | 1.657486e-3 |
| 513 | 6656 | 3 / 3 | 16,32,64 | 1.954079e-3 |

Table errors include FP16 output rounding. The one-hot cases additionally match
their direct column-selection analytical FP16 answers bit-for-bit. Aggregate
mean absolute error was 3.036034e-5 and RMSE 6.191861e-5 for each configuration.
The intentionally wrong old-layout control failed 98.207%-99.908% of elements
in its selected nonzero-scale adapter. Source weight slots and private guard
slots remained unchanged, and every pool allocation was released.

The six packed-loading/attention tests and one variable-length decode test also
passed on GPU0. Their actual outputs are saved beside the JSON as
`attention_and_loading_tests.log` and `decode_variable_lengths_tests.log`.
These include prefill/decode attention consumption in serial, parallel and
Top-1 retention modes, private lengths through 512, and GQA attention. The
reconstruction matrices themselves are MHA-shaped synthetic test inputs; the
5120/6656 cases are not full 13B/30B checkpoint inference.

## Remaining Semantic Boundary

The earlier attention tests checked the generic linear-reconstruction contract,
not real pretrained-model prefix equivalence. The subsequent real-checkpoint
audit found and fixed the missing delta-K RoPE operation. Normal service/profile
entrypoints now bind prefix-position RoPE, and serial/parallel actual 32-layer
Llama-7B inference passed an independent HF/PEFT reconstructed-cache reference.
See `try/MODEL_E2E_AUDIT.md` for before/after results and source-loading limits.
The normal server still initializes placeholder sources; the real audit injects
actual extracted sources. MiniKV approximation loss remains separate from
execution correctness. Accuracy experiments under `accexp` were not changed.
