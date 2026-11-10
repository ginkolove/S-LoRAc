"""Independent full-element correctness audit, not a performance benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slora.common.lowra_reconstruct import lowra_reconstruct_batched
from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded


def check_schedule():
    checked = 0
    for count in (1, 7, 8, 9, 16, 17, 20, 33):
        for workers in (4, 27, 648, 1296):
            p_tiles, n_tiles, group_size = 3, 5, 8
            total = ((count + 7) // 8) * 8 * p_tiles * n_tiles
            visits = {}
            for worker in range(workers):
                for tile in range(worker, total, workers):
                    n = tile % n_tiles
                    group = tile // (n_tiles * p_tiles * group_size)
                    adapter = group * group_size + (tile // n_tiles) % group_size
                    p = (tile // (n_tiles * group_size)) % p_tiles
                    if adapter < count:
                        key = (adapter, p, n)
                        visits[key] = visits.get(key, 0) + 1
            expected = {(a, p, n) for a in range(count)
                        for p in range(p_tiles) for n in range(n_tiles)}
            assert set(visits) == expected and set(visits.values()) == {1}
            checked += 1
    return checked


@torch.no_grad()
def check_case(prefix, embed, source_ranks, selected_count, workers):
    count = len(source_ranks)
    total_rank = sum(source_ranks)
    max_rank = max(source_ranks)
    options = dict(device="cuda", dtype=torch.float16)
    base_k = torch.randn(prefix, embed, **options) * 0.1
    base_v = torch.randn_like(base_k) * 0.1
    mini_k = torch.randn(prefix, total_rank, **options) * 0.1
    mini_v = torch.randn_like(mini_k) * 0.1
    b = torch.randn(2 * total_rank + 13, embed, **options) * 0.1
    b_rows = torch.randperm(b.shape[0], device="cuda")[:2 * total_rank]
    k_locs = torch.zeros(count, max_rank, device="cuda", dtype=torch.long)
    v_locs = torch.zeros_like(k_locs)
    starts = []
    cursor = 0
    for index, rank in enumerate(source_ranks):
        starts.append(cursor)
        k_locs[index, :rank] = b_rows[cursor:cursor + rank]
        v_locs[index, :rank] = b_rows[total_rank + cursor:total_rank + cursor + rank]
        cursor += rank
    scales = torch.tensor([(-0.5, 0.0, 0.25, 1.0, 1.75)[i % 5]
                           for i in range(count)], **options)
    # Shuffle source order independently from destination order, as after a batch filter.
    selected = torch.randperm(count, device="cuda")[:selected_count]
    source_ids = selected.cpu().tolist()
    rank_starts = torch.tensor(starts, device="cuda", dtype=torch.long)[selected]
    rank_lens = torch.tensor(source_ranks, device="cuda", dtype=torch.long)[selected]
    rows = count * prefix + 13
    locations = torch.randperm(rows, device="cuda")[:selected_count * prefix].view(selected_count, prefix)
    untouched = torch.ones(rows, device="cuda", dtype=torch.bool)
    untouched[locations.reshape(-1)] = False
    report = dict(prefix=prefix, embed=embed, source_ranks=source_ranks,
                  selected_count=selected_count, workers=workers, outputs={})
    for name, base, mini, b_locs in (
        ("K", base_k, mini_k, k_locs), ("V", base_v, mini_v, v_locs),
    ):
        indices = b_locs[selected].contiguous()
        actual = torch.full((rows, embed), 13.0, **options)
        actual[locations.reshape(-1)] = float("nan")
        original = actual.clone()
        lowra_reconstruct_batched(base, mini, b, original, locations, indices,
                                  rank_starts, rank_lens, scales[selected])
        lowra_reconstruct_bounded(base, mini, b, actual, locations, indices,
                                  rank_starts, rank_lens, scales[selected], workers=workers,
                                  launch_config=(32, 128, 16, 4))
        # Every selected element must replace NaN; guards cover all unselected rows.
        assert bool(torch.isfinite(actual).all())
        assert bool((actual[untouched] == 13).all())
        torch.testing.assert_close(actual, original, atol=1e-4, rtol=1e-3)
        max_old_error = 0.0
        max_reference_error = 0.0
        exact = True
        for row, source in enumerate(source_ids):
            start, rank = starts[source], source_ranks[source]
            expected = base.float() + (
                mini[:, start:start + rank].float() @ b[b_locs[source, :rank]].float()
            ) * scales[source].float()
            result = actual[locations[row]]
            previous = original[locations[row]]
            torch.testing.assert_close(result.float(), expected, atol=0.0005, rtol=0.001)
            exact = exact and torch.equal(result, previous)
            max_old_error = max(max_old_error, (result.float() - previous.float()).abs().max().item())
            max_reference_error = max(max_reference_error, (result.float() - expected).abs().max().item())
        report["outputs"][name] = dict(
            checked_elements=selected_count * prefix * embed,
            bitwise_equal_to_original=exact,
            max_abs_error_vs_original=max_old_error,
            max_abs_error_vs_fp32_reference=max_reference_error,
            all_selected_elements_written=True, unselected_rows_unchanged=True,
        )
    torch.cuda.synchronize()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(0)
    torch.manual_seed(20261106)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [
        (257, 4097, [1, 3, 8, 16, 40] * 7, 33, 27),
        (4608, 4096, [16] * 20, 20, 648),
        (4608, 4096, [16] * 20, 20, 1296),
        (4609, 4096, [8, 16] * 10, 17, 1296),
    ]
    report = dict(schedule_cases=check_schedule(), seed=20261106,
                  torch=torch.__version__, device=torch.cuda.get_device_name(0),
                  fp32_reference_tf32=False, cases=[])
    for case in cases:
        result = check_case(*case)
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
    report["all_passed"] = True
    paths = ["slora/common/lowra_reconstruct.py", "slora/common/lowra_reconstruct_bounded.py",
             "slora/common/lowra_layer_runtime.py", "try/audit_bounded_reconstruct.py"]
    report["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
