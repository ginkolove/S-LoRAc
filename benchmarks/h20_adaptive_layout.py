"""Queue-head admission alternatives for the H20 actual/16 MiniKV experiment."""
from dataclasses import dataclass


def tiled_rank(rank):
    return ((rank + 15) // 16) * 16


@dataclass
class HeadComparison:
    keep: object
    release: object
    released: dict

    def record(self, choice):
        return dict(request_id=self.release.selected[0].request_id, window=self.keep.window,
                    keep_score=self.keep.score, release_score=self.release.score,
                    candidate_released_layers=self.released, admitted=bool(choice.selected))


def blocked_head_comparison(queue, running, waiting, ranks, window):
    """Compare keeping ready KV with admitting the FCFS head at the same horizon."""
    if not running or not waiting:
        return None
    head = waiting[:1]
    retained = queue.prefix_cache_state.get("retained_layers", {})
    targets = {r.adapter_dir: retained.get(r.adapter_dir, 0) for r in running + head}
    donors = [d for d, layers in targets.items() if layers]
    if not donors or queue._layout(running, head, ranks, targets, window).fits:
        return None

    # Both actions must be possible over the comparison horizon. A non-memory
    # admission limit (or an infeasible next decode) is handled by the router.
    serial = dict.fromkeys(targets, 0)
    def both_fit(horizon):
        return (queue._layout(running, head, ranks, serial, horizon).fits and
                queue._layout(running, [], ranks, retained, horizon).fits)

    if not both_fit(window):
        if not both_fit(1):
            return None
        lo, hi = 1, window - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if both_fit(mid):
                lo = mid
            else:
                hi = mid - 1
        window = lo
        if queue._layout(running, head, ranks, targets, window).fits:
            return None

    # The allocator charges a scratch row when a full prefix becomes partial.
    # Use its exact rounded cost even for the last layer we can afford to keep.
    ready = queue.prefix_cache_state.get("ready_layers", {})
    last_used = queue.prefix_cache_state.get("last_used", {})
    donors.sort(key=lambda d: (tiled_rank(ranks[d]), ready.get(d, 0) > 0,
                               last_used.get(d, 0), -targets[d], d))
    released = {}
    for directory in donors:
        old = targets[directory]
        targets[directory] = 0
        if not queue._layout(running, head, ranks, targets, window).fits:
            released[directory] = old
            continue
        lo, hi = 0, old - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            targets[directory] = mid
            if queue._layout(running, head, ranks, targets, window).fits:
                lo = mid
            else:
                hi = mid - 1
        targets[directory] = lo
        released[directory] = old - lo
        break

    keep = queue._evaluate(running, [], ranks,
                           {r.adapter_dir: retained.get(r.adapter_dir, 0) for r in running}, window)
    release = queue._evaluate(running, head, ranks, targets, window)
    return HeadComparison(keep, release, released)
