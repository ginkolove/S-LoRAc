"""FCFS continuous batching with event-driven, fixed-batch prefix KV layouts.

Only this experimental policy changes. Admission depends on one-forward
allocator/hard limits, never predicted throughput, future output lengths, or a
planning horizon. The original adaptive queue still owns elastic storage and
the measured cost model.
"""
from dataclasses import dataclass
import math
import time

from slora.server.router.adaptive_req_queue import (
    AdaptiveReqQueue, MeasuredCostModel, SpaceDecision,
)


POLICY = "minikv-adaptive-v2"
POLICY_REVISION = "fcfs-local-layout-v1"


def tiled_rank(rank):
    return ((rank + 15) // 16) * 16


def prefix_slots(layers, prefix, layer_num):
    """Persistent rows plus one shared, rounded workspace row per streamer."""
    return (sum((k * prefix + layer_num - 1) // layer_num for k in layers)
            + (sum(k < layer_num for k in layers) * prefix + layer_num - 1) // layer_num)


def local_layout_candidates(weights, layer_num, prefix, budget, ready=None):
    """Enumerate common floors, then fill expensive adapters.

    If prefix is divisible by layer_num, grouped affine reconstruction has
    an optimum in this family: some full entries, at most one entry above the
    common floor but below full, and all remaining entries at the floor.
    Full entries reclaim their scratch row. With nonintegral row sizes, use
    exact allocator rounding to generate safe candidates, without claiming
    global optimality. Ready layers break equal-weight ties.
    """
    count = len(weights)
    if not count:
        yield ()
        return
    if prefix <= 0:
        yield (0,) * count
        return
    ready = ready or (0,) * count
    order = sorted(range(count), key=lambda i: (-weights[i], -ready[i], i))
    if prefix_slots((layer_num,) * count, prefix, layer_num) <= budget:
        yield (layer_num,) * count
        return
    unit, remainder = divmod(prefix, layer_num)
    if not remainder:
        capacity = budget // unit
        for common in range(min(layer_num - 2, capacity // count - 1) + 1):
            target = [common] * count
            # With zero incremental weights, spending the remainder may save
            # no time. Keep the common-floor-only candidate for stable ties.
            yield tuple(target)
            spare = capacity - count * (common + 1)
            promotion = layer_num - common - 1
            for i in order:
                if spare >= promotion:
                    target[i] = layer_num
                    spare -= promotion
                else:
                    target[i] += spare
                    break
            yield tuple(target)
        return
    for common in range(layer_num):
        target = [common] * count
        if prefix_slots(target, prefix, layer_num) > budget:
            break
        yield tuple(target)
        for i in order:
            target[i] = layer_num
            if prefix_slots(target, prefix, layer_num) <= budget:
                continue
            lo, hi = common, layer_num - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                target[i] = mid
                if prefix_slots(target, prefix, layer_num) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            target[i] = lo
        yield tuple(target)


class TiledCostModel(MeasuredCostModel):
    def reconstruction_ms(self, ranks, prefix):
        return self.reconstruction_group_ms(len(ranks), sum(map(tiled_rank, ranks)), prefix)

    def layer_intervals_ms(self, intervals, prefix):
        return super().layer_intervals_ms(
            [(first, end, tiled_rank(rank)) for first, end, rank in intervals], prefix)


@dataclass
class LocalDecision(SpaceDecision):
    reconstruction_ms: float = 0.
    first_forward_ms: float = 0.
    first_forward_reconstruction_ms: float = 0.


class AdaptiveV2ReqQueue(AdaptiveReqQueue):
    def __init__(self, *args, **kwargs):
        # Legacy horizon/search/SLO options remain parseable, but they cannot
        # delay admission. interval still controls the runtime sampling rate.
        config = dict(kwargs.pop("space_plan_config", None) or {})
        config.setdefault("hysteresis", 0.)
        super().__init__(*args, space_plan_config=config, **kwargs)
        self.cost = TiledCostModel()
        self.last_membership = None
        self.last_capacity = None
        self.was_calibrated = False
        self.search_evaluations = 0
        self.allocator_checks = 0
        self.layout_budget_slots = 0
        self.solver_exact = False

    def _targets(self, running, selected):
        retained = self.prefix_cache_state.get("retained_layers", {})
        return {r.adapter_dir: retained.get(r.adapter_dir, 0) for r in running + selected}

    def _probe(self, running, selected, ranks, targets):
        self.allocator_checks += 1
        return self._layout(running, selected, ranks, targets, 1)

    def _admit_fcfs(self, running, waiting, ranks):
        """Largest feasible FCFS prefix, allowing reclaimable cache to shrink."""
        if not self._probe(running, [], ranks,
                           dict.fromkeys(self._targets(running, []), 0)).fits:
            return []
        lo, hi = 0, min(len(waiting), max(0, self.running_max_req_size - len(running)))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            selected = waiting[:mid]
            serial = dict.fromkeys(self._targets(running, selected), 0)
            if self._probe(running, selected, ranks, serial).fits:
                lo = mid
            else:
                hi = mid - 1
        return waiting[:lo]

    def _evaluate(self, running, selected, ranks, targets, window=1):
        """Fixed-batch steady decode cost; first-forward cost is diagnostic only."""
        plan = self._probe(running, selected, ranks, targets)
        result = LocalDecision(list(selected), dict(targets), plan, 1)
        requests = list(running) + list(selected)
        if not plan.fits or not requests:
            return result
        dirs = list(dict.fromkeys(r.adapter_dir for r in requests))
        private = sum(self.known_tokens(r) for r in requests)
        core = self.cost.core_ms(len(requests), private,
                                 sum(ranks[r.adapter_dir] for r in requests), self.prefix_len)
        result.reconstruction_ms = self.cost.layer_intervals_ms(
            [(targets[d], self.layer_num, ranks[d]) for d in dirs], self.prefix_len)
        result.step_ms = core + result.reconstruction_ms
        result.score = 1000. * len(requests) / max(.001, result.step_ms)

        # Streaming rows and newly retained rows in ONE forward share the
        # same layer launch; do not add two independent reconstruction calls.
        forward = list(selected) if selected else list(running)
        forward_dirs = list(dict.fromkeys(r.adapter_dir for r in forward))
        ready = self.prefix_cache_state.get("ready_layers", {})
        result.first_forward_reconstruction_ms = self.cost.layer_intervals_ms(
            [(min(targets[d], ready.get(d, 0)), self.layer_num, ranks[d])
             for d in forward_dirs], self.prefix_len)
        result.first_forward_ms = result.first_forward_reconstruction_ms + self.cost.core_ms(
            len(forward), sum(self.known_tokens(r) for r in forward),
            sum(ranks[r.adapter_dir] for r in forward), self.prefix_len, prefill=bool(selected))
        if selected:
            new_dirs = set(forward_dirs) - {r.adapter_dir for r in running}
            result.first_forward_ms += self.cost.load_ms_per_slot * sum(4 * ranks[d] for d in new_dirs)
        return result

    def _stability(self, choice, current):
        ready = self.prefix_cache_state.get("ready_layers", {})
        return (sum(min(k, ready.get(d, 0)) for d, k in choice.layers.items()),
                -sum(abs(k - current[d]) for d, k in choice.layers.items()),
                -choice.first_forward_ms, -choice.plan.required_slots)

    def _optimize_layout(self, running, selected, ranks, current, old):
        serial = dict.fromkeys(current, 0)
        fallback = self._evaluate(running, selected, ranks, serial)
        if not fallback.plan.fits:
            return fallback
        if not self.cost.calibrated:
            return old if old.plan.fits else fallback
        dirs = list(current)
        if not dirs:
            return fallback
        state = self.prefix_cache_state
        idle = set(state.get("inactive_lora_dirs", ())) - set(dirs) - set(fallback.plan.evictions)
        reclaimable = sum(state.get("prefix_sizes", {}).get(d, 0) for d in idle)
        # A feasible serial plan may have left idle caches in the pool. They
        # are available for useful active retention; _layout chooses actual
        # LRU evictions only after the candidate layout is known.
        fixed = (fallback.plan.required_slots
                 - prefix_slots(serial.values(), self.prefix_len, self.layer_num) - reclaimable)
        self.layout_budget_slots = self.max_total_tokens - fixed
        a, b, c = self.cost.coef["reconstruct"]
        weights = [b * self.prefix_len / 10000.
                   + c * tiled_rank(ranks[d]) * self.prefix_len / 160000. for d in dirs]
        # Below the model's numerical floor, group cost is not affine.
        self.solver_exact = bool(self.prefix_len > 0 and self.prefix_len % self.layer_num == 0
                                 and a + min(weights) >= .001)
        ready = state.get("ready_layers", {})
        seeds = [tuple(current.values()), (0,) * len(dirs)]
        seeds.extend(local_layout_candidates(weights, self.layer_num, self.prefix_len,
                                             self.layout_budget_slots,
                                             tuple(ready.get(d, 0) for d in dirs)))
        best, seen = fallback, set()
        for seed in seeds:
            if seed in seen:
                continue
            seen.add(seed)
            candidate = self._evaluate(running, selected, ranks, dict(zip(dirs, seed)))
            self.search_evaluations += 1
            if not candidate.plan.fits:
                continue
            equal = math.isclose(candidate.reconstruction_ms, best.reconstruction_ms,
                                 rel_tol=1e-9, abs_tol=1e-9)
            if ((equal and self._stability(candidate, current) > self._stability(best, current))
                    or (not equal and candidate.reconstruction_ms < best.reconstruction_ms)):
                best = candidate
        # Optional layout-only hysteresis. The selected FCFS prefix is fixed;
        # neither this margin nor a bad cost estimate can cancel admission.
        if (old.plan.fits and self.config.hysteresis
                and best.reconstruction_ms >= old.reconstruction_ms * (1 - self.config.hysteresis)):
            best = old
        return best

    def choose(self, running, ranks, allow_admission=True, now=None):
        start = time.perf_counter()
        running = list(running)
        waiting = ([r for r in self.waiting_req_list if not r.aborted] if allow_admission else [])
        self.search_evaluations = self.allocator_checks = 0
        self.layout_budget_slots, self.solver_exact = 0, False
        selected = self._admit_fcfs(running, waiting, ranks) if waiting else []
        current = self._targets(running, selected)
        old = self._evaluate(running, selected, ranks, current)
        membership = frozenset((r.request_id, r.adapter_dir) for r in running + selected)
        reasons = []
        # An idle-server reset may reuse request IDs from a warmup. A real
        # admission is always an event, even when that signature matches.
        if selected or membership != self.last_membership:
            reasons.append("batch_changed")
        if not old.plan.fits:
            reasons.append("memory_pressure")
        if self.cost.calibrated and not self.was_calibrated:
            reasons.append("model_calibrated")
        if self.last_capacity is not None and self.max_total_tokens != self.last_capacity:
            reasons.append("capacity_changed")
        choice = (self._optimize_layout(running, selected, ranks, current, old) if reasons else old)
        self.pending = choice if choice.plan.fits else None
        self.last_membership, self.last_capacity = membership, self.max_total_tokens
        self.was_calibrated = self.cost.calibrated
        if reasons:
            self.last_heavy_step = self.decode_steps
        self.decisions += 1
        self.last_record = dict(
            event="space_plan", policy=POLICY, policy_revision=POLICY_REVISION,
            objective="fixed_batch_steady_decode", admission_policy="fcfs_max_feasible",
            decode_step=self.decode_steps, heavy=bool(reasons), trigger_reasons=reasons,
            calibrated=self.cost.calibrated, urgency_override=False, fits=choice.plan.fits,
            window=1, selected=[r.request_id for r in choice.selected],
            running=len(running), waiting_considered=len(waiting), retained_layers=choice.layers,
            required_slots=choice.plan.required_slots, capacity_slots=self.max_total_tokens,
            evictions=choice.plan.evictions, predicted_progress_tokens_s=choice.score,
            predicted_step_ms=choice.step_ms, predicted_reconstruction_ms=choice.reconstruction_ms,
            predicted_first_forward_ms=choice.first_forward_ms,
            predicted_first_forward_reconstruction_ms=choice.first_forward_reconstruction_ms,
            step_slo_feasible=True, planning_ms=(time.perf_counter() - start) * 1000.,
            search_evaluations=self.search_evaluations, allocator_checks=self.allocator_checks,
            admission_candidates=[len(selected)], layout_budget_slots=self.layout_budget_slots,
            solver="common_floor", solver_exact_model=self.solver_exact,
            layout_changed=choice.layers != current,
            model=self.cost.state_dict(), head_pressure=None)
        return choice
