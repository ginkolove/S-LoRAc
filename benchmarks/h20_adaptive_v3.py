"""H20 V3: price layout changes after resource-feasible FCFS admission."""
from collections import deque
import math
from statistics import median
import time

from .h20_adaptive_v2 import AdaptiveV2ReqQueue, local_layout_candidates, prefix_slots, tiled_rank


POLICY = "minikv-adaptive-v3"
POLICY_REVISION = "fcfs-layout-payback-v1"
OBJECTIVE = "fixed_batch_net_reconstruction_savings"


class AdaptiveV3ReqQueue(AdaptiveV2ReqQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_dirs = None
        self.active_since = 0
        self.mean_group_steps = 2.
        self.recheck_at = None
        self.apply_seq = 0
        self.apply_unit_samples = deque(maxlen=32)
        self.apply_ms_per_dir = 0.

    def update_prefix_cache_state(self, state):
        super().update_prefix_cache_state(state)
        for sample in self.prefix_cache_state.get("v3_apply_samples", ()):
            if sample["seq"] <= self.apply_seq:
                continue
            self.apply_seq = sample["seq"]
            # Host submission and CUDA work overlap. Neither time includes a forward.
            elapsed = max(sample["host_ms"], sample["device_ms"])
            if sample["changed_dirs"] and math.isfinite(elapsed) and elapsed > 0:
                self.apply_unit_samples.append(elapsed / sample["changed_dirs"])
                self.apply_ms_per_dir = median(self.apply_unit_samples)

    def _group_horizon(self, dirs, new_busy_period):
        # Observed decode steps only. An unusually long live group can outgrow
        # the historical estimate. Prefill-only bursts consume zero decode steps.
        if new_busy_period or self.active_dirs is None:
            self.active_since, self.mean_group_steps = self.decode_steps, 2.
        elif dirs != self.active_dirs:
            duration = min(self.config.window, self.decode_steps - self.active_since)
            self.mean_group_steps = .75 * self.mean_group_steps + .25 * duration
            self.active_since = self.decode_steps
        self.active_dirs = dirs
        age = self.decode_steps - self.active_since
        return min(self.config.window, max(1, math.ceil(self.mean_group_steps), age))

    def _apply_cost(self, choice, current):
        changed = sum(k != current[d] for d, k in choice.layers.items())
        return (changed + len(choice.plan.evictions)) * self.apply_ms_per_dir

    def _initial_reconstruction(self, choice, prefill_dirs, ranks):
        """Optional prefill and the first merged decode, each with its actual sources."""
        total = choice.first_forward_reconstruction_ms
        if prefill_dirs:
            ready = self.prefix_cache_state.get("ready_layers", {})
            # Only adapters in the new mini-batch had their retained KV built
            # by prefill. Running-only promotions first materialize during decode.
            intervals = [(k if d in prefill_dirs else min(k, ready.get(d, 0)),
                          self.layer_num, ranks[d]) for d, k in choice.layers.items()]
            total += self.cost.layer_intervals_ms(intervals, self.prefix_len)
        return total

    def _choose_layout(self, running, selected, ranks, current, old, group_steps, record):
        serial = self._evaluate(running, selected, ranks, dict.fromkeys(current, 0))
        reference = old if old.plan.fits else serial
        self.recheck_at = None
        record["payback_reference"] = "current" if old.plan.fits else "all_serial"
        if not serial.plan.fits or not current or not self.cost.calibrated:
            return reference

        dirs = list(current)
        count = len(running) + len(selected)
        state = self.prefix_cache_state
        idle = set(state.get("inactive_lora_dirs", ())) - set(dirs) - set(serial.plan.evictions)
        reclaimable = sum(state.get("prefix_sizes", {}).get(d, 0) for d in idle)
        serial_slots = prefix_slots(serial.layers.values(), self.prefix_len, self.layer_num)
        budget = self.max_total_tokens - serial.plan.required_slots + serial_slots + reclaimable
        record["layout_budget_slots"] = budget

        def runway(layers):
            # First decode already fits. Each following decode adds one private slot/request.
            spare = budget - prefix_slots(layers, self.prefix_len, self.layer_num)
            return 1 + spare // count

        max_steps = min(group_steps, runway(reference.layers.values()))
        prefill_dirs = {r.adapter_dir for r in selected}
        reference_initial = self._initial_reconstruction(reference, prefill_dirs, ranks)
        reference_apply = self._apply_cost(reference, current)
        _, b, c = self.cost.coef["reconstruct"]
        weights = [self.prefix_len * (b / 10000. + c * tiled_rank(ranks[d]) / 160000.)
                   for d in dirs]
        ready = tuple(state.get("ready_layers", {}).get(d, 0) for d in dirs)
        best, best_gain = reference, 0.
        fastest_steady = reference.reconstruction_ms
        record.update(reuse_steps=max_steps, predicted_apply_ms=reference_apply,
                      predicted_net_saving_ms=0.)
        seen = {tuple(reference.layers.values())}
        # Consider a layout that leaves room for expected private KV growth as
        # well as the immediate capacity frontier. Admission is already fixed.
        for slots in dict.fromkeys((budget, budget - (max_steps - 1) * count)):
            for seed in local_layout_candidates(weights, self.layer_num, self.prefix_len, slots, ready):
                if seed in seen:
                    continue
                seen.add(seed)
                choice = self._evaluate(running, selected, ranks, dict(zip(dirs, seed)))
                self.search_evaluations += 1
                if not choice.plan.fits:
                    continue
                fastest_steady = min(fastest_steady, choice.reconstruction_ms)
                steps = min(max_steps, runway(seed))
                initial = self._initial_reconstruction(choice, prefill_dirs, ranks)
                apply = self._apply_cost(choice, current)
                # Compare both layouts for identical forwards. After either
                # would face pressure we credit no further saving.
                saving = (reference_initial - initial
                          + (steps - 1) * (reference.reconstruction_ms - choice.reconstruction_ms))
                gain = saving + reference_apply - apply
                if old.plan.fits and gain <= self.config.hysteresis * (
                        reference_initial + (steps - 1) * reference.reconstruction_ms):
                    continue
                if (gain > best_gain + 1e-9 or
                        (math.isclose(gain, best_gain, abs_tol=1e-9) and best is not old
                         and self._stability(choice, current) > self._stability(best, current))):
                    best, best_gain = choice, gain
                    record.update(reuse_steps=steps, predicted_apply_ms=apply,
                                  predicted_net_saving_ms=gain)
        if best.reconstruction_ms > fastest_steady + 1e-9 and group_steps < self.config.window:
            self.recheck_at = self.decode_steps + group_steps
        record["fastest_candidate_steady_ms"] = fastest_steady
        return best

    def choose(self, running, ranks, allow_admission=True, now=None):
        start = time.perf_counter()
        running = list(running)
        waiting = [r for r in self.waiting_req_list if not r.aborted] if allow_admission else []
        self.search_evaluations = self.allocator_checks = 0
        selected = self._admit_fcfs(running, waiting, ranks) if waiting else []
        current = self._targets(running, selected)
        old = self._evaluate(running, selected, ranks, current)
        group_steps = self._group_horizon(frozenset(current), not running)
        membership = frozenset((r.request_id, r.adapter_dir) for r in running + selected)
        reasons = []
        if selected or membership != self.last_membership:
            reasons.append("batch_changed")
        if not old.plan.fits:
            reasons.append("memory_pressure")
        if self.cost.calibrated and not self.was_calibrated:
            reasons.append("model_calibrated")
        if self.last_capacity is not None and self.max_total_tokens != self.last_capacity:
            reasons.append("capacity_changed")
        if self.recheck_at is not None and self.decode_steps >= self.recheck_at:
            reasons.append("reuse_recheck")
        record = dict(group_horizon_steps=group_steps, reuse_steps=None, payback_reference=None,
                      layout_budget_slots=None, predicted_net_saving_ms=None,
                      predicted_apply_ms=None, apply_ms_per_changed_dir=self.apply_ms_per_dir,
                      apply_sample_count=len(self.apply_unit_samples))
        apply_samples = self.prefix_cache_state.get("v3_apply_samples")
        if apply_samples:
            record["last_measured_apply"] = apply_samples[-1]
        choice = old
        if reasons:
            choice = self._choose_layout(running, selected, ranks, current, old, group_steps, record)
        self.pending = choice if choice.plan.fits else None
        self.last_membership, self.last_capacity = membership, self.max_total_tokens
        self.was_calibrated = self.cost.calibrated
        if reasons:
            self.last_heavy_step = self.decode_steps
        self.decisions += 1
        self.last_record = dict(
            event="space_plan", policy=POLICY, policy_revision=POLICY_REVISION,
            objective=OBJECTIVE, admission_policy="fcfs_max_feasible", decode_step=self.decode_steps,
            heavy=bool(reasons), trigger_reasons=reasons, calibrated=self.cost.calibrated,
            fits=choice.plan.fits, selected=[r.request_id for r in selected], running=len(running),
            waiting_considered=len(waiting), retained_layers=choice.layers, evictions=choice.plan.evictions,
            required_slots=choice.plan.required_slots, capacity_slots=self.max_total_tokens,
            layout_changed=choice.layers != current, predicted_progress_tokens_s=choice.score,
            predicted_step_ms=choice.step_ms, predicted_reconstruction_ms=choice.reconstruction_ms,
            predicted_first_forward_ms=choice.first_forward_ms,
            predicted_first_forward_reconstruction_ms=choice.first_forward_reconstruction_ms,
            search_evaluations=self.search_evaluations, allocator_checks=self.allocator_checks,
            solver="common_floor_payback", solver_exact_model=False, model=self.cost.state_dict(),
            planning_ms=(time.perf_counter() - start) * 1000., **record)
        return choice
