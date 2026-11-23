"""Full-first, FCFS MiniKV admission with rank-grouped retention and measured feedback."""
from copy import copy
from dataclasses import dataclass
import math
import time

from slora.server.router.adaptive_req_queue import AdaptiveReqQueue, SpaceDecision
from slora.server.router.online_req_queue import OnlineReqQueue


POLICY = "minikv-adaptive-v4"
POLICY_REVISION = "full-first-rank-feedback-v1"


@dataclass
class LayoutTrial:
    before: dict
    baseline: float
    signature: tuple
    low_windows: int = 0


class AdaptiveV4ReqQueue(AdaptiveReqQueue):
    def __init__(self, *args, space_plan_config=None, **kwargs):
        options = dict(space_plan_config or {})
        self.feedback_seconds = options.pop("feedback_seconds", 1.0)
        if not math.isfinite(self.feedback_seconds) or self.feedback_seconds <= 0:
            raise ValueError("V4 feedback window must be positive and finite")
        super().__init__(*args, space_plan_config=options, **kwargs)
        self.reset_feedback()

    def reset_feedback(self):
        self.reserve = 0
        self.seed_reserve = True
        self.last_capacity = None
        self.members = set()
        self.release_pending = False
        self.trial = self.rollback = self.veto = None
        self.window_start = self.last_rate = None
        self.output_count = 0
        self.pending = None
        self.last_record = None

    def update_prefix_cache_state(self, state):
        # V4 uses output tokens per wall-clock second, independent of CUDA cost fits.
        OnlineReqQueue.update_prefix_cache_state(self, state)

    def observe_output(self, count):
        if self.seed_reserve and count:
            # Seed headroom from one observed output batch. Later preemptions refine it.
            self.reserve = max(self.reserve, count)
            self.seed_reserve = False
        self.output_count += count

    def observe_preemption(self, deficit):
        self.reserve = max(self.reserve, deficit)

    def _new_window(self, now):
        self.window_start = now
        self.output_count = 0

    def _sample(self, now, reasons):
        if self.window_start is None:
            self._new_window(now)
        elapsed = now - self.window_start
        if elapsed < self.feedback_seconds:
            return
        self.last_rate = self.output_count / elapsed
        self._new_window(now)
        if self.trial:
            if self.last_rate < self.trial.baseline * (1 - self.config.hysteresis):
                self.trial.low_windows += 1
                if self.trial.low_windows >= 2:
                    self.rollback = self.trial
                    self.veto = self.trial.signature
                    self.trial = None
                    reasons.append("throughput_drop")
            else:
                self.trial = None
                reasons.append("trial_accepted")

    def _current_levels(self, running, ranks):
        retained = self.prefix_cache_state.get("retained_layers", {})
        levels = {}
        for req in running:
            rank = ranks[req.adapter_dir]
            layers = retained.get(req.adapter_dir, self.layer_num)
            levels[rank] = min(levels.get(rank, self.layer_num), layers)
        return levels

    def _choice(self, running, selected, ranks, levels):
        layers = {r.adapter_dir: levels.get(ranks[r.adapter_dir], self.layer_num)
                  for r in [*running, *selected]}
        plan = self._layout(running, selected, ranks, layers)
        return SpaceDecision(selected, layers, plan, 1)

    def _pack(self, running, waiting, ranks, levels, initial):
        """Longest feasible FCFS prefix at this layout. Existing privately-held KV is counted once."""
        best = initial
        lo = len(best.selected) + 1
        hi = min(len(waiting), self.running_max_req_size - len(running))
        while lo <= hi:
            n = (lo + hi) // 2
            probe = self._choice(running, waiting[:n], ranks, levels)
            if probe.plan.fits:
                best, lo = probe, n + 1
            else:
                hi = n - 1
        return best

    def _highest_partial(self, running, selected, ranks, levels, rank, lower, upper, best):
        # Below full retention the scratch set stays fixed and required slots are monotone.
        while lower <= upper:
            k = (lower + upper) // 2
            probe = self._choice(running, selected, ranks, {**levels, rank: k})
            if probe.plan.fits:
                best, lower = probe, k + 1
            else:
                upper = k - 1
        return best

    def _shrink(self, running, selected, ranks, levels):
        active = {ranks[r.adapter_dir] for r in [*running, *selected]}
        serial = self._choice(running, selected, ranks, dict.fromkeys(active, 0))
        if not serial.plan.fits:
            return serial
        reduced = levels.copy()
        for rank in sorted(active):
            old = reduced.get(rank, self.layer_num)
            if old == 0:
                continue
            reduced[rank] = 0
            zero = self._choice(running, selected, ranks, reduced)
            if zero.plan.fits:
                return self._highest_partial(running, selected, ranks, reduced, rank,
                                             1, old - 1, zero)
        return serial

    def _promote(self, running, ranks, levels):
        # Probe the same cache and LRU order with a smaller capacity. Admission retains the physical limit.
        planner = copy(self)
        planner.max_total_tokens -= self.reserve
        for rank in sorted(levels, reverse=True):
            old = levels[rank]
            if old == self.layer_num:
                continue
            full = planner._choice(running, [], ranks, {**levels, rank: self.layer_num})
            if full.plan.fits:
                return full
            partial = planner._highest_partial(running, [], ranks, levels, rank, old + 1,
                                               self.layer_num - 1, None)
            if partial:
                return partial
        return None

    def _start_trial(self, choice, ranks, levels, action, head, now):
        before = {ranks[d]: levels.get(ranks[d], self.layer_num) for d, k in choice.layers.items()
                  if k != levels.get(ranks[d], self.layer_num)}
        signature = (action, head, frozenset(choice.layers.items()))
        if not before or signature == self.veto:
            return False
        if self.last_rate is not None:
            self.trial = LayoutTrial(before, self.last_rate, signature)
        self.last_rate = None
        self._new_window(now)
        return True

    def _admission(self, running, waiting, ranks, levels, base, now, reasons):
        selected = self._pack(running, waiting, ranks, levels, base)
        n = len(selected.selected)
        if n == len(waiting):
            return selected
        # A runnable batch supplies a baseline before changing its retention.
        if self.trial or (self.last_rate is None and (running or n)):
            return selected
        ready = self._shrink(running, waiting[:n + 1], ranks, levels)
        if not ready.plan.fits:
            return selected
        if not self._start_trial(ready, ranks, levels, "shrink", waiting[n].adapter_dir, now):
            return selected
        reasons.append("admission_pressure")
        new_levels = {ranks[d]: k for d, k in ready.layers.items()}
        return self._pack(running, waiting, ranks, new_levels, ready)

    def choose(self, running, ranks, allow_admission=True, now=None):
        now = time.monotonic() if now is None else now
        running = list(running)
        if not running and self.members:
            self.reset_feedback()
        reasons = []
        prior = self.trial or self.rollback
        reference = prior.baseline if prior else None
        self._sample(now, reasons)
        observed = self.last_rate
        waiting = [r for r in self.waiting_req_list if not r.aborted]
        active_ids = {r.request_id for r in running}
        self.release_pending |= bool(self.members - active_ids)
        if self.last_capacity is not None and self.last_capacity != self.max_total_tokens:
            self.release_pending |= self.max_total_tokens > self.last_capacity
            self.trial = self.rollback = self.veto = self.last_rate = None
            self._new_window(now)
            reasons.append("capacity_changed")
        self.last_capacity = self.max_total_tokens
        # One prompt token and the first output, plus the smallest configured adapter.
        smallest_rank = min((rank for directory, rank in ranks.items() if directory is not None), default=0)
        serial_only = self.max_total_tokens < self.prefix_len + 4 * smallest_rank + 2
        levels = dict.fromkeys(ranks.values(), 0) if serial_only else self._current_levels(running, ranks)
        choice = self._choice(running, [], ranks, levels)
        if not choice.plan.fits and not serial_only:
            # Already admitted private histories grew. The router preempts only if all-serial also fails.
            choice = self._shrink(running, [], ranks, levels)
            self.trial = None
            self.last_rate = None
            self._new_window(now)
            reasons.append("running_pressure")
        elif self.rollback and not serial_only:
            restore = self._choice(running, [], ranks, {**levels, **self.rollback.before})
            if restore.plan.fits:
                choice = restore
                self.rollback = None
                self.last_rate = None
                self._new_window(now)
                reasons.append("rollback")
            else:
                reasons.append("rollback_deferred")
        elif allow_admission and waiting and choice.plan.fits:
            if serial_only:
                choice = self._pack(running, waiting, ranks, levels, choice)
            else:
                choice = self._admission(running, waiting, ranks, levels, choice, now, reasons)
        elif (not waiting and running and self.release_pending and not serial_only
              and self.trial is None and self.last_rate is not None):
            self.release_pending = False
            promotion = self._promote(running, ranks, levels)
            if promotion and self._start_trial(promotion, ranks, levels, "promote", None, now):
                choice = promotion
                reasons.append("capacity_released")
        if serial_only:
            reasons.append("serial_fallback")
        old = self.prefix_cache_state.get("retained_layers", {})
        changed = choice.plan.fits and any(k != old.get(d, 0) for d, k in choice.layers.items())
        if changed and not running and not reasons:
            reasons.append("initial_full")
        elif changed and not reasons and choice.selected:
            reasons.append("group_join")
        self.members = active_ids | {r.request_id for r in choice.selected}
        self.pending = choice if choice.plan.fits else None
        self.last_record = dict(event="space_plan", policy=POLICY, policy_revision=POLICY_REVISION,
                                trigger_reasons=reasons, layout_changed=changed, fits=choice.plan.fits,
                                running=len(running), selected=[r.request_id for r in choice.selected],
                                waiting_considered=len(waiting) if allow_admission else 0,
                                retained_layers=choice.layers, evictions=choice.plan.evictions,
                                capacity_slots=self.max_total_tokens, required_slots=choice.plan.required_slots,
                                reserve_slots=self.reserve, serial_only=serial_only, output_tokens_s=observed,
                                feedback_seconds=self.feedback_seconds, trial_active=self.trial is not None,
                                reference_tokens_s=self.trial.baseline if self.trial else reference,
                                rollback_pending=self.rollback is not None)
        return choice
