"""Experimental measured-cost admission and per-prefix retained-layer planning.

No reference output lengths, generation caps, or future arrivals are consulted.
The finite-horizon progress objective is a surrogate, not a global optimum.
"""

from collections import deque
from dataclasses import dataclass
import math
import time
import uuid

import numpy as np

from slora.common.space_plan_config import SpacePlanConfig
from slora.server.io_struct import Batch
from slora.server.router.online_req_queue import OnlinePlan, OnlineReqQueue


class MeasuredCostModel:
    """Small nonnegative regressions, fitted only to completed CUDA samples."""
    def __init__(self):
        self.samples = {name: deque(maxlen=96) for name in ("decode", "prefill", "reconstruct")}
        self.coef = {}
        self.last_seq = 0
        self.error = 0.2
        self.load_ms_per_slot = 0.0

    @staticmethod
    def core_features(batch, tokens, rank_sum, prefix, prefill=False):
        work = tokens if prefill else batch
        return np.asarray([1., work / 64., (tokens + prefix * work) / 1e6, rank_sum / 1024.])

    @staticmethod
    def reconstruction_features(unique, rank_sum, prefix):
        return np.asarray([1., unique * prefix / 10000., rank_sum * prefix / 160000.])

    def update(self, rows):
        dirty = set()
        for row in rows:
            if row["seq"] <= self.last_seq:
                continue
            self.last_seq = row["seq"]
            # Avoid initial model/kernel warmup contaminating the first fit.
            if row["seq"] <= 2:
                continue
            name = "prefill" if row["prefill"] else "decode"
            x = self.core_features(row["batch"], row["private_tokens"], row["rank_sum"],
                                   row["prefix_len"], row["prefill"])
            y = float(row["core_ms"])
            if not math.isfinite(y) or y <= 0:
                continue
            if name in self.coef:
                error = abs(float(x @ self.coef[name]) - y) / y
                self.error = 0.8 * self.error + 0.2 * min(error, 1.0)
            self.samples[name].append((x, y))
            dirty.add(name)
            for record in row["reconstruction"]:
                y = float(record["ms"])
                if math.isfinite(y) and y > 0:
                    x = self.reconstruction_features(record["unique"], record["rank_sum"], row["prefix_len"])
                    self.samples["reconstruct"].append((x, y))
                    dirty.add("reconstruct")
        for name in dirty:
            x = np.asarray([row[0] for row in self.samples[name]])
            y = np.asarray([row[1] for row in self.samples[name]])
            gram = x.T @ x + np.eye(x.shape[1]) * 0.01
            rhs = x.T @ y
            coef = self.coef.get(name, np.zeros(x.shape[1])).copy()
            # Coordinate descent for nonnegative ridge least squares.
            for _ in range(24):
                for j in range(len(coef)):
                    coef[j] = max(0., coef[j] + (rhs[j] - gram[j] @ coef) / gram[j, j])
            self.coef[name] = coef

    @property
    def calibrated(self):
        return len(self.samples["decode"]) >= 3 and len(self.samples["reconstruct"]) >= 3

    def core_ms(self, batch, tokens, rank_sum, prefix, prefill=False):
        name = "prefill" if prefill else "decode"
        features = self.core_features(batch, tokens, rank_sum, prefix, prefill)
        if name not in self.coef:
            # Cold start is explicitly reported; no retention optimization yet.
            return max(1., tokens / 100. if prefill else batch)
        return max(0.001, float(features @ self.coef[name]))

    def reconstruction_ms(self, ranks, prefix):
        if not ranks:
            return 0.
        return self.reconstruction_group_ms(len(ranks), sum(ranks), prefix)

    def reconstruction_group_ms(self, count, rank_sum, prefix):
        if not count:
            return 0.
        if "reconstruct" not in self.coef:
            return 0.1 * count
        a, b, c = self.coef["reconstruct"]
        return max(0.001, float(a + b * count * prefix / 10000. + c * rank_sum * prefix / 160000.))

    def layer_intervals_ms(self, intervals, prefix):
        events = {}
        for first, end, rank in intervals:
            if end <= first:
                continue
            for layer, sign in ((first, 1), (end, -1)):
                counts = events.setdefault(layer, [0, 0])
                counts[0] += sign
                counts[1] += sign * rank
        count = ranks = previous = 0
        total = 0.
        for layer in sorted(events):
            total += (layer - previous) * self.reconstruction_group_ms(count, ranks, prefix)
            delta = events[layer]
            count += delta[0]
            ranks += delta[1]
            previous = layer
        return total

    def state_dict(self):
        return dict(calibrated=self.calibrated, relative_error=self.error,
                    samples={name: len(rows) for name, rows in self.samples.items()},
                    coefficients={name: value.tolist() for name, value in self.coef.items()})


@dataclass
class SpaceDecision:
    selected: list
    layers: dict
    plan: OnlinePlan
    window: int
    score: float = 0.
    step_ms: float = 0.
    slo_feasible: bool = True


class AdaptiveReqQueue(OnlineReqQueue):
    def __init__(self, *args, space_plan_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = SpacePlanConfig(**(space_plan_config or {}))
        self.cost = MeasuredCostModel()
        self.decode_steps = 0
        self.last_heavy_step = -self.config.interval
        self.pending = None
        self.last_record = None
        self.last_signature = None
        self.decisions = 0

    def update_prefix_cache_state(self, state):
        super().update_prefix_cache_state(state)
        self.cost.update(self.prefix_cache_state.get("adaptive_profile", ()))

    def _layout(self, running, selected, ranks, targets, window=1):
        requests = list(running) + list(selected)
        needed = {r.adapter_dir for r in requests}
        if None in needed:
            raise ValueError("adaptive planning requires LoRA requests")
        sizes = dict(self.prefix_cache_state.get("prefix_sizes", {}))
        for directory in needed:
            sizes[directory] = (targets.get(directory, 0) * self.prefix_len + self.layer_num - 1) // self.layer_num
        scratch = sum(targets.get(d, 0) < self.layer_num for d in needed)
        scratch_slots = (scratch * self.prefix_len + self.layer_num - 1) // self.layer_num
        private = sum(self.known_tokens(r) for r in requests) + len(selected)
        growth = (window - 1) * len(requests)
        adapters = sum(4 * ranks[d] for d in needed)
        required = private + growth + adapters + scratch_slots + sum(sizes.values())
        evictions = []
        inactive = set(self.prefix_cache_state.get("inactive_lora_dirs", ())) - needed
        used = self.prefix_cache_state.get("last_used", {})
        for directory in sorted(inactive, key=lambda d: (used.get(d, 0), d)):
            if required <= self.max_total_tokens:
                break
            required -= sizes.pop(directory, 0)
            evictions.append(directory)
        prefill = sum(self.known_tokens(r) for r in selected)
        fits = (required <= self.max_total_tokens and prefill <= self.batch_max_tokens
                and len(requests) <= self.running_max_req_size
                and (self.max_active_loras is None or len(needed) <= self.max_active_loras)
                and (self.workspace_lora_limit is None or len(needed) <= self.workspace_lora_limit))
        return OnlinePlan(fits, required, evictions, prefill, 0)

    def plan(self, running, selected, lora_ranks):
        layers = self.prefix_cache_state.get("retained_layers", {})
        return self._layout(running, selected, lora_ranks, layers)

    def _evaluate(self, running, selected, ranks, targets, window):
        plan = self._layout(running, selected, ranks, targets, window)
        result = SpaceDecision(list(selected), dict(targets), plan, window)
        if not plan.fits:
            return result
        requests = list(running) + list(selected)
        if not requests:
            return result
        dirs = list(dict.fromkeys(r.adapter_dir for r in requests))
        private = sum(self.known_tokens(r) for r in requests)
        rank_sum = sum(ranks[r.adapter_dir] for r in requests)
        count = len(requests)
        core = self.cost.core_ms(count, private + (window - 1) * count / 2,
                                 rank_sum, self.prefix_len)
        rebuild = self.cost.layer_intervals_ms(
            [(targets.get(d, 0), self.layer_num, ranks[d]) for d in dirs], self.prefix_len)
        ready = self.prefix_cache_state.get("ready_layers", {})
        build = self.cost.layer_intervals_ms(
            [(ready.get(d, 0), targets.get(d, 0), ranks[d]) for d in dirs], self.prefix_len)
        admit = 0.
        if selected:
            admit = self.cost.core_ms(len(selected), plan.prefill_tokens,
                sum(ranks[r.adapter_dir] for r in selected), self.prefix_len, prefill=True)
            selected_dirs = set(r.adapter_dir for r in selected)
            admit += self.cost.layer_intervals_ms(
                [(targets.get(d, 0), self.layer_num, ranks[d]) for d in selected_dirs], self.prefix_len)
            new_dirs = set(r.adapter_dir for r in selected) - set(r.adapter_dir for r in running)
            admit += self.cost.load_ms_per_slot * sum(4 * ranks[d] for d in new_dirs)
        result.step_ms = core + rebuild
        result.score = 1000. * window * count / max(0.001, admit + build + window * result.step_ms)
        result.slo_feasible = (not self.config.max_step_ms or
                               max(result.step_ms, admit + build) <= self.config.max_step_ms)
        return result

    @staticmethod
    def _order(choice):
        return choice.slo_feasible, choice.score

    def _search(self, running, selected, ranks, window, deadline, heavy):
        dirs = list(dict.fromkeys(r.adapter_dir for r in list(running) + list(selected)))
        current = self.prefix_cache_state.get("retained_layers", {})
        seeds = [{d: current.get(d, 0) for d in dirs}]
        if heavy:
            seeds += [{d: 0 for d in dirs}, {d: self.layer_num for d in dirs}]
        if not self.cost.calibrated:
            seeds = [{d: 0 for d in dirs}]
        seen, beam, best = set(), [], None
        for seed in seeds:
            choice = self._evaluate(running, selected, ranks, seed, window)
            if choice.plan.fits:
                beam.append(choice)
                if best is None or self._order(choice) > self._order(best):
                    best = choice
        if not heavy or not self.cost.calibrated:
            return best
        # Search one-step changes and swaps; all-serial is always a feasible seed
        # whenever this candidate request set can fit at all.
        for group in (self.config.group, 1):
            while beam and time.perf_counter() < deadline:
                proposals = []
                for parent in beam[:self.config.beam]:
                    changes = []
                    for d in dirs:
                        value = parent.layers[d]
                        for target in {max(0, value - group), min(self.layer_num, value + group), 0, self.layer_num}:
                            if target != value:
                                new = dict(parent.layers, **{d: target})
                                changes.append(new)
                    # Bounded pair swaps allow spending reclaimed bytes elsewhere.
                    for a in dirs[:8]:
                        for b in dirs[:8]:
                            if a != b and parent.layers[a] >= group and parent.layers[b] + group <= self.layer_num:
                                new = dict(parent.layers)
                                new[a] -= group
                                new[b] += group
                                changes.append(new)
                    for new in changes:
                        if time.perf_counter() >= deadline:
                            break
                        key = tuple(new[d] for d in dirs)
                        if key in seen:
                            continue
                        seen.add(key)
                        choice = self._evaluate(running, selected, ranks, new, window)
                        if choice.plan.fits:
                            proposals.append(choice)
                            if best is None or self._order(choice) > self._order(best):
                                best = choice
                if not proposals:
                    break
                beam = sorted(proposals, key=self._order, reverse=True)[:self.config.beam]
        return best

    def choose(self, running, ranks, allow_admission=True, now=None):
        now = time.time() if now is None else now
        start = time.perf_counter()
        deadline = start + self.config.budget_ms / 1000.
        running = list(running)
        waiting = [r for r in self.waiting_req_list if not r.aborted][:self.config.candidates] if allow_admission else []
        urgent = bool(waiting and self.config.max_wait_ms and
                      (now - getattr(waiting[0], "arrival_time", now)) * 1000 >= self.config.max_wait_ms)
        current = self.plan(running, [], ranks)
        heavy = (not running or not current.fits or urgent or
                 self.decode_steps - self.last_heavy_step >= self.config.interval)
        window = self.config.window if heavy else max(1, self.config.window - (self.decode_steps - self.last_heavy_step))
        candidates = []
        # FCFS prefixes only. Test all lengths for capacity before spending time
        # on optimization, so a costly search cannot hide a feasible old request.
        for n in range(len(waiting) + 1):
            selected = waiting[:n]
            targets = {r.adapter_dir: 0 for r in running + selected}
            feasible = self._layout(running, selected, ranks, targets, window)
            if not feasible.fits:
                break
            candidates.append(selected)
        if not running and waiting and len(candidates) == 1:
            # A ten-step reserve must not reject a prompt that can make progress
            # with a one-step window. Output length is still not consulted.
            window = 1
            for n in range(1, len(waiting) + 1):
                selected = waiting[:n]
                targets = {r.adapter_dir: 0 for r in selected}
                if not self._layout([], selected, ranks, targets, window).fits:
                    break
                candidates.append(selected)
        if not candidates:
            window = 1
            heavy = True
            targets = {r.adapter_dir: 0 for r in running}
            choice = self._evaluate(running, [], ranks, targets, 1)
        else:
            if not self.cost.calibrated:
                selected = candidates[-1]
                choice = self._evaluate(running, selected, ranks,
                                        {r.adapter_dir: 0 for r in running + selected}, window)
            else:
                choices = []
                # Evaluate endpoints first, then every feasible FCFS prefix.
                indices = list(dict.fromkeys([0, len(candidates) - 1, *range(len(candidates))]))
                for index in indices:
                    selected = candidates[index]
                    result = self._search(running, selected, ranks, window, deadline, heavy=False)
                    if result is None and heavy:
                        result = self._evaluate(running, selected, ranks,
                            {r.adapter_dir: 0 for r in running + selected}, window)
                    if result is not None and result.plan.fits:
                        choices.append(result)
                if urgent and any(c.selected for c in choices):
                    choices = [c for c in choices if c.selected]
                if not choices:
                    choices = [self._evaluate(running, [], ranks,
                        {r.adapter_dir: 0 for r in running}, 1)]
                choices.sort(key=self._order, reverse=True)
                choice = choices[0]
                if heavy:
                    # Divide the remaining search time between the best request sets.
                    shortlist = choices[:min(4, len(choices))]
                    for i, seed in enumerate(shortlist):
                        remaining = max(0., deadline - time.perf_counter())
                        sub_deadline = time.perf_counter() + remaining / (len(shortlist) - i)
                        result = self._search(running, seed.selected, ranks, window, sub_deadline, True)
                        if result is not None and self._order(result) > self._order(choice):
                            choice = result
                old = self._evaluate(running, choice.selected, ranks,
                    {r.adapter_dir: self.prefix_cache_state.get("retained_layers", {}).get(r.adapter_dir, 0)
                     for r in running + choice.selected}, choice.window)
                margin = max(self.config.hysteresis, min(0.5, self.cost.error))
                if old.plan.fits and old.slo_feasible and choice.score <= old.score * (1 + margin):
                    choice = old
        if heavy:
            self.last_heavy_step = self.decode_steps
        self.pending = choice if choice.plan.fits else None
        self.decisions += 1
        self.last_record = dict(event="space_plan", decode_step=self.decode_steps, heavy=heavy,
            calibrated=self.cost.calibrated, urgency_override=urgent, fits=choice.plan.fits,
            window=choice.window, selected=[r.request_id for r in choice.selected],
            running=len(running), retained_layers=choice.layers, required_slots=choice.plan.required_slots,
            capacity_slots=self.max_total_tokens, evictions=choice.plan.evictions,
            predicted_progress_tokens_s=choice.score, predicted_step_ms=choice.step_ms,
            step_slo_feasible=choice.slo_feasible,
            planning_ms=(time.perf_counter() - start) * 1000., model=self.cost.state_dict())
        return choice

    def generate_new_batch(self, current_batch, lora_ranks):
        if self.pending is None or not self.pending.selected:
            return None
        running = current_batch.reqs if current_batch else []
        selected = [r for r in self.pending.selected if not r.aborted]
        self.pending = None
        if not selected or not self.plan(running, selected, lora_ranks).fits:
            return None
        ids = {r.request_id for r in selected}
        self.waiting_req_list = [r for r in self.waiting_req_list if r.request_id not in ids and not r.aborted]
        batch = Batch(uuid.uuid4().hex, selected)
        batch.lowra_evict_prefix_lora_dirs = []
        batch.prefix_prefill_tokens = 0
        batch.prefill_token_num = sum(self.known_tokens(r) for r in selected)
        return batch
