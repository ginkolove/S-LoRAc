"""Deterministic, experiment-only policy. No oracle lengths or timing estimates."""

from collections import Counter

from slora.server.router.adaptive_req_queue import AdaptiveReqQueue, SpaceDecision


class ForkQueue(AdaptiveReqQueue):
    arm = "shrink"
    opportunity = None
    blocked_head = False

    def update_prefix_cache_state(self, state):
        self.prefix_cache_state = state or {}

    def choose(self, running, ranks, allow_admission=True, now=None):
        running = list(running)
        waiting = [r for r in self.waiting_req_list if not r.aborted] if allow_admission else []
        current = self.prefix_cache_state.get("retained_layers", {})
        ready = self.prefix_cache_state.get("ready_layers", {})
        counts = Counter(r.adapter_dir for r in running)
        targets = {d: current.get(d, 0) for d in counts}
        self.opportunity = None
        self.blocked_head = False

        def layout(selected, layers):
            return self._layout(running, selected, ranks, layers, 1)

        def finish(selected, layers):
            plan = layout(selected, layers)
            choice = SpaceDecision(list(selected), layers, plan, 1)
            self.pending = choice if plan.fits else None
            return choice

        # Use the exact allocator model, including the scratch introduced by L -> L-1.
        # Only sacrifice active prefixes; exhausted idle LRU is already in _layout.
        def shrink(selected, layers):
            donors = sorted((d for d in counts if layers[d]), key=lambda d:
                            (ranks[d], layers[d] == self.layer_num, counts[d], d))
            trial = dict(layers)
            if not layout(selected, dict(trial, **{d: 0 for d in donors})).fits:
                return None
            for d in donors:
                for k in range(trial[d] - 1, -1, -1):
                    trial[d] = k
                    if layout(selected, trial).fits:
                        return trial
            return None

        if not layout([], targets).fits:
            # Mandatory running progress is not an optional A/B decision.
            repaired = shrink([], targets)
            return finish([], targets if repaired is None else repaired)

        selected = []
        for r in waiting:
            candidate = dict(targets)
            candidate[r.adapter_dir] = current.get(r.adapter_dir, 0)
            if not layout(selected + [r], candidate).fits:
                break
            selected.append(r)
            targets = candidate

        if not selected and running and waiting:
            self.blocked_head = True
            # Fix the candidate width before observing either arm. A contiguous
            # FCFS prefix only; never admit a later request around its head.
            cap = (1, 2, 4, 8)[self.decode_steps % 4]
            for n in range(min(cap, len(waiting)), 0, -1):
                extra = waiting[:n]
                old = {r.adapter_dir: current.get(r.adapter_dir, 0) for r in running + extra}
                new = shrink(extra, old)
                if new is None:
                    continue
                extra_dirs = {r.adapter_dir for r in extra}
                lost = {d: old[d] - new[d] for d in counts if old[d] > new[d]}
                new_groups = sorted(d for d in extra_dirs if d not in counts and new[d] < self.layer_num)
                self.opportunity = dict(b0=len(running), n=n, rho=n / len(running),
                    head=extra[0].request_id, extra_ids=[r.request_id for r in extra],
                    new_groups=new_groups, lost_layers=lost, ranks={d: ranks[d] for d in old},
                    lost_ready={d: max(0, ready.get(d, 0) - new[d]) for d in lost},
                    newly_streamed_active=sorted(d for d in lost if old[d] == self.layer_num),
                    old_layers=old, new_layers=new, query_tokens=sum(self.known_tokens(r) for r in extra),
                    blocked_required_slots=layout(extra, old).required_slots,
                    chosen_required_slots=layout(extra, new).required_slots,
                    dynamic_capacity=self.max_total_tokens)
                return finish([], targets) if self.arm == "wait" else finish(extra, new)

        # Roll-in/continuation policy: use slack for retention only when the
        # visible queue clears; do not evict an idle entry merely to promote.
        if len(selected) == len(waiting) and (selected or allow_admission):
            base = layout(selected, targets)
            for d in sorted(targets, key=lambda x: (-counts[x], x)):
                if targets[d] < self.layer_num:
                    trial = dict(targets, **{d: self.layer_num})
                    p = layout(selected, trial)
                    if p.fits and not set(p.evictions) - set(base.evictions):
                        targets = trial
        return finish(selected, targets)
