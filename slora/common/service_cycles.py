"""Record complete busy periods without changing request admission or execution."""

import json
import time


class ServiceCycleRecorder:
    def __init__(self, output, clock=time.monotonic):
        self.output = output
        self.clock = clock
        self.current = None
        self.cycle_id = 0

    def begin(self, start=None):
        if self.current is not None:
            raise RuntimeError("service cycle already active")
        self.cycle_id += 1
        self.current = dict(cycle_id=self.cycle_id, start=self.clock() if start is None else start,
                            admitted=set(), completed=set(), aborted=set(), generated_tokens=0,
                            waiting_min=None, admissions=0)

    def admit(self, request_ids, waiting):
        if self.current is None:
            return
        self.current["admitted"].update(request_ids)
        self.current["admissions"] += 1
        self.observe_waiting(waiting)

    def tokens(self, count):
        if self.current is not None:
            self.current["generated_tokens"] += count

    def complete(self, request_id, aborted=False):
        if self.current is not None:
            self.current["aborted" if aborted else "completed"].add(request_id)

    def observe_waiting(self, waiting):
        if self.current is not None:
            previous = self.current["waiting_min"]
            self.current["waiting_min"] = waiting if previous is None else min(previous, waiting)

    def finish(self, waiting):
        if self.current is None:
            return
        self.observe_waiting(waiting)
        cycle, self.current = self.current, None
        end = self.clock()
        admitted, completed, aborted = (cycle[key] for key in ("admitted", "completed", "aborted"))
        record = dict(event="service_cycle", cycle_id=cycle["cycle_id"], start=cycle["start"],
                      end=end, duration=end - cycle["start"], admissions=cycle["admissions"],
                      admitted_requests=len(admitted), completed_requests=len(completed),
                      completed_request_ids=sorted(completed), aborted_requests=len(aborted),
                      generated_tokens=cycle["generated_tokens"], waiting_min=cycle["waiting_min"],
                      continuously_backlogged=cycle["waiting_min"] is not None and cycle["waiting_min"] > 0,
                      valid=bool(admitted) and admitted == completed and not aborted)
        self.output.write(json.dumps(record, allow_nan=False) + "\n")
        self.output.flush()

    def reset(self):
        if self.current is not None:
            raise RuntimeError("cannot reset an active service cycle")
