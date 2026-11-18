"""Opt-in planner settings shared by the server and benchmark launchers."""

from dataclasses import asdict, dataclass
import math


@dataclass
class SpacePlanConfig:
    interval: int = 10
    window: int = 10
    group: int = 4
    beam: int = 8
    candidates: int = 64
    budget_ms: float = 10.0
    max_wait_ms: float = 3000.0
    max_step_ms: float = 0.0
    hysteresis: float = 0.05

    def __post_init__(self):
        for name in ("interval", "window", "group", "beam", "candidates"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError("space-plan-%s must be a positive integer" % name)
        for name in ("budget_ms", "max_wait_ms", "max_step_ms", "hysteresis"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("space-plan-%s must be finite and nonnegative" % name)
        if not self.budget_ms or self.hysteresis >= 1:
            raise ValueError("planner budget must be positive and hysteresis below one")


def add_space_plan_arguments(parser):
    defaults = SpacePlanConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument("--space-plan-" + name.replace("_", "-"),
                            type=int if isinstance(value, int) else float, default=value)


def space_plan_options(args):
    defaults = SpacePlanConfig()
    return asdict(SpacePlanConfig(**{
        name: getattr(args, "space_plan_" + name, value)
        for name, value in asdict(defaults).items()}))


def space_plan_command(args):
    result = []
    for name, value in space_plan_options(args).items():
        result.extend(["--space-plan-" + name.replace("_", "-"), str(value)])
    return result
