"""Reference-independent final-number scoring for the LEMA GSM8K prompt protocol."""

from functools import lru_cache
import math
import re

if __package__:
    from .data.llama2_gsm8k import PROMPT_PREFIX
    from .gsm8k_score import LOOSE_NUMBER, marked_number, normalize_number
else:
    from data.llama2_gsm8k import PROMPT_PREFIX
    from gsm8k_score import LOOSE_NUMBER, marked_number, normalize_number


POLICY_NAME = "gsm8k_lema_first_task_final_number_v1"
BOUNDARY = re.compile(r"(?m)^[ \t]*(?:" + re.escape(PROMPT_PREFIX) + r"|(?:### )?Question:)")
FINAL_MARKER = re.compile(r"####|\bthe (?:final )?answer is\s*:?[ \t]*", re.IGNORECASE)


@lru_cache(maxsize=8192)
def _extract_prediction(text, trim_boundary=True):
    if not isinstance(text, str):
        raise ValueError("GSM8K prediction must be text")
    boundary = BOUNDARY.search(text) if trim_boundary else None
    answer = text[:boundary.start()].rstrip() if boundary else text
    marker = FINAL_MARKER.search(answer)
    strict, flexible, reason = None, None, "missing_final_marker"
    if marker:
        suffix = answer[marker.end():].lstrip(" \t")
        strict = marked_number(suffix.splitlines()[0] if suffix else "")
        flexible = strict
        reason = "final_marker" if strict is not None else "invalid_final_marker"
    else:
        numbers = list(LOOSE_NUMBER.finditer(answer))
        if numbers:
            flexible = normalize_number(numbers[-1][1])
            reason = "last_number_fallback"
    return answer, strict, flexible, reason, boundary is not None, boundary.start() if boundary else None


def extract_prediction(text, trim_boundary=True):
    keys = ("answer_text", "strict", "flexible", "reason", "boundary_trimmed", "boundary_offset")
    return dict(zip(keys, _extract_prediction(text, trim_boundary)))


def _score(predictions, references, samples, trim_boundary):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("GSM8K scoring requires nonempty, equally sized inputs")
    strict, flexible = [], []
    for text, answers in zip(predictions, references):
        if not isinstance(text, str) or not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
            raise ValueError("Each GSM8K question requires one numeric reference and a string prediction")
        expected = normalize_number(answers[0])
        if expected is None:
            raise ValueError("GSM8K references must contain final numbers, not entire solutions")
        value = extract_prediction(text, trim_boundary)
        strict.append(float(value["strict"] is not None and value["strict"] == expected))
        flexible.append(float(value["flexible"] is not None and value["flexible"] == expected))
    return dict(exact_match=math.fsum(strict) / len(strict),
                flexible_exact_match=math.fsum(flexible) / len(flexible))


def score(predictions, references, samples):
    return _score(predictions, references, samples, trim_boundary=True)


def score_unfiltered(predictions, references, samples):
    return _score(predictions, references, samples, trim_boundary=False)
