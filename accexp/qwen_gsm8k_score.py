"""Numeric EM with declared-answer extraction, identical for all Qwen evaluation modes."""

from functools import lru_cache
import math
import re

if __package__:
    from .gsm8k_score import LOOSE_NUMBER, marked_number, normalize_number
else:
    from gsm8k_score import LOOSE_NUMBER, marked_number, normalize_number


POLICY_NAME = "gsm8k_qwen_first_declared_final_number_v2"
BOUNDARY = re.compile(r"<\|im_(?:start|end)\|>|(?m:^[ \t]*(?:### )?Question:)")
FINAL_MARKER = re.compile(r"####|\\boxed\s*\{|<answer>|\bthe (?:final )?answer is\s*:?[ \t]*", re.IGNORECASE)


def declared_number(suffix, marker):
    value = suffix.strip()
    # Only unwrap adjacent formatting, e.g. "The answer is:\n\\[\\boxed{5}\\]".
    # Never search past an invalid/wrong answer for a later correct declaration.
    for _ in range(16):
        if marker.lower() == "<answer>":
            closing = re.search(r"</answer>", value, re.IGNORECASE)
            if closing is None:
                return None
            value, marker = value[:closing.start()].strip(), ""
        if marker.lower().startswith("\\boxed"):
            return normalize_number(value.split("}", 1)[0]) if "}" in value else None
        wrapper = re.match(r"(?:\\\[|\\\(|\$\$?|\*\*)\s*", value)
        if wrapper:
            value = value[wrapper.end():].lstrip()
            continue
        nested = FINAL_MARKER.match(value)
        if nested:
            marker, value = nested[0], value[nested.end():].lstrip()
            continue
        if not value:
            return None
        line = value.splitlines()[0]
        line = re.sub(r"(?:\*\*|\\\]|\\\)|\$\$?)\s*[.!]?\s*$", "", line)
        return marked_number(line)
    return None


@lru_cache(maxsize=8192)
def _extract_prediction(text, trim_boundary=True):
    if not isinstance(text, str):
        raise ValueError("GSM8K prediction must be text")
    boundary = BOUNDARY.search(text) if trim_boundary else None
    answer = text[:boundary.start()].rstrip() if boundary else text
    marker = FINAL_MARKER.search(answer)
    strict, flexible, reason = None, None, "missing_final_marker"
    if marker:
        strict = declared_number(answer[marker.end():], marker[0])
        flexible = strict
        reason = "final_marker" if strict is not None else "invalid_final_marker"
    else:
        numbers = list(LOOSE_NUMBER.finditer(answer))
        if numbers:
            flexible = normalize_number(numbers[-1][1])
            reason = "last_number_fallback"
    return answer, strict, flexible, reason, boundary is not None, boundary.start() if boundary else None


def extract_prediction(text, trim_boundary=True):
    return dict(zip(("answer_text", "strict", "flexible", "reason", "boundary_trimmed", "boundary_offset"),
                    _extract_prediction(text, trim_boundary)))


def _score(predictions, references, samples, trim_boundary):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("GSM8K scoring requires nonempty, equally sized inputs")
    strict, flexible = [], []
    for text, answers in zip(predictions, references):
        if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
            raise ValueError("Each GSM8K reference must contain one numeric string")
        expected = normalize_number(answers[0])
        if expected is None:
            raise ValueError("Invalid numeric GSM8K reference")
        parsed = extract_prediction(text, trim_boundary)
        strict.append(float(parsed["strict"] is not None and parsed["strict"] == expected))
        flexible.append(float(parsed["flexible"] is not None and parsed["flexible"] == expected))
    return dict(exact_match=math.fsum(strict) / len(strict), flexible_exact_match=math.fsum(flexible) / len(flexible))


def score(predictions, references, samples):
    return _score(predictions, references, samples, True)


def score_unfiltered(predictions, references, samples):
    return _score(predictions, references, samples, False)
