"""Reference-independent, current-answer-only numeric exact match for GSM8K."""

from decimal import Decimal, InvalidOperation
from functools import lru_cache
import math
import re


PROMPT_PREFIX = "Please answer the following question: "
POLICY_NAME = "gsm8k_first_task_final_number_v1"
REFERENCE = "https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py"
BOUNDARY = re.compile(r"(?m)^[ \t]*(?:" + re.escape(PROMPT_PREFIX.rstrip()) + r"|(?:### )?Question:)")
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
FINAL_NUMBER = re.compile(r"\$?(" + NUMBER + r")\.?\Z")
LOOSE_NUMBER = re.compile(r"(?<![\w./])\$?(" + NUMBER + r")(?![\w/]|\.\d)")
MARKED_NUMBER = re.compile(r"\$?(" + NUMBER + r")(.*)\Z")


def normalize_number(text):
    """Compare finite decimals exactly; never evaluate arithmetic or use floats."""
    match = FINAL_NUMBER.fullmatch(text.strip())
    if match is None:
        return None
    try:
        value = Decimal(match[1].replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if value == 0 else normalized.lstrip("+")


def gold_number(solution):
    if solution.count("####") != 1:
        raise ValueError("A GSM8K gold solution must have exactly one #### marker")
    value = normalize_number(solution.split("####", 1)[1])
    if value is None:
        raise ValueError("Invalid numeric GSM8K gold answer")
    return value


def marked_number(line):
    match = MARKED_NUMBER.fullmatch(line.strip())
    if match is None:
        return None
    tail = match[2].lstrip()
    # The official convention extracts a numeric prefix, so trailing units are
    # harmless. Do not silently accept the first operand of an expression.
    if tail and tail != "." and (tail[0] in "/+*-^=.,\u00d7\u00f7" or re.match(r"[eE][+-]?\d", tail)):
        return None
    return normalize_number(match[1])


@lru_cache(maxsize=8192)
def _extract_prediction(text, trim_boundary=True):
    """Freeze extraction before scoring; no reference or model/mode is accepted."""
    boundary = BOUNDARY.search(text) if trim_boundary else None
    answer = text[:boundary.start()].rstrip() if boundary else text
    strict, flexible, reason = None, None, "missing_final_marker"
    if "####" in answer:
        # The first declared answer is binding, even when malformed or wrong.
        suffix = answer.split("####", 1)[1].lstrip(" \t")
        line = suffix.splitlines()[0] if suffix else ""
        strict = marked_number(line)
        flexible = strict
        reason = "final_marker" if strict is not None else "invalid_final_marker"
    else:
        matches = list(LOOSE_NUMBER.finditer(answer))
        if matches:
            flexible = normalize_number(matches[-1][1])
            reason = "last_number_fallback"
    return answer, strict, flexible, reason, boundary is not None, boundary.start() if boundary else None


def extract_prediction(text, trim_boundary=True):
    keys = ("answer_text", "strict", "flexible", "reason", "boundary_trimmed", "boundary_offset")
    return dict(zip(keys, _extract_prediction(text, trim_boundary)))


def _score(predictions, references, samples, trim_boundary):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("GSM8K scoring requires nonempty, equally sized inputs")
    strict, flexible = [], []
    for prediction, answers in zip(predictions, references):
        if not isinstance(prediction, str) or not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
            raise ValueError("Each GSM8K question requires one numeric reference and a string prediction")
        expected = normalize_number(answers[0])
        if expected is None:
            raise ValueError("GSM8K references must contain final numbers, not entire solutions")
        value = extract_prediction(prediction, trim_boundary)
        strict.append(float(value["strict"] is not None and value["strict"] == expected))
        flexible.append(float(value["flexible"] is not None and value["flexible"] == expected))
    return dict(exact_match=math.fsum(strict) / len(strict),
                flexible_exact_match=math.fsum(flexible) / len(flexible))


def score(predictions, references, samples):
    return _score(predictions, references, samples, trim_boundary=True)


def score_unfiltered(predictions, references, samples):
    return _score(predictions, references, samples, trim_boundary=False)
