"""DROP EM/F1 with grouped gold answers and reference-independent prediction parsing.

Metric semantics follow the official evaluator:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
"""

from functools import lru_cache
import json
import math
import re
import string

import numpy as np
from scipy.optimize import linear_sum_assignment


OFFICIAL_SCORER = (
    "https://github.com/allenai/allennlp-reading-comprehension/"
    "blob/master/allennlp_rc/eval/drop_eval.py"
)
PREDICTION_POLICY = "whole output; complete JSON string arrays are multi-span answers; otherwise one span"
TURN_POLICY = "drop_turn_boundary_v1"
TURN_BOUNDARY = re.compile(
    r"(?m)^[ \t]*(?:### (?:Passage|Question|Instruction):|Passage:|Question:|Q:|"
    r"Given a passage, you need to accurately identify and extract relevant spans of text that answer "
    r"specific questions\.)"
)


def extract_answer(text):
    if not isinstance(text, str):
        raise ValueError("DROP prediction must be text")
    match = TURN_BOUNDARY.search(text)
    return dict(answer_text=(text[:match.start()] if match else text).strip(),
                boundary_trimmed=match is not None, boundary_offset=match.start() if match else None)


def is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def normalize_answer(text):
    tokens = []
    for token in re.split(" |-", text):
        token = token.lower()
        if not is_number(token):
            token = "".join(char for char in token if char not in string.punctuation)
        if is_number(token):
            token = str(float(token))
        token = " ".join(re.sub(r"\b(a|an|the)\b", " ", token).split())
        if token:
            tokens.append(token)
    return " ".join(tokens)


def answer_spans(answer):
    """Convert one original annotation, not a flattened list of annotator answers."""
    if answer.get("number"):
        return (str(answer["number"]),), "number"
    if answer.get("spans"):
        spans = tuple(answer["spans"])
        if not all(isinstance(span, str) for span in spans):
            raise ValueError("DROP answer spans must be strings")
        return spans, "span" if len(spans) == 1 else "spans"
    date = answer.get("date", {})
    if any(date.get(key) for key in ("day", "month", "year")):
        return (" ".join(str(date.get(key, "")) for key in ("day", "month", "year")),), "date"
    raise ValueError("Empty or unrecognized DROP annotation")


def prediction_spans(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return (text,)
    if isinstance(value, list) and value and all(isinstance(span, str) and span.strip() for span in value):
        return tuple(value)
    return (text,)


def get_metrics(predicted, gold):
    """Official number-aware, optimally aligned EM/F1; return fractions in [0, 1]."""
    predicted = [predicted] if isinstance(predicted, str) else predicted
    gold = [gold] if isinstance(gold, str) else gold
    if not predicted or not gold:
        raise ValueError("DROP metrics require nonempty span lists")
    pred_normal = [normalize_answer(span) for span in predicted]
    gold_normal = [normalize_answer(span) for span in gold]
    exact = float(set(pred_normal) == set(gold_normal) and len(pred_normal) == len(gold_normal))
    pred_bags, gold_bags = [set(span.split()) for span in pred_normal], [set(span.split()) for span in gold_normal]
    alignment = np.zeros((len(gold_bags), len(pred_bags)))
    for i, expected in enumerate(gold_bags):
        expected_numbers = {token for token in expected if is_number(token)}
        for j, actual in enumerate(pred_bags):
            actual_numbers = {token for token in actual if is_number(token)}
            if expected_numbers and not expected_numbers.intersection(actual_numbers):
                continue
            overlap = len(expected.intersection(actual))
            precision = overlap / len(actual) if actual else 1.0
            recall = overlap / len(expected) if expected else 1.0
            alignment[i, j] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows, columns = linear_sum_assignment(-alignment)
    matched = np.zeros(max(len(gold_bags), len(pred_bags)))
    matched[rows] = alignment[rows, columns]
    f1 = float(round(np.mean(matched), 2))
    return exact, f1


@lru_cache(maxsize=8192)
def example_scores(prediction, candidates):
    predicted = prediction_spans(prediction)
    scores = [get_metrics(predicted, candidate) for candidate in candidates if candidate[0].strip()]
    if not scores:
        raise ValueError("No nonempty DROP gold answers")
    return max(value[0] for value in scores), max(value[1] for value in scores)


def score(predictions, references, samples):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("DROP scoring requires nonempty, equally sized inputs")
    values = []
    for prediction, sample in zip(predictions, samples):
        candidates = sample.get("metadata", {}).get("drop_gold_answers")
        if not isinstance(prediction, str) or not isinstance(candidates, list) or not candidates or any(
                not isinstance(candidate, list) or not candidate or not all(isinstance(span, str) for span in candidate)
                for candidate in candidates):
            raise ValueError("DROP scoring requires grouped metadata.drop_gold_answers; flattened references are unsafe")
        values.append(example_scores(prediction, tuple(tuple(candidate) for candidate in candidates)))
    return {name: math.fsum(value[index] for value in values) / len(values)
            for index, name in enumerate(("exact_match", "drop_f1"))}


def score_turn_filtered(predictions, references, samples):
    """Same metric, cutting only explicit next-turn markers, never using gold text."""
    return score([extract_answer(text)["answer_text"] for text in predictions], references, samples)
