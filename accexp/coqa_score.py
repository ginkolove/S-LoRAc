"""CoQA leave-one-annotator-out EM/F1, with uniform next-turn filtering."""

from collections import Counter
from functools import lru_cache
import math
import re
import string


OFFICIAL_SCORER = "https://downloads.cs.stanford.edu/nlp/data/coqa/evaluate-v1.0.py"
POLICY_NAME = "coqa_qa_turn_boundary_v1"
INSTRUCTION = "Read the passage and answer each question briefly. If the answer is unknown, say unknown."
BOUNDARY = re.compile(r"(?m)^[ \t]*(?:Q:|Question:|Passage:|### (?:Question|Passage):|"
                      + re.escape(INSTRUCTION) + r")")


def extract_answer(text):
    if not isinstance(text, str):
        raise ValueError("CoQA prediction must be text")
    match = BOUNDARY.search(text)
    return dict(answer_text=(text[:match.start()] if match else text).strip(),
                boundary_trimmed=match is not None)


def normalize_answer(text):
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def pair_scores(gold, prediction):
    expected, actual = normalize_answer(gold), normalize_answer(prediction)
    exact = float(expected == actual)
    gold_tokens, pred_tokens = expected.split(), actual.split()
    if not gold_tokens or not pred_tokens:
        return exact, float(gold_tokens == pred_tokens)
    overlap = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return exact, 2 * precision * recall / (precision + recall) if overlap else 0.0


@lru_cache(maxsize=8192)
def turn_scores(prediction, references):
    # Keep duplicate annotator answers: removing duplicates changes official scores.
    values = [pair_scores(gold, prediction) for gold in references]
    if len(values) == 1:
        return values[0]
    return tuple(math.fsum(max(value[metric] for j, value in enumerate(values) if j != excluded)
                           for excluded in range(len(values))) / len(values) for metric in (0, 1))


def _score(predictions, references, samples, filtered):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("CoQA scoring requires nonempty, equally sized inputs")
    values = []
    for prediction, answers, sample in zip(predictions, references, samples):
        gold = sample.get("metadata", {}).get("coqa_gold_answers")
        if not isinstance(gold, list) or not gold or not all(isinstance(a, str) for a in gold):
            raise ValueError("CoQA scoring requires metadata.coqa_gold_answers")
        if answers != gold:
            raise ValueError("CoQA references differ from official annotations")
        if not isinstance(prediction, str):
            raise ValueError("CoQA prediction must be text")
        values.append(turn_scores(extract_answer(prediction)["answer_text"] if filtered else prediction,
                                  tuple(gold)))
    return {name: math.fsum(value[i] for value in values) / len(values)
            for i, name in enumerate(("exact_match", "coqa_f1"))}


def score(predictions, references, samples):
    return _score(predictions, references, samples, True)


def score_unfiltered(predictions, references, samples):
    return _score(predictions, references, samples, False)
