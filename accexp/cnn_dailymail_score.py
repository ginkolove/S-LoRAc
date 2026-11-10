"""Macro-average ROUGE F1 on raw summaries, with no answer-dependent filtering."""

from functools import lru_cache
import math
import re


METRICS = ("rouge1", "rouge2", "rougeL")
TURN_POLICY = "cnn_summary_boundary_v1"
TURN_BOUNDARY = re.compile(
    r"(?m)^[ \t]*(?:### (?:Article|Instruction):|Article:|"
    r"You are given a news article below\. Please summarize the article, including only its highlights\.)"
)


def extract_summary(text):
    if not isinstance(text, str):
        raise ValueError("CNN/DailyMail prediction must be text")
    match = TURN_BOUNDARY.search(text)
    return dict(summary_text=(text[:match.start()] if match else text).strip(),
                boundary_trimmed=match is not None, boundary_offset=match.start() if match else None)


@lru_cache(maxsize=1)
def get_scorer():
    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError as error:
        raise RuntimeError("Install accexp/requirements.txt in the evaluation environment for ROUGE") from error
    return RougeScorer(METRICS, use_stemmer=True)


@lru_cache(maxsize=8192)
def example_scores(prediction, references):
    # Partial-result snapshots revisit predictions; cache only lexical scores.
    values = get_scorer().score_multi(list(references), prediction)
    return tuple(float(values[name].fmeasure) for name in METRICS)


def score(predictions, references, samples):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("ROUGE requires nonempty, equally sized predictions/references/samples")
    values = []
    for prediction, answers in zip(predictions, references):
        if not isinstance(prediction, str) or not answers or not all(isinstance(a, str) for a in answers):
            raise ValueError("ROUGE requires string predictions and nonempty string reference lists")
        values.append(example_scores(prediction, tuple(answers)))
    return {name: math.fsum(row[index] for row in values) / len(values)
            for index, name in enumerate(METRICS)}


def score_turn_filtered(predictions, references, samples):
    """Keep the first complete summary, cutting only explicit next-article markers."""
    return score([extract_summary(text)["summary_text"] for text in predictions], references, samples)
