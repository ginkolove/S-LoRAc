"""Generic English overlap scores, with an identical first-answer policy for every mode."""

from functools import lru_cache
import math

if __package__:
    from .data.common import ALPACA_PROMPTS
    from .eval_minikv import default_score
    from .rescore_turn_boundaries import extract_answer
else:
    from data.common import ALPACA_PROMPTS
    from eval_minikv import default_score
    from rescore_turn_boundaries import extract_answer


POLICY = "qwen_alpaca_first_answer_v1"
METRICS = ("exact_match", "generic_token_f1", "rougeL")
BOUNDARIES = dict(headers=["### Instruction:"], preambles=sorted({
    " ".join(template.split("### Instruction:")[0].split()) for template in ALPACA_PROMPTS.values()}))


def extract_prediction(text):
    if not isinstance(text, str):
        raise ValueError("Alpaca prediction must be text")
    return extract_answer(text, BOUNDARIES)


@lru_cache(maxsize=1)
def rouge_scorer():
    from rouge_score.rouge_scorer import RougeScorer
    return RougeScorer(["rougeL"], use_stemmer=True)


@lru_cache(maxsize=8192)
def example_scores(prediction, references):
    values = default_score([prediction], [references], [{}])
    values["rougeL"] = float(rouge_scorer().score_multi(list(references), prediction)["rougeL"].fmeasure)
    return tuple(values[name] for name in METRICS)


def score_unfiltered(predictions, references, samples):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("Require nonempty, equally sized predictions, references and samples")
    values = []
    for text, answers in zip(predictions, references):
        if not isinstance(text, str) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
            raise ValueError("Require string predictions and nonempty string references")
        values.append(example_scores(text, tuple(answers)))
    return {name: math.fsum(row[index] for row in values) / len(values)
            for index, name in enumerate(METRICS)}


def score(predictions, references, samples):
    return score_unfiltered([extract_prediction(text)["text"] for text in predictions], references, samples)
