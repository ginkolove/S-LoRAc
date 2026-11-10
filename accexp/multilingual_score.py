"""Language-neutral reference overlap, with separate scores for each language.

These are generic lexical metrics, not an official Alpaca scorer or a semantic
quality judge. Accents and articles are retained; punctuation separates tokens.
"""

from collections import Counter, defaultdict
import unicodedata


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char
                   for char in text)
    return " ".join(text.split())


def example_score(prediction, answers):
    prediction = normalize_answer(prediction)
    left = prediction.split()
    exact, f1 = [], []
    for answer in answers:
        answer = normalize_answer(answer)
        right = answer.split()
        exact.append(float(prediction == answer))
        if not left or not right:
            f1.append(float(left == right))
        else:
            overlap = sum((Counter(left) & Counter(right)).values())
            f1.append(2.0 * overlap / (len(left) + len(right)))
    return dict(exact_match=max(exact), generic_token_f1=max(f1))


def score(predictions, references, samples):
    if not predictions or not len(predictions) == len(references) == len(samples):
        raise ValueError("Scoring requires nonempty, equally sized inputs")
    grouped = defaultdict(list)
    for prediction, answers, sample in zip(predictions, references, samples):
        language = sample.get("metadata", {}).get("language")
        if not language or not answers:
            raise ValueError("Multilingual scoring requires a language and reference answers")
        grouped[language].append(example_score(prediction, answers))
    metrics = ("exact_match", "generic_token_f1")
    by_language = {
        language: dict(samples=len(values), **{
            metric: sum(value[metric] for value in values) / len(values) for metric in metrics})
        for language, values in sorted(grouped.items())}
    result = {metric: sum(value[metric] * value["samples"] for value in by_language.values()) /
              len(predictions) for metric in metrics}
    result.update({f"macro_{metric}": sum(value[metric] for value in by_language.values()) /
                   len(by_language) for metric in metrics})
    result["by_language"] = by_language
    return result
