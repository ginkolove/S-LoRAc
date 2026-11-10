"""Macro-average ROUGE F1 for Multi-News, retaining full summaries before next-turn markers."""

import re

if __package__:
    from .cnn_dailymail_score import METRICS, score
    from .data.multi_news import ALPACA_PREAMBLE, INSTRUCTION
else:
    from cnn_dailymail_score import METRICS, score
    from data.multi_news import ALPACA_PREAMBLE, INSTRUCTION


TURN_POLICY = "multi_news_summary_boundary_v2"
TURN_BOUNDARY = re.compile(r"(?m)^[ \t]*(?:### (?:Documents|Instruction):|Documents:|"
                           + re.escape(INSTRUCTION) + "|" + re.escape(ALPACA_PREAMBLE) + r")")


def extract_summary(text):
    if not isinstance(text, str):
        raise ValueError("Multi-News prediction must be text")
    match = TURN_BOUNDARY.search(text)
    return dict(summary_text=(text[:match.start()] if match else text).strip(),
                boundary_trimmed=match is not None, boundary_offset=match.start() if match else None)


def score_turn_filtered(predictions, references, samples):
    return score([extract_summary(text)["summary_text"] for text in predictions], references, samples)
