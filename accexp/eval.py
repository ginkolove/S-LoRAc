"""Convenient entry point for the single-file Transformers/PEFT evaluator."""

if __package__:
    from .eval_minikv import parse_args, run
else:
    from eval_minikv import parse_args, run


if __name__ == "__main__":
    run(parse_args())
