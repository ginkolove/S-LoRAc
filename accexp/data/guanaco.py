"""timdettmers/openassistant-guanaco: train demonstrations, native eval queries."""

import re

if __package__:
    from .common import clean_examples, export_cli, fingerprint, hf_json, read_json
else:
    from common import clean_examples, export_cli, fingerprint, hf_json, read_json


REPO = "timdettmers/openassistant-guanaco"


def format_example(row, index):
    text = row.get("text")
    if not isinstance(text, str):
        return None
    markers = list(re.finditer(r"### (Human|Assistant):", text))
    if not markers or text[:markers[0].start()].strip() or len(markers) % 2:
        return None
    first_question = None
    for turn, marker in enumerate(markers):
        if marker.group(1) != ("Human" if turn % 2 == 0 else "Assistant"):
            return None
        end = markers[turn + 1].start() if turn + 1 < len(markers) else len(text)
        body = text[marker.end():end].strip()
        if not body:
            return None
        if turn == 0:
            first_question = body
    boundary = markers[-1].end()
    while boundary < len(text) and text[boundary].isspace():
        boundary += 1
    return dict(id=str(row.get("id", index)), query=text[:boundary],
                references=[text[boundary:].strip()], group_key=fingerprint(first_question))


def read_splits(args):
    train_rows = (read_json(args.train_file) if args.train_file else
                  hf_json(REPO, "openassistant_best_replies_train.jsonl", args.local_files_only))
    train, train_stats = clean_examples(train_rows, format_example)
    if args.test_file:
        test_rows, source = read_json(args.test_file), "provided_test_file"
    elif args.holdout_fraction is not None:
        test_rows, source = None, "explicit_train_holdout"
    else:
        test_rows = hf_json(REPO, "openassistant_best_replies_eval.jsonl", args.local_files_only)
        source = "native_eval_file"
    test, test_stats = clean_examples(test_rows, format_example) if test_rows is not None else (None, {})
    return train, test, dict(source=REPO, train_file=args.train_file, test_file=args.test_file,
                             test_source=source, train_stats=train_stats, test_stats=test_stats)


if __name__ == "__main__":
    export_cli(read_splits, "guanaco")
