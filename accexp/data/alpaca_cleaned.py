"""yahma/alpaca-cleaned: fixed few-shot examples and the remaining data for test."""

if __package__:
    from .common import alpaca_example, clean_examples, export_cli, hf_json, read_json
else:
    from common import alpaca_example, clean_examples, export_cli, hf_json, read_json


REPO = "yahma/alpaca-cleaned"


def read_splits(args):
    rows = (read_json(args.train_file) if args.train_file else
            hf_json(REPO, "alpaca_data_cleaned.json", args.local_files_only))
    train, train_stats = clean_examples(rows, alpaca_example)
    test, test_stats = (clean_examples(read_json(args.test_file), alpaca_example)
                        if args.test_file else (None, {}))
    return train, test, dict(source=REPO, train_file=args.train_file, test_file=args.test_file,
                             test_source="provided_test_file" if args.test_file else "no_official_test",
                             train_stats=train_stats, test_stats=test_stats)


if __name__ == "__main__":
    export_cli(read_splits, "alpaca_cleaned")
