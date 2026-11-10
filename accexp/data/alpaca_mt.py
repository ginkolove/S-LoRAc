"""HiTZ/alpaca_mt: fixed few-shot examples and remaining test data from raw JSON."""

import ast
from pathlib import Path

if __package__:
    from .common import ALPACA_PROMPTS, alpaca_example, clean_examples, export_cli, read_json
else:
    from common import ALPACA_PROMPTS, alpaca_example, clean_examples, export_cli, read_json


REPO = "HiTZ/alpaca_mt"
URL = "https://raw.githubusercontent.com/juletx/alpaca-lora-mt/main/data/alpaca_data_cleaned_{language}.json"


def language_templates(language, local_files_only, loader_path=None):
    if language == "en":
        return ALPACA_PROMPTS
    if loader_path is None:
        from huggingface_hub import hf_hub_download

        loader_path = hf_hub_download(REPO, "alpaca_mt.py", repo_type="dataset",
                                      local_files_only=local_files_only)
    # Read literal prompt strings from the original loader; do not execute it.
    tree = ast.parse(Path(loader_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PROMPTS"
                                               for target in node.targets):
            return ast.literal_eval(node.value)[language]
    raise ValueError("HiTZ source loader does not contain its PROMPTS dictionary")


def read_splits(args):
    templates = language_templates(args.language, args.local_files_only)
    if args.train_file:
        rows = read_json(args.train_file)
    else:
        from datasets import DownloadConfig, DownloadManager

        path = DownloadManager(download_config=DownloadConfig(local_files_only=args.local_files_only)).download(
            URL.format(language=args.language))
        rows = read_json(path)
    formatter = lambda row, index: alpaca_example(row, index, templates)
    train, train_stats = clean_examples(rows, formatter)
    test, test_stats = (clean_examples(read_json(args.test_file), formatter)
                        if args.test_file else (None, {}))
    return train, test, dict(source=REPO, raw_url=URL.format(language=args.language), language=args.language,
                             train_file=args.train_file, test_file=args.test_file,
                             test_source="provided_test_file" if args.test_file else "no_official_test",
                             train_stats=train_stats, test_stats=test_stats)


if __name__ == "__main__":
    export_cli(read_splits, "alpaca_mt")
