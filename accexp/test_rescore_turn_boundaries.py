"""Answer boundaries, false-positive guards, and offline score provenance."""

import json
from pathlib import Path
import tempfile
import unittest

from multilingual_score import score
from rescore_turn_boundaries import MODES, extract_answer, load_boundaries, parse_args, run


BOUNDARIES = dict(headers=["### Instruction:", "### Instrucci\u00f3n:"],
                  preambles=["Below is an instruction. Write an answer."])


class BoundaryTests(unittest.TestCase):
    def test_localized_next_turn_and_first_boundary(self):
        text = "11\n\n### Instrucci\u00f3n:\nOther task\n\n### Instruction:\nMore"
        result = extract_answer(text, BOUNDARIES)
        self.assertEqual(result["text"], "11")
        self.assertEqual(text[result["cut_offset"]:], text[text.index("###"):])
        self.assertEqual(result["reason"], "instruction_header")

    def test_complete_preamble_removed_even_when_wrapped(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nBelow is an instruction.\nWrite an answer.\n\n### Instruction:\nNext"
        result = extract_answer(text, BOUNDARIES)
        self.assertEqual(result["text"], "First paragraph.\n\nSecond paragraph.")
        self.assertEqual(result["reason"], "template_restart")

    def test_ordinary_paragraphs_headings_and_partial_marker_are_kept(self):
        for text in ("Paragraph one.\n\nParagraph two.", "### Explanation:\nAnswer",
                     "An inline ### Instruction: example.", "Answer\n\n### Instruct"):
            self.assertEqual(extract_answer(text, BOUNDARIES)["text"], text)

    def test_fenced_and_indented_code_and_blockquotes_are_protected(self):
        for text in ("```text\n### Instruction:\n```", "~~~\n### Instruction:\n~~~",
                     "````\n```\n### Instruction:\n````", "    ### Instruction:\n    quoted code",
                     "\t### Instruction:\n", "> ### Instruction:\nquoted text"):
            self.assertFalse(extract_answer(text, BOUNDARIES)["truncated"])
        text = "```\n### Instruction:\n```\nAnswer\n\n### Instruction:\nNext"
        self.assertEqual(extract_answer(text, BOUNDARIES)["text"], "```\n### Instruction:\n```\nAnswer")

    def test_initial_header_produces_empty_answer_instead_of_salvaging_next_answer(self):
        result = extract_answer("### Instruction:\nNext\n### Response:\n11", BOUNDARIES)
        self.assertTrue(result["empty_answer"])
        self.assertEqual(result["text"], "")

    def test_actual_wrong_answer_is_preserved(self):
        self.assertEqual(extract_answer("True\n\n### Instruction:\nNext", BOUNDARIES)["text"], "True")

    def test_loader_reads_literal_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loader.py"
            path.write_text("raise RuntimeError('must not execute')\nPROMPTS = " + repr({"en": {
                "prompt_no_input": "Below is an instruction. Write an answer.\n\n### Instruction:\n{instruction}\n\n### Response:\n"}}))
            self.assertEqual(load_boundaries(path), dict(headers=["### Instruction:"], preambles=BOUNDARIES["preambles"]))

    def test_end_to_end_uniform_scoring_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = dict(id="one", query="Question", prefix="Prefix", references=["11"], metadata=dict(language="es"))
            data = root / "data.jsonl"
            data.write_text(json.dumps(sample) + "\n")
            template = root / "loader.py"
            template.write_text("PROMPTS = " + repr({"en": {
                "prompt_no_input": "Below is an instruction. Write an answer.\n\n### Instruction:\n{instruction}\n\n### Response:\n"}}))
            text = "11\n\n### Instruction:\nNext task"
            raw_scores = score([text], [["11"]], [sample])
            result = root / "result.json"
            result.write_text(json.dumps(dict(config=dict(model="tiny"), scores={mode: raw_scores for mode in MODES},
                predictions=[dict(id="one", references=["11"], metadata=sample["metadata"], predictions={
                    mode: dict(text=text, output_tokens=12, stopped_on_eos=False) for mode in MODES})])))
            before = {path: path.read_bytes() for path in (data, template, result)}
            args = parse_args(["--results", str(result), "--data", str(data), "--template-loader", str(template),
                               "--output-dir", str(root / "out")])
            report = run(args)
            for mode in MODES:
                self.assertEqual(report["models"]["result"]["filtered_scores"][mode]["exact_match"], 1)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertTrue(report["inputs_unchanged"])
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
