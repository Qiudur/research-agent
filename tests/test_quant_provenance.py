"""Trusted quant input sealing rejects changed bytes and substituted paths."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from quant_research import provenance
from quant_research.provenance import REQUIRED_INPUTS, capture_inputs, verify_inputs


class QuantProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "runs/run-test/task_package/quant-demo/workspace"
        self.workspace.mkdir(parents=True)
        (self.workspace / "original.py").write_text("def compute_signal(history):\n    return {}\n")
        (self.workspace / "profile.json").write_text('{"schema":"profile/1"}\n')
        self.config = {"schema": "agent/1", "prompts": {}}
        (self.root / "config/prompts").mkdir(parents=True)
        for role in ("planner", "executor", "diagnoser", "summarizer"):
            relative = f"config/prompts/{role}.md"
            self.config["prompts"][role] = relative
            (self.root / relative).write_text(f"Instructions for {role}\n")
        self.agent_config = self.root / "config/quant_agent.json"
        self._write_config()
        self.prices = self.root / "config/quant_prices.json"
        self.prices.write_text('{"schema":"prices/1"}\n')
        self.hard_caps = self.root / "config/hard_caps.json"
        self.hard_caps.write_text('{"schema":"hard_caps/1"}\n')
        self.triggers = self.root / "config/triggers.json"
        self.triggers.write_text('{"schema":"triggers/1"}\n')

    def _write_config(self) -> None:
        self.agent_config.write_text(json.dumps(self.config) + "\n")

    def _capture(self, **overrides) -> dict:
        kwargs = {"agent_config": self.agent_config, "prices": self.prices,
                  "hard_caps": self.hard_caps, "triggers": self.triggers,
                  "task_workspace": self.workspace}
        kwargs.update(overrides)
        return capture_inputs(self.root, self.config, **kwargs)

    def _descriptor(self, path: Path) -> dict:
        return {"path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_capture_seals_all_four_prompts_and_task_inputs(self) -> None:
        inputs = self._capture()
        self.assertEqual(set(inputs), REQUIRED_INPUTS)
        for label, descriptor in inputs.items():
            self.assertEqual(set(descriptor), {"path", "sha256"})
            self.assertFalse(Path(descriptor["path"]).is_absolute())
            self.assertEqual(len(descriptor["sha256"]), 64)
        verify_inputs(self.root, inputs)

    def test_each_input_change_is_detected(self) -> None:
        inputs = self._capture()
        for label, descriptor in inputs.items():
            with self.subTest(label=label):
                path = self.root / descriptor["path"]
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    verify_inputs(self.root, inputs)
                path.write_bytes(original)

    def test_missing_unknown_and_invalid_descriptors_are_rejected(self) -> None:
        inputs = self._capture()
        missing = copy.deepcopy(inputs)
        missing.pop("prompt_summarizer")
        extra = {**inputs, "unexpected": inputs["prices"]}
        cases = [None, [], missing, extra]
        for value in (None, {}, [], {"path": "config/quant_prices.json"},
                      {**inputs["prices"], "other": "unexpected"}):
            modified = copy.deepcopy(inputs)
            modified["prices"] = value
            cases.append(modified)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_inputs(self.root, value)

    def test_invalid_relative_paths_and_digests_are_rejected(self) -> None:
        inputs = self._capture()
        for value in ("", " ", None, 4, ".", "config/../config/quant_prices.json",
                      str(self.prices), "config/missing.json", "config"):
            modified = copy.deepcopy(inputs)
            modified["prices"]["path"] = value
            with self.subTest(path=value), self.assertRaises(ValueError):
                verify_inputs(self.root, modified)
        for digest in (None, "", 0, "a" * 63, "x" * 64):
            modified = copy.deepcopy(inputs)
            modified["prices"]["sha256"] = digest
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                verify_inputs(self.root, modified)

    def test_prompt_substitution_is_rejected_even_when_bytes_match(self) -> None:
        inputs = self._capture()
        original = self.root / inputs["prompt_planner"]["path"]
        replacement = self.root / "config/prompts/other.md"
        replacement.write_bytes(original.read_bytes())
        inputs["prompt_planner"] = self._descriptor(replacement)
        with self.assertRaisesRegex(ValueError, "sealed agent configuration"):
            verify_inputs(self.root, inputs)

    def test_prompt_escape_and_symlink_escape_are_rejected(self) -> None:
        with TemporaryDirectory() as outside:
            secret = Path(outside) / "prompt.md"
            secret.write_text("External source\n")
            link = self.root / "config/prompts/external.md"
            link.symlink_to(secret)
            for value in (str(secret), "../prompt.md", "config/prompts/external.md", None, ""):
                with self.subTest(path=value):
                    self.config["prompts"]["planner"] = value
                    self._write_config()
                    with self.assertRaises(ValueError):
                        self._capture()

    def test_symlink_retarget_after_capture_is_rejected(self) -> None:
        inputs = self._capture()
        price_path = self.root / inputs["prices"]["path"]
        with TemporaryDirectory() as outside:
            other = Path(outside) / "same-prices.json"
            other.write_bytes(price_path.read_bytes())
            price_path.unlink()
            price_path.symlink_to(other)
            with self.assertRaises(ValueError):
                verify_inputs(self.root, inputs)

    def test_task_inputs_must_stay_together_in_expected_workspace(self) -> None:
        inputs = self._capture()
        other = self.root / "runs/other/task_package/quant-demo/workspace"
        other.mkdir(parents=True)
        (other / "profile.json").write_bytes((self.workspace / "profile.json").read_bytes())
        inputs["task_profile"] = self._descriptor(other / "profile.json")
        with self.assertRaisesRegex(ValueError, "task workspace"):
            verify_inputs(self.root, inputs)
        invalid = self.root / "visible/workspace"
        invalid.mkdir(parents=True)
        for name in ("original.py", "profile.json"):
            (invalid / name).write_bytes((self.workspace / name).read_bytes())
        with self.assertRaisesRegex(ValueError, "task workspace"):
            self._capture(task_workspace=invalid)

    def test_fixed_config_paths_cannot_be_substituted(self) -> None:
        for label in ("hard_caps", "triggers"):
            with self.subTest(label=label):
                inputs = self._capture()
                source = self.root / inputs[label]["path"]
                other = self.root / f"config/other_{label}.json"
                other.write_bytes(source.read_bytes())
                inputs[label] = self._descriptor(other)
                with self.assertRaisesRegex(ValueError, "substituted"):
                    verify_inputs(self.root, inputs)
                with self.assertRaisesRegex(ValueError, "substituted"):
                    self._capture(**{label: other})

    def test_capture_rejects_config_argument_different_from_source(self) -> None:
        self.config["new_option"] = True
        with self.assertRaisesRegex(ValueError, "differs from the source file"):
            self._capture()

    def test_config_change_during_capture_cannot_replace_original_seal(self) -> None:
        reader = provenance._read
        mutated = False

        def read_then_mutate(path: Path, label: str) -> bytes:
            nonlocal mutated
            raw = reader(path, label)
            if label == "agent_config" and not mutated:
                mutated = True
                replacement = {**self.config, "new_option": True}
                self.agent_config.write_text(json.dumps(replacement) + "\n")
            return raw

        with patch("quant_research.provenance._read", side_effect=read_then_mutate):
            with self.assertRaisesRegex(ValueError, "agent_config.*SHA-256"):
                self._capture()


if __name__ == "__main__":
    unittest.main()
