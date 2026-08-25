"""Calculate strict and loose prompt-level accuracy from evaluation artifacts.

By default, this script reports every record found in an evaluation directory.
Pass ``--canonical-data`` to restrict the calculation to prompts in a canonical
dataset, which is useful for comparing legacy runs against Portuguese-IFEval's
535-prompt release.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATIONS_DIR = PROJECT_DIR / "experiments" / "evaluations"
DEFAULT_PORTUGUESE_DATA = PROJECT_DIR / "data" / "pt_input_data.jsonl"


@dataclass(frozen=True)
class Accuracy:
    passed: int
    total: int

    @property
    def percent(self) -> float:
        return 0.0 if self.total == 0 else (self.passed / self.total) * 100


def normalize_prompt(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    dash_map = {
        ord("\u2010"): "-",
        ord("\u2011"): "-",
        ord("\u2012"): "-",
        ord("\u2013"): "-",
        ord("\u2014"): "-",
    }
    return " ".join(value.translate(dash_map).strip().split())


def load_canonical_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = normalize_prompt(record.get("prompt"))
            if prompt is None:
                raise ValueError(f"{path}:{line_number}: missing prompt")
            prompts.add(prompt)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def calculate_accuracy(path: Path, canonical_prompts: set[str] | None = None) -> Accuracy | None:
    if not path.exists():
        return None

    passed = 0
    total = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error

            if canonical_prompts is not None:
                prompt = normalize_prompt(record.get("prompt"))
                if prompt not in canonical_prompts:
                    continue

            total += 1
            passed += bool(record.get("follow_all_instructions", False))

    return Accuracy(passed=passed, total=total)


def find_evaluation_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def format_score(score: Accuracy | None) -> str:
    return "N/A" if score is None else f"{score.percent:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate strict and loose prompt-level evaluation accuracy."
    )
    parser.add_argument(
        "--evaluations-dir",
        type=Path,
        default=DEFAULT_EVALUATIONS_DIR,
        help=f"Directory containing evaluation subdirectories (default: {DEFAULT_EVALUATIONS_DIR})",
    )
    parser.add_argument(
        "--canonical-data",
        type=Path,
        help=(
            "Restrict metrics to prompts in this JSONL dataset. Use "
            f"{DEFAULT_PORTUGUESE_DATA} for the canonical 535 Portuguese prompts."
        ),
    )
    args = parser.parse_args()

    canonical_prompts = (
        load_canonical_prompts(args.canonical_data) if args.canonical_data else None
    )
    if canonical_prompts is not None:
        print(f"Filtering to {len(canonical_prompts)} canonical prompts: {args.canonical_data}")

    print(f"{'MODEL / LANGUAGE':<65} | {'PROMPTS':>7} | {'STRICT %':>8} | {'LOOSE %':>8}")
    print("-" * 103)

    for directory in find_evaluation_directories(args.evaluations_dir):
        strict = calculate_accuracy(directory / "eval_results_strict.jsonl", canonical_prompts)
        loose = calculate_accuracy(directory / "eval_results_loose.jsonl", canonical_prompts)
        if strict is None and loose is None:
            continue

        prompt_count = strict.total if strict is not None else loose.total
        print(
            f"{directory.name[:65]:<65} | {prompt_count:>7} | "
            f"{format_score(strict):>8} | {format_score(loose):>8}"
        )


if __name__ == "__main__":
    main()
