"""Evaluate Portuguese (PT) and translated-Portuguese (PTEN) response files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSES_DIR = PROJECT_DIR / "experiments" / "generated_responses"
DEFAULT_EVALUATIONS_DIR = PROJECT_DIR / "experiments" / "generated_evaluations"
DEFAULT_PT_INPUT = PROJECT_DIR / "data" / "pt_input_data.jsonl"
DEFAULT_PTEN_INPUT = PROJECT_DIR / "experiments" / "data_close" / "pten_input_data.jsonl"

LANGUAGE_ALIASES = {"pt": "pt", "pten": "pten", "pt_en": "pten"}


def discover_response_files(responses_dir: Path) -> list[Path]:
    if not responses_dir.is_dir():
        return []
    return sorted(
        path
        for path in responses_dir.iterdir()
        if path.is_file() and path.suffix == ".jsonl" and "_input_response_data_" in path.name
    )


def parse_language_and_model(path: Path) -> tuple[str | None, str | None]:
    prefix, marker, model_name = path.stem.partition("_input_response_data_")
    if not marker or not model_name:
        return None, None
    return LANGUAGE_ALIASES.get(prefix), model_name


def load_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"{path}:{line_number}: missing prompt")
            prompts.add(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def validate_response_coverage(input_data: Path, response_data: Path) -> int:
    required_prompts = load_prompts(input_data)
    response_prompts = load_prompts(response_data)
    missing_prompts = required_prompts - response_prompts
    if missing_prompts:
        raise ValueError(
            f"{response_data} is missing {len(missing_prompts)} prompts required by {input_data}"
        )
    return len(required_prompts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PT and PTEN response files stored in one directory."
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--responses-dir",
        "--data-new-dir",
        dest="responses_dir",
        type=Path,
        default=DEFAULT_RESPONSES_DIR,
    )
    parser.add_argument(
        "--evaluations-dir", type=Path, default=DEFAULT_EVALUATIONS_DIR
    )
    parser.add_argument("--pt-input-data", type=Path, default=DEFAULT_PT_INPUT)
    parser.add_argument("--pten-input-data", type=Path, default=DEFAULT_PTEN_INPUT)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("pt", "pten"),
        default=("pt", "pten"),
        help="Languages to evaluate (default: pt pten).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input/response coverage and print planned evaluations without running them.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    eval_script = project_dir / "evaluation_main.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"evaluation_main.py not found in: {project_dir}")

    input_by_language = {"pt": args.pt_input_data, "pten": args.pten_input_data}
    for language in args.languages:
        if not input_by_language[language].is_file():
            raise FileNotFoundError(
                f"Input data for {language.upper()} not found: {input_by_language[language]}"
            )

    response_files = discover_response_files(args.responses_dir)
    selected_files = []
    for response_path in response_files:
        language, model_name = parse_language_and_model(response_path)
        if language is None:
            print(f"Skipping unrecognized file name: {response_path.name}")
        elif language in args.languages:
            selected_files.append((response_path, language, model_name))

    if not selected_files:
        requested = ", ".join(args.languages)
        print(f"No {requested.upper()} response files found in: {args.responses_dir}")
        return

    args.evaluations_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, str]] = []

    for response_path, language, model_name in selected_files:
        input_data = input_by_language[language]
        output_dir = args.evaluations_dir / response_path.stem
        try:
            prompt_count = validate_response_coverage(input_data, response_path)
            command = [
                sys.executable,
                str(eval_script),
                "--input_data",
                str(input_data),
                "--input_response_data",
                str(response_path),
                "--output_dir",
                str(output_dir),
            ]
            print(
                f"{language.upper()} | {model_name} | {prompt_count} prompts | "
                f"{response_path.name}"
            )
            if args.dry_run:
                continue
            subprocess.run(command, check=True)
            print(f"OK: {response_path.name}")
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            message = str(error)
            if response_path.name.startswith("pt_en_input_response_data_"):
                message += " Use --pten-input-data with the matching legacy input dataset."
            failures.append((response_path, message))
            print(f"FAIL: {response_path.name}: {message}")

    print(f"\nEvaluated files: {len(selected_files) - len(failures)}/{len(selected_files)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
