import os
import re
import sys
import subprocess
import argparse


def discover_response_files(data_new_dir):
    if not os.path.isdir(data_new_dir):
        return []
    return sorted(
        os.path.join(data_new_dir, f)
        for f in os.listdir(data_new_dir)
        if f.endswith(".jsonl") and "_input_response_data_" in f
    )


def parse_lang_and_model(filename):
    match = re.match(r"^([a-z]{2})_input_response_data_(.+)\.jsonl$", filename)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def resolve_input_data(data_dir, lang):
    final_clean = os.path.join(data_dir, f"{lang}_input_data_FINAL_CLEAN.jsonl")
    fallback = os.path.join(data_dir, f"{lang}_input_data.jsonl")
    if os.path.exists(final_clean):
        return final_clean
    if os.path.exists(fallback):
        return fallback
    return None


def main():
    parser = argparse.ArgumentParser(description="Run evaluations for all response files in data_new")
    parser.add_argument("--project_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data_new_dir", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--evaluations_dir", default=None)
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    data_new_dir = args.data_new_dir or os.path.join(project_dir, "data_new")
    data_dir = args.data_dir or os.path.join(project_dir, "data")
    evaluations_dir = args.evaluations_dir or os.path.join(project_dir, "eval_new")
    eval_script = os.path.join(project_dir, "evaluation_main.py")

    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"evaluation_main.py not found in: {project_dir}")

    os.makedirs(evaluations_dir, exist_ok=True)

    response_files = discover_response_files(data_new_dir)
    if not response_files:
        print(f"No response files found in: {data_new_dir}")
        return

    failures = []

    for resp_path in response_files:
        filename = os.path.basename(resp_path)
        lang, safe_model_name = parse_lang_and_model(filename)
        if not lang:
            print(f"Skipping unrecognized file name: {filename}")
            continue

        input_data = resolve_input_data(data_dir, lang)
        if not input_data:
            failures.append((filename, f"missing input data for lang '{lang}'"))
            print(f"Missing input data for {lang}: {filename}")
            continue

        output_dir = os.path.join(evaluations_dir, f"{lang}_input_response_data_{safe_model_name}")
        cmd = [
            sys.executable,
            eval_script,
            "--input_data",
            input_data,
            "--input_response_data",
            resp_path,
            "--output_dir",
            output_dir,
        ]

        print(f"Evaluating: {filename}")
        try:
            subprocess.run(cmd, check=True)
            print(f"OK: {filename}")
        except subprocess.CalledProcessError as exc:
            failures.append((filename, str(exc)))
            print(f"FAIL: {filename}")

    print("\nDone.")
    print(f"Total files: {len(response_files)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailed files:")
        for filename, reason in failures:
            print(f"- {filename}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
