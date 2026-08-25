"""Executa inferência e avaliação do benchmark para os conjuntos PT e PTEN.

Exemplo:
    python benchmark_runner.py --datasets pt pten --models Qwen/Qwen3-8B
"""

import argparse
import gc
from pathlib import Path
import subprocess
import sys
import time
from datetime import timedelta

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSES_DIR = PROJECT_DIR / "experiments" / "generated_responses"
DEFAULT_EVALUATIONS_DIR = PROJECT_DIR / "experiments" / "generated_evaluations"
DATASETS = ("pt", "pten")

MODELS_TO_BENCHMARK = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-4-E4B-it",
    "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    "Polygl0t/Tucano2-qwen-3.7B-Instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
]

GPU_UTIL_DEFAULT = 0.90
GPU_UTIL_LARGE = 0.88


def install_dependencies():
    """Instala dependências de inferência quando solicitado explicitamente."""
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "vllm==0.19.1", "bitsandbytes==0.45.1", "hf-transfer==0.1.9",
        "psutil", "langdetect", "janome", "ja_sentence_segmenter", "spacy",
        "nltk", "accelerate",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "transformers>=5.5.0",
    ])


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def force_gpu_cleanup(wait_for_free=False):
    """Libera memória da GPU entre modelos, quando disponível."""
    import torch

    if sys.platform != "win32":
        for pattern in ("vllm.v1.engine.core", "EngineCore"):
            subprocess.run(["pkill", "-9", "-f", pattern], check=False)

    gc.collect()
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if not wait_for_free:
        return

    deadline = time.time() + 30
    while time.time() < deadline:
        free, total = torch.cuda.mem_get_info()
        if free > total * 0.85:
            return
        time.sleep(2)
        torch.cuda.empty_cache()


def response_path(responses_dir, dataset, safe_model_name):
    return responses_dir / f"{dataset}_input_response_data_{safe_model_name}.jsonl"


def responses_exist(responses_dir, safe_model_name, datasets):
    return all(response_path(responses_dir, dataset, safe_model_name).exists() for dataset in datasets)


def build_inference_cmd(model, responses_dir, datasets, dry_run=False):
    is_large = any(size in model.lower() for size in ("13b", "30b", "32b", "34b", "70b"))
    gpu_util = GPU_UTIL_LARGE if is_large else GPU_UTIL_DEFAULT
    command = [
        sys.executable, str(PROJECT_DIR / "universal_inference.py"),
        "--model_name", model,
        "--output-dir", str(responses_dir),
        "--datasets", *datasets,
        "--batch_size", "16" if is_large else "64",
        "--gpu_memory_utilization", str(gpu_util),
    ]
    if is_large:
        command.append("--load_in_8bit")
    if dry_run:
        command.append("--dry-run")
    return command


def build_evaluation_cmd(responses_dir, evaluations_dir, datasets, dry_run=False):
    command = [
        sys.executable, str(PROJECT_DIR / "run_eval_only.py"),
        "--responses-dir", str(responses_dir),
        "--evaluations-dir", str(evaluations_dir),
        "--languages", *datasets,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def run_benchmark(args):
    responses_dir = args.responses_dir.resolve()
    evaluations_dir = args.evaluations_dir.resolve()
    responses_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Respostas: {responses_dir}")
    print(f"Avaliações: {evaluations_dir}")

    for model in args.models:
        safe_model_name = model.replace("/", "__")
        print(f"\n{'=' * 60}\nModelo: {model}\n{'=' * 60}")
        if args.skip_existing and responses_exist(responses_dir, safe_model_name, args.datasets):
            print("Respostas já existem; inferência ignorada.")
            continue

        if not args.dry_run:
            force_gpu_cleanup(wait_for_free=True)
        command = build_inference_cmd(model, responses_dir, args.datasets, args.dry_run)
        print("Inferência:", subprocess.list2cmdline(command))
        subprocess.run(command, check=True)
        if not args.dry_run:
            force_gpu_cleanup()

    if not args.skip_evaluation:
        command = build_evaluation_cmd(responses_dir, evaluations_dir, args.datasets, args.dry_run)
        print("\nAvaliação:", subprocess.list2cmdline(command))
        subprocess.run(command, check=True)

    print(f"\nConcluído em {format_time(time.time() - started_at)}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["pt"])
    parser.add_argument("--models", nargs="+", default=MODELS_TO_BENCHMARK)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--evaluations-dir", type=Path, default=DEFAULT_EVALUATIONS_DIR)
    parser.add_argument("--skip-existing", action="store_true", help="Não gera respostas já existentes.")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--install-dependencies", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Valida comandos sem carregar modelos.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.install_dependencies:
        install_dependencies()
    run_benchmark(arguments)
