"""Small, stage-specific YAML loader. No model or API initialization here."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import math
from functools import lru_cache

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
STAGES = {"data_gen", "inference", "metrics"}
ENTRY_STAGES = {
    "gen_input_data.py": "data_gen", "model_handler.py": "data_gen",
    "universal_inference.py": "inference", "get_responses.py": "inference",
    "benchmark_runner.py": "inference", "evaluation_main.py": "metrics",
    "run_eval_only.py": "metrics", "calc_eval.py": "metrics", "tables.py": "metrics",
}


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _cli_value(flag):
    value = None
    for index, arg in enumerate(sys.argv[1:], 1):
        if arg == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a filename")
            value = sys.argv[index + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def config_path(stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown configuration stage: {stage}")
    value = os.getenv(f"IFEVAL_{stage.upper()}_CONFIG")
    flag = "--config" if ENTRY_STAGES.get(Path(sys.argv[0]).name) == stage else None
    if Path(sys.argv[0]).name == "benchmark_runner.py" and stage == "metrics":
        flag = "--metrics-config"
    if flag:
        value = _cli_value(flag) or value
    return Path(value).resolve() if value else PROJECT_DIR / f"{stage}.yml"


@lru_cache(maxsize=None)
def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def _validate_section(stage, section, value):
    """Validate operational settings without importing unrelated stages."""
    if not isinstance(value, dict):
        raise ValueError(f"{stage}.{section} must be a mapping")
    boolean_fields = {
        'universal_inference': 'load_in_8bit prefer_transformers transformers_trust_remote_code tokenizer_trust_remote_code vllm_trust_remote_code vllm_enforce_eager chat_template_tokenize chat_template_add_generation_prompt qwen3_enable_thinking qwen35_enable_thinking qwen3_do_sample default_do_sample return_full_text',
        'get_responses': 'use_progress_bar',
        'model_handler': 'enable_thinking fallback_enable_thinking',
    }
    positive_fields = {
        'universal_inference': 'batch_size max_new_tokens tensor_parallel_size maximum_context_tokens context_cap minimum_candidate_words minimum_clean_words fallback_characters progress_interval',
        'benchmark_runner': 'batch_size large_batch_size divider_width',
        'get_responses': 'vllm_max_tokens anthropic_max_tokens',
        'model_handler': 'retry_attempts resp_max_tokens_default',
        'evaluation_main': 'divider_width',
        'calc_eval': 'model_width prompt_width score_width separator_width',
    }
    nonnegative_fields = {
        'universal_inference': 'qwen3_temperature vllm_qwen3_temperature vllm_default_temperature reasoning_marker_max_index',
        'get_responses': 'vllm_temperature anthropic_temperature',
        'model_handler': 'retry_wait_seconds resp_temperature_default',
        'benchmark_runner': 'cleanup_wait_seconds cleanup_poll_seconds',
        'calc_eval': 'score_decimals',
    }
    probability_fields = {
        'universal_inference': 'gpu_memory_utilization gpu_util_cap qwen3_top_p vllm_qwen3_top_p',
        'benchmark_runner': 'gpu_util_default gpu_util_large minimum_free_memory_fraction',
        'model_handler': 'resp_top_p_default',
    }
    rules = [
        (boolean_fields, lambda v: type(v) is bool, 'a boolean'),
        (positive_fields, lambda v: type(v) is int and v > 0, 'a positive integer'),
        (nonnegative_fields, lambda v: type(v) in (int, float) and math.isfinite(v) and v >= 0, 'a finite nonnegative number'),
        (probability_fields, lambda v: type(v) in (int, float) and 0 < v <= 1, 'in (0, 1]'),
    ]
    for fields, predicate, description in rules:
        for key in fields.get(section, '').split():
            if not predicate(value.get(key)):
                raise ValueError(f"{stage}.{section}.{key} must be {description}")
    if section == 'universal_inference':
        # <= 0 selects the tokenizer's detected context limit, as in the original.
        if type(value.get('max_model_len')) is not int:
            raise ValueError('universal_inference.max_model_len must be an integer')
        if 'transformers_stop_strings' in value:
            raise ValueError('transformers_stop_strings was unused and has been removed; use the cleaning patterns')
        penalty = value.get('repetition_penalty')
        if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
            raise ValueError('universal_inference.repetition_penalty must be finite and positive')
    if section == 'calc_eval' and type(value.get('score_decimals')) is not int:
        raise ValueError('calc_eval.score_decimals must be an integer')
    if section == 'evaluation_main':
        for key in ('strict_filename', 'loose_filename'):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f'evaluation_main.{key} must be a nonempty filename')
        if value['strict_filename'] == value['loose_filename']:
            raise ValueError('Strict and Loose outputs must have different filenames')
    if section == 'paths':
        for key in ('generated_evaluations', 'report_evaluations', 'report_output_dir'):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f"metrics.paths.{key} must be a nonempty path")
    if section == 'nlp':
        for key in ('spanish_model', 'multilingual_sentence_model', 'portuguese_model', 'sentence_component', 'nltk_resource'):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f"metrics.nlp.{key} must be a nonempty string")
        disabled = value.get('portuguese_disable')
        if not isinstance(disabled, list) or any(not isinstance(item, str) for item in disabled):
            raise ValueError('metrics.nlp.portuguese_disable must be a list of strings')
    if section == 'gen_input_data':
        if value.get('output_mode') not in ('a', 'w'):
            raise ValueError('gen_input_data.output_mode must be a or w')
        sizes = value.get('combination_sizes')
        if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError('gen_input_data.combination_sizes must contain positive integers')
    if section == 'instruction_defaults':
        for language, defaults in value.items():
            _validate_instruction_defaults(language, defaults)


def _resolve_report_paths(value):
    if isinstance(value, dict) and value.get('report_evaluations') is None:
        value['report_evaluations'] = value.get('generated_evaluations')
    return value


def load_section(stage, section, key=None, path=None):
    """Read only a requested section; NLP never resolves metrics.source/io."""
    if stage not in STAGES:
        raise ValueError(f"Unknown configuration stage: {stage}")
    selected = Path(path).resolve() if path else config_path(stage)
    raw = _read(str(selected)).get(section)
    if not isinstance(raw, dict):
        raise ValueError(f"{stage}.{section} must be a mapping")
    if key is not None:
        if key not in raw:
            raise ValueError(f"Missing {stage}.{section}.{key}")
        value = copy.deepcopy(raw[key])
        if stage == 'data_gen' and section == 'instruction_defaults':
            _validate_instruction_defaults(key, value)
        return value
    value = copy.deepcopy(raw)
    if stage == 'metrics' and section == 'paths':
        _resolve_report_paths(value)
    _validate_section(stage, section, value)
    return value


def _validate_instruction_defaults(language, value):
    if not isinstance(value, dict):
        raise ValueError(f'instruction_defaults.{language} must be a mapping')
    for key, setting in value.items():
        valid = (isinstance(setting, list) and bool(setting) and all(isinstance(item, str) for item in setting)) or (type(setting) is int and setting > 0)
        if not valid:
            raise ValueError(f'instruction_defaults.{language}.{key} must be a positive integer or nonempty string list')
    for unit in ('words', 'letters'):
        low, high = f'num_{unit}_lower_limit', f'num_{unit}_upper_limit'
        if low in value and high in value and value[low] > value[high]:
            raise ValueError(f'instruction_defaults.{language}: {low} exceeds {high}')


def load_config(stage, path=None):
    """Load one stage; metrics imports only the inference I/O contract."""
    if stage not in STAGES:
        raise ValueError(f"Unknown configuration stage: {stage}")
    selected = Path(path).resolve() if path else config_path(stage)
    value = copy.deepcopy(_read(str(selected)))
    if stage == "metrics":
        _resolve_report_paths(value.get('paths'))
        source = value.get("source")
        if not isinstance(source, dict):
            raise ValueError("metrics.source must be a mapping")
        if source.get("mode") == "inference":
            reference = Path(source["config"])
            if not reference.is_absolute():
                reference = selected.parent / reference
            # A full benchmark must evaluate the exact I/O contract used to infer.
            if Path(sys.argv[0]).name == "benchmark_runner.py":
                reference = config_path("inference")
            elif Path(sys.argv[0]).name in {"run_eval_only.py", "evaluation_main.py"}:
                override = _cli_value("--inference-config")
                if override:
                    reference = Path(override).resolve()
            value["io"] = copy.deepcopy(_read(str(reference.resolve()))["io"])
        elif source.get("mode") == "explicit":
            value["io"] = source["io"]
        else:
            raise ValueError("metrics.source.mode must be inference or explicit")
    sections = {
        "data_gen": ("gen_input_data", "model_handler", "instruction_defaults"),
        "inference": ("io", "universal_inference", "get_responses", "benchmark_runner"),
        "metrics": ("io", "paths", "run_eval_only", "evaluation_main", "calc_eval", "tables", "nlp"),
    }
    for section in sections[stage]:
        _validate_section(stage, section, value.get(section))
    if "io" in value:
        io = value["io"]
        if not isinstance(io.get("input_files"), dict):
            raise ValueError("io.input_files must be a mapping")
        for language in ("pt", "pten"):
            if not isinstance(io["input_files"].get(language), str) or not io["input_files"][language]:
                raise ValueError(f"io.input_files.{language} must be a nonempty path")
        for key in ("data_dir", "responses_dir", "response_marker", "response_extension", "response_template", "input_template"):
            if not isinstance(io.get(key), str) or not io[key]:
                raise ValueError(f"io.{key} must be a nonempty string")
        if not isinstance(io.get("language_aliases"), dict):
            raise ValueError("io.language_aliases must be a mapping")
        expected = "{dataset}" + io["response_marker"] + "{model}" + io["response_extension"]
        if io["response_template"] != expected:
            raise ValueError("io.response_template must agree with response_marker and response_extension")
        if not io["input_template"].startswith("{dataset}") or io["input_template"] == "{dataset}":
            raise ValueError("io.input_template must start with {dataset}")
    return value
