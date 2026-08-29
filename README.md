# Portuguese-IFEval: Extending M-IFEval to Portuguese

<span style="display: inline; gap: 5px;">
<a href="https://github.com/ManuPassinato/M-IFEvalFork/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-purple?color=%23b304d6" height="20"/></a>
<a href="https://www.arxiv.org/abs/2502.04688"><img src="https://img.shields.io/badge/M--IFEval-ArXiv-gray?logo=arxiv&labelColor=%23B31B1B" height="20"/></a>
</span>

> This repository extends [M-IFEval: Multilingual Instruction-Following Evaluation](https://www.arxiv.org/abs/2502.04688) with a language-aware Portuguese benchmark while preserving the original multilingual evaluation setting.

## Overview

Instruction-following benchmarks are essential for measuring whether large language models comply with verifiable user constraints. The original [IFEval](https://arxiv.org/abs/2311.07911) introduced deterministic evaluation in English, and M-IFEval extended that framework to Spanish, French, and Japanese.

This fork adds **Portuguese** through semantic regionalization rather than literal translation. The corrected paper reports 535 validated prompts containing up to three verifiable instructions and evaluates 25 proprietary, open-weight, and Portuguese-tuned models. The benchmark adapts existing instruction types to natural Portuguese usage and introduces six Portuguese-specific deterministic constraints targeting orthography, morphosyntax, pragmatics, and discourse. Their validators are rule-based and operate on explicit surface-form conditions rather than full syntactic or semantic parsing.

## Benchmark Construction

Portuguese-IFEval is built through semantic regionalization of the original IFEval instruction categories rather than direct translation. Candidate prompts were generated from scratch with `Qwen/Qwen3-235B-A22B-FP8`, with each prompt containing between one and three verifiable instructions and an additional high-level directive such as an email, summary, dialogue, recipe, or social-media post.

The prompts were then rewritten with the same model to improve linguistic naturalness while preserving the verifiable constraints. This process produced 595 candidate prompts. An automatic satisfiability-filtering stage was applied first, followed by structured manual verification by three native Portuguese-speaking NLP researchers. Sixty prompts were removed during manual review, resulting in the final set of **535 validated prompts** used in Portuguese-IFEval.

## How to Run

### Local Setup

Clone this fork and enter its directory:

```bash
git clone https://github.com/ManuPassinato/M-IFEvalFork.git
cd M-IFEvalFork
```

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The evaluation registry imports language-specific modules for the multilingual benchmark. Install the required spaCy models and NLTK tokenizer data before running an evaluation:

```bash
python -m spacy download pt_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download xx_sent_ud_sm
python -m nltk.downloader punkt_tab
```

> **Environment note:** `vllm` and `bitsandbytes` are primarily intended for supported Linux/GPU environments. API response files and the rule-based evaluator can be used without running a local vLLM model.

### Choose the Prompt Set

This repository uses two Portuguese evaluation settings:

| Setting | Input file | Meaning |
| --- | --- | --- |
| **PT** | `data/pt_input_data_FINAL_CLEAN.jsonl` | Portuguese-IFEval: the language-aware, semantically regionalized benchmark introduced in this work |
| **PT_EN** | `data/pten_input_data.jsonl` | Translated IFEval: all 541 original English IFEval prompts translated into Portuguese with GPT-5, with the required evaluator adaptations |

`PT_EN` is the translated Portuguese control benchmark. It is not a bilingual or mixed Portuguese-English split. The repository uses the shorter `pten` prefix in filenames for this setting.

For exact paper reproduction, the PT input must correspond to the final 535-prompt validated split described in the corrected paper.

### Prompt and Response Format

Each line in a prompt JSONL file contains the text sent to the model and the metadata later used by the deterministic evaluator:

```json
{"key": 0, "instruction_id_list": ["pt:keywords:existence"], "prompt": "...", "kwargs": [{"keywords": ["saudade", "saúde"]}]}
```

The generated response file must contain one output for every input prompt:

```json
{"prompt": "...", "response": "..."}
```

The `prompt` value must be preserved exactly because the evaluator uses it to match each response to its instruction metadata.

### Response Generation Workflow

The evaluation pipeline has two distinct stages:

1. Select a prompt dataset (`PT` or `PT_EN`).
2. Run a model over every `prompt` entry in the selected JSONL file.
3. Store each generated output together with its original prompt in a response JSONL file.
4. Verify that every input prompt has exactly one non-empty response.
5. Pass the prompt file and the generated response file to `evaluation_main`.

In short:

`prompt JSONL → model inference → response JSONL → deterministic evaluation`

### Generate Responses from the Prompts

#### API and Configured Backends

`get_responses.py` supports the OpenAI, Anthropic, and vLLM backends configured in `SUPPORTED_MODELS`. To add a model, map its exact identifier to the appropriate backend:

```python
SUPPORTED_MODELS = {
    # Existing entries...
    "gpt-5": "openai",
    "claude-haiku-4-5-20251001": "anthropic",
    "Qwen/Qwen3-8B": "vllm",
}
```

Set the credential required by the selected provider. For example:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export ANTHROPIC_API_KEY="YOUR_KEY"
```

On Windows PowerShell, use `$env:OPENAI_API_KEY="YOUR_KEY"` or `$env:ANTHROPIC_API_KEY="YOUR_KEY"`.

Run response generation with the same model identifier added to `SUPPORTED_MODELS`:

```bash
python get_responses.py --model_name gpt-5
```

The script performs the following steps:

1. Finds every file matching `data/*_input_data.jsonl`.
2. Reads the `prompt` field from every JSONL row.
3. Sends each prompt to the selected model without adding benchmark metadata to the request.
4. Writes only `prompt` and `response` to a new JSONL file.
5. Replaces `/` with `__` in model identifiers when constructing the output filename.

> **API usage note:** one invocation processes every matching input file, not only PT or `PT_EN`. Review the selected `paths` before running a paid API model.

For example, responses generated from `data/pten_input_data.jsonl` with `gpt-5` are written to:

```text
data/pten_input_response_data_gpt-5.jsonl
```

The `PT_EN` file already matches the helper's input pattern. The regionalized file `pt_input_data_FINAL_CLEAN.jsonl` does not match `*_input_data.jsonl`; when using `get_responses.py` for PT, point its `paths` list to that file explicitly or use a working filename ending in `_input_data.jsonl`.

#### Local Open-Weight Models

`universal_inference.py` reads the multilingual prompt files, prefers `{language}_input_data_FINAL_CLEAN.jsonl`, applies the model chat template, generates one response per prompt, and writes `{language}_input_response_data_{model}.jsonl`:

```bash
python universal_inference.py \
  --model_name Qwen/Qwen3-8B \
  --data_dir ./data \
  --gpu_memory_utilization 0.90 \
  --max_model_len 8096
```

This produces the Portuguese response file:

```text
data/pt_input_response_data_Qwen__Qwen3-8B.jsonl
```

The paper reports three runs for open-weight models and presents their mean and standard deviation. The universal inference pipeline uses a default maximum generation length of 32,768 tokens, an effective generation limit capped by the tokenizer-detected context length, a default batch size of 4, `gpu_memory_utilization=0.90` for vLLM (capped at 0.95), and a default `max_model_len=8096` unless further constrained by the tokenizer.

Qwen 3 is the only reported open-weight family evaluated with stochastic decoding: thinking mode is enabled with `temperature=0.7` and `top_p=0.8`. Qwen 3.5 is evaluated with thinking mode disabled. The remaining reported open-weight families use deterministic decoding with temperature zero; under Transformers, this corresponds to `do_sample=False` and `repetition_penalty=1.2`. Closed/API models use the default generation configuration exposed by each provider, without manual overrides to decoding or generation parameters.

Before evaluation, verify that the response file contains one non-empty response for every prompt in the selected input file. Missing or modified prompt strings prevent exact matching.

### Evaluate the Generated Responses

Evaluate a response file generated for Portuguese-IFEval:

```bash
python -m evaluation_main \
  --input_data=./data/pt_input_data_FINAL_CLEAN.jsonl \
  --input_response_data=./data/pt_input_response_data_Qwen__Qwen3-8B.jsonl \
  --output_dir=./evaluation_runs/pt_Qwen__Qwen3-8B
```

Evaluate a response file generated for the translated `PT_EN` benchmark:

```bash
python -m evaluation_main \
  --input_data=./data/pten_input_data.jsonl \
  --input_response_data=./data/pten_input_response_data_openai__gpt-5.jsonl \
  --output_dir=./evaluation_runs/pten_openai__gpt-5
```

Each command writes `eval_results_strict.jsonl` and `eval_results_loose.jsonl`. The evaluator computes compliance at the instruction level, and prompt-level success requires that **all** instructions attached to a prompt be satisfied. This requirement applies to both Strict and Loose evaluation. Unless otherwise stated, the paper's overall benchmark tables report prompt-level performance, while the language-specific comparison in Table 5 reports instruction-level Strict accuracy. The same command structure applies to the other language splits when the input and response files contain identical prompt sets.

## Language Coverage

M-IFEval remains a multilingual benchmark. This fork covers the following evaluation contexts:

- **English:** reference alignment with the original IFEval benchmark and cross-language comparison.
- **Spanish:** language-specific instructions and evaluation artifacts inherited from M-IFEval.
- **French:** language-specific instructions and evaluation artifacts inherited from M-IFEval.
- **Japanese:** language-specific instructions and evaluation artifacts inherited from M-IFEval.
- **Portuguese:** the central contribution of this fork, with regionalized instructions, new language-specific constraints, datasets, utilities, tests, model responses, and evaluation outputs.

Portuguese extends the benchmark; it does not replace the multilingual context used for cross-language analysis. `PT_EN` is used as a translation-based Portuguese baseline, while `PT` refers to Portuguese-IFEval.

## What This Fork Adds for Portuguese

The Portuguese extension regionalizes the original verifiable instruction categories and adds six Portuguese-specific constraints. The corresponding validators are deterministic, rule-based checkers based on explicit surface-form conditions rather than full syntactic parsing:

- **Cedilla frequency:** require exactly `N` occurrences of the literal character `ç` after lowercasing the response.
- **Tilde absence:** reject responses containing any character from the validator's closed set `ã`, `õ`, `ñ`, `Ã`, `Õ`, or `Ñ`.
- **Grave accent (crase):** require at least one case-insensitive match to an accepted crase surface form such as `à`, `às`, `àquele`, `àquela`, `àqueles`, `àquelas`, or `àquilo`.
- **Mesoclisis:** require at least one hyphenated token matching the validator's mesoclitic surface pattern `stem-pronoun-suffix`, with pronouns and future/conditional endings drawn from predefined lists. The checker does not verify broader grammatical or semantic appropriateness.
- **Second-person plural address (`vós`):** use a lexical approximation that rejects competing forms such as `você`, `vocês`, and `tu`, requires at least one token from the `vós` family, and looks within a five-token window for a compatible verb form.
- **Four porquês:** require exactly one occurrence of a form from the `porque`/`porquê`/`por que`/`por quê` family and apply deterministic contextual heuristics for the target subtype specified by the prompt.

The implementation is provided by `instructions/pt_instructions.py`, `instruction_utils/pt_instructions_util.py`, and the Portuguese entries in `instructions_registry.py`. Focused checks are available in `tests/test_pt_instructions2.py`.

## Portuguese Dataset and Artifacts

The main Portuguese benchmark assets are:

| Path | Description |
| --- | --- |
| `data/pt_input_data_FINAL_CLEAN.jsonl` | Portuguese-IFEval prompt collection used for the regionalized PT setting |
| `data/pten_input_data.jsonl` | `PT_EN`: the complete 541-prompt IFEval benchmark translated into Portuguese |
| `data/pt_input_response_data_*.jsonl` | Historical model responses for the Portuguese evaluation |
| `evaluations/pt_*` | Portuguese evaluation outputs |
| `evaluations/pten_*` | Translated `PT_EN` evaluation outputs |

## Experimental Results

All values below come from the corrected EMNLP paper. In the overall benchmark tables, Strict and Loose are reported at the prompt level: a prompt is successful only when every attached instruction is satisfied. The Portuguese-specific comparison later in this README follows Table 5 of the paper and reports instruction-level Strict accuracy. For open-weight models, `mean ± standard deviation` is reported over three runs; API models use one provider-default run.

### Portuguese-IFEval Overall

Table 4 of the paper reports performance on the full regionalized Portuguese-IFEval instruction set.

| Group | Model | Strict | Loose | Average |
| --- | --- | ---: | ---: | ---: |
| Instruction-tuned | Llama 3.1 Instruct 8B | 26.15 ± 0.18 | 29.94 ± 0.62 | 28.05 |
| Instruction-tuned | Llama 3.2 Instruct 3B | 21.64 ± 0.70 | 24.82 ± 0.54 | 23.23 |
| Instruction-tuned | Llama 3.2 Instruct 1B | 11.95 ± 0.20 | 13.70 ± 0.66 | 12.83 |
| Instruction-tuned | Gemma 4 E2B | 36.56 ± 1.06 | 40.17 ± 1.31 | 38.37 |
| Instruction-tuned | **Gemma 4 E4B** | **44.45 ± 1.06** | **48.23 ± 1.31** | **46.34** |
| Instruction-tuned | Qwen 3 8B | 42.98 ± 0.66 | 46.83 ± 1.05 | 44.91 |
| Instruction-tuned | Qwen 3 4B | 39.50 ± 0.51 | 43.30 ± 0.10 | 41.40 |
| Instruction-tuned | Qwen 3 1.7B | 31.55 ± 0.20 | 35.70 ± 0.17 | 33.63 |
| Instruction-tuned | Qwen 3 0.6B | 23.73 ± 0.81 | 27.56 ± 0.57 | 25.65 |
| Instruction-tuned | Qwen 3.5 9B | 42.99 ± 0.00 | 47.29 ± 0.00 | 45.14 |
| Instruction-tuned | Qwen 3.5 4B | 39.94 ± 0.60 | 43.18 ± 0.19 | 41.56 |
| Instruction-tuned | Qwen 3.5 2B | 26.98 ± 1.03 | 30.59 ± 0.76 | 28.79 |
| Instruction-tuned | Qwen 3.5 0.8B | 17.13 ± 0.43 | 19.88 ± 0.43 | 18.51 |
| Instruction-tuned | Nemotron 3 Nano 4B | 39.63 ± 0.00 | 42.43 ± 0.00 | 41.03 |
| API | Claude 4.5 Haiku | 48.95 | 53.92 | 51.44 |
| API | GPT-4o | 48.57 | 53.92 | 51.25 |
| API | **GPT-5** | **65.20** | **67.30** | **66.25** |
| API | Gemini 2.5 Pro | 58.51 | 63.10 | 60.81 |
| API / Portuguese-tuned | Sabiá 3 | 47.99 | 52.96 | 50.48 |
| Portuguese-tuned | **Gaia 4B** | **28.13 ± 0.40** | **32.13 ± 0.34** | **30.13** |
| Portuguese-tuned | Tucano 2 Qwen 0.5B | 12.84 ± 1.09 | 16.47 ± 0.60 | 14.66 |
| Portuguese-tuned | Tucano 2 Qwen 1.5B | 18.69 ± 0.73 | 22.33 ± 0.71 | 20.51 |
| Portuguese-tuned | Tucano 2 Qwen 3.7B | 22.30 ± 0.51 | 25.72 ± 0.76 | 24.01 |
| Portuguese-tuned | Tucano Instruct 1B | 4.93 ± 0.35 | 5.63 ± 0.22 | 5.28 |
| Portuguese-tuned | Tucano Instruct 2B | 3.25 ± 0.59 | 4.23 ± 0.68 | 3.74 |

**GPT-5** is the strongest model overall, **Gemma 4 E4B** leads the instruction-tuned group, and **Gaia 4B** leads the open Portuguese-tuned group.

### Portuguese-IFEval vs. Translated PT_EN

Table 2 of the paper compares Portuguese-IFEval (`PT`) with the direct Portuguese translation of IFEval (`PT_EN`). `Δ` is the translated score minus the Portuguese-IFEval score, so a larger value indicates a larger gap hidden by translation-based evaluation.

| Group | Model | PT Strict | PT Loose | PT_EN Strict | PT_EN Loose | Δ Strict | Δ Loose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruction-tuned | Llama 3.1 Instruct 8B | 27.28 ± 0.62 | 31.21 ± 0.35 | 56.87 ± 0.70 | 61.86 ± 0.83 | 29.59 | 30.65 |
| Instruction-tuned | **Gemma 4 E4B** | 46.69 ± 0.72 | 50.50 ± 0.49 | 71.78 ± 0.09 | 73.69 ± 0.09 | 25.09 | 23.19 |
| Instruction-tuned | Qwen 3 8B | 45.13 ± 0.93 | 48.88 ± 0.53 | 69.25 ± 0.09 | 72.64 ± 0.00 | 24.12 | 23.77 |
| Instruction-tuned | Nemotron 3 Nano 4B | 39.70 ± 0.00 | 42.51 ± 0.00 | 71.66 ± 0.09 | 73.57 ± 0.00 | 31.96 | 31.06 |
| API | Claude 4.5 Haiku | 48.95 | 53.92 | 65.19 | 70.37 | 16.24 | 16.45 |
| API | GPT-4o | 48.57 | 53.92 | 65.00 | 69.26 | 16.43 | 15.34 |
| API | **GPT-5** | **65.20** | **67.30** | **73.15** | **75.93** | 7.95 | 8.63 |
| API | Gemini 2.5 Pro | 58.51 | 63.10 | 67.22 | 70.56 | 8.71 | 7.76 |
| API / Portuguese-tuned | Sabiá 3 | 47.99 | 52.96 | 67.22 | 69.81 | 19.23 | 16.85 |
| Portuguese-tuned | **Gaia 4B** | 30.02 ± 0.97 | 33.83 ± 0.88 | 51.14 ± 0.17 | 60.44 ± 0.15 | 21.11 | 26.61 |
| Portuguese-tuned | Tucano 2 Qwen 3.7B | 23.10 ± 0.87 | 26.53 ± 1.02 | 43.38 ± 0.46 | 46.70 ± 0.97 | 20.28 | 20.17 |

The translated benchmark produces substantially higher scores and compresses model differences. For example, GPT-5 leads Gemma 4 E4B by only 1.37 strict points on `PT_EN`, while the gap is approximately 18 points on Portuguese-IFEval.

### Portuguese-Specific Instructions

The PT column of Table 5 reports strict instruction-level accuracy restricted to language-specific Portuguese instructions.

| Group | Model | PT-Specific Strict |
| --- | --- | ---: |
| Instruction-tuned | Llama 3.1 Instruct 8B | 25.53 |
| Instruction-tuned | Llama 3.2 Instruct 3B | 23.85 |
| Instruction-tuned | Llama 3.2 Instruct 1B | 18.92 |
| Instruction-tuned | Gemma 4 E2B | 36.42 |
| Instruction-tuned | **Gemma 4 E4B** | **46.58** |
| Instruction-tuned | Qwen 3 8B | 32.11 |
| Instruction-tuned | Qwen 3 4B | 28.79 |
| Instruction-tuned | Qwen 3 1.7B | 21.57 |
| Instruction-tuned | Qwen 3 0.6B | 13.85 |
| Instruction-tuned | Qwen 3.5 9B | 37.08 |
| Instruction-tuned | Qwen 3.5 4B | 31.12 |
| Instruction-tuned | Qwen 3.5 2B | 21.50 |
| Instruction-tuned | Qwen 3.5 0.8B | 17.23 |
| Instruction-tuned | Nemotron 3 Nano 4B | 21.94 |
| API | Claude 4.5 Haiku | 59.88 |
| API | GPT-4o | 49.13 |
| API | **GPT-5** | **74.60** |
| API | Gemini 2.5 Pro | 66.67 |
| API / Portuguese-tuned | Sabiá 3 | 49.54 |
| Portuguese-tuned | **Gaia 4B** | **33.26** |
| Portuguese-tuned | Tucano 2 Qwen 0.5B | 20.30 |
| Portuguese-tuned | Tucano 2 Qwen 1.5B | 19.13 |
| Portuguese-tuned | Tucano 2 Qwen 3.7B | 27.08 |
| Portuguese-tuned | Tucano Instruct 1B | 10.34 |
| Portuguese-tuned | Tucano Instruct 2B | 12.63 |

These results show that strong performance on a directly translated benchmark does not guarantee reliable compliance with fine-grained Portuguese constraints.

## References

If you use the multilingual foundation of this work, cite M-IFEval:

```bibtex
@inproceedings{Dussolle2025MIFEval,
  title={M-IFEval: Multilingual Instruction-Following Evaluation},
  author={Dussolle, Antoine and Cardeña, Andrea and Sato, Shota and Devine, Peter},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={6161--6176},
  year={2025},
  url={https://arxiv.org/abs/2502.04688}
}
```

The original IFEval benchmark can be cited as:

```bibtex
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```

The Portuguese-IFEval paper is currently an anonymous submission. Its final citation should be added after publication metadata becomes available.

## License

This project is licensed under the Apache License 2.0. See `LICENSE.txt` for details.

This repository includes code derived from [Instruction Following Evaluation for Large Language Models](https://github.com/google-research/google-research/tree/master/instruction_following_eval), also licensed under Apache 2.0.

