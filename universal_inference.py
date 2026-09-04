
from config import load_config, project_path
CONFIG = load_config('inference')
CFG = CONFIG['universal_inference']
IO = CONFIG["io"]

import os
import argparse
import gc
import sys
import traceback
import re
import importlib.util
import json
from pathlib import Path

HF_TOKEN = os.getenv(CFG['hf_token_env']) or os.getenv(CFG['hf_token_fallback_env'])
DEFAULT_MAX_NEW_TOKENS = CFG['max_new_tokens']
DEFAULT_BATCH_SIZE = CFG['batch_size']
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = project_path(IO["responses_dir"])
DATASET_INPUTS = {name: project_path(path) for name, path in IO["input_files"].items()}

def configure_pipeline_for_batching(hf_pipeline, tokenizer):
    """Configura pad token para evitar warnings de padding."""
    if tokenizer.pad_token_id is None:
        eos_id = tokenizer.eos_token_id
        if eos_id is None and hasattr(hf_pipeline.model.config, "eos_token_id"):
            eos_id = hf_pipeline.model.config.eos_token_id
        if eos_id is not None:
            tokenizer.pad_token_id = eos_id
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

def build_generation_config(tokenizer, is_qwen_3, max_generation_tokens):
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig()
    gen_cfg.max_new_tokens = max_generation_tokens
    gen_cfg.max_length = CFG['generation_max_length']
    gen_cfg.eos_token_id = tokenizer.eos_token_id
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    if is_qwen_3:
        gen_cfg.do_sample = CFG['qwen3_do_sample']
        gen_cfg.temperature = CFG['qwen3_temperature']
        gen_cfg.top_p = CFG['qwen3_top_p']
    else:
        gen_cfg.do_sample = CFG['default_do_sample']
        gen_cfg.repetition_penalty = CFG['repetition_penalty']
    return gen_cfg

# Tenta importar vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Aviso: vLLM nao encontrado; o modo Transformers sera usado se a inferencia for executada.")

# --- A FUNÇÃO DE LIMPEZA DEFINITIVA (V8 - NUCLEAR) ---
def clean_response(text, stop_tokens_str):
    # Second argument is retained for compatibility; cleaning is defined by CFG patterns.
    if not text: return ""

    # 1. REMOVE BLOCO DE PENSAMENTO FECHADO (<think>...</think>)
    text = re.sub(CFG['closed_thinking_pattern'], '', text, flags=re.DOTALL)

    if CFG["closing_thinking_tag"] in text:
        text = re.sub(CFG['leading_thinking_pattern'], '', text, flags=re.DOTALL)

    # 2. SEGURANÇA PARA PENSAMENTO NÃO FINALIZADO
    if CFG["opening_thinking_tag"] in text:
        parts = text.split(CFG["opening_thinking_tag"])
        if len(parts[0].strip()) > 0:
            text = parts[0]
        else:
            return ""

    # 2b. PENSAMENTO ESTRUTURADO SEM TAGS (Qwen3.5 vLLM leak)
    # Detecta blocos "1. **Analyze..." e tenta recuperar conteúdo real após eles
    structured_thinking = re.match(
        CFG['structured_thinking_pattern'],
        text, re.IGNORECASE
    )
    if structured_thinking:
        # Tenta encontrar conteúdo real após o bloco de pensamento
        # Padrões que indicam fim do thinking e início da resposta
        end_markers = CFG['structured_thinking_end_patterns']
        found = False
        for pattern in end_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = text[match.end():].strip()
                if len(candidate.split()) >= CFG["minimum_candidate_words"]:
                    text = candidate
                    found = True
                    break
        if not found:
            return ""  # pensamento sem resposta real — descarta

    # 3. REMOVE MARCADORES DE PENSAMENTO EM INGLÊS
    markers = CFG['reasoning_markers']

    for m in markers:
        if m in text:
            index = text.find(m)
            if index < CFG["reasoning_marker_max_index"]:
                parts = text.split("\n\n")
                if len(parts) > 1:
                    text = "\n\n".join(parts[1:])
                else:
                    return ""
            else:
                text = text[:index]

    # 4. LIMPEZA FINAL
    text = text.lstrip(CFG['leading_newlines']).lstrip(CFG["leading_strip_fragments"][0]).lstrip(CFG["leading_strip_fragments"][1]).strip()
    text = re.sub(CFG['special_token_pattern'], '', text)
    text = re.sub(CFG['sentence_token_pattern'], '', text)
    text = text.replace(CFG['pad_token'], "")
    text = text.replace(CFG['eos_token'], "")
    text = re.sub(CFG['trailing_pad_pattern'], '', text)
    text = re.sub(CFG['trailing_sentence_pattern'], '', text)

    return text.strip()

def build_transformers_pipeline(model_name, tokenizer, use_8bit):
    import torch
    from transformers import pipeline

    model_kwargs = {"dtype": getattr(torch, CFG["transformers_dtype"])}
    if use_8bit:
        model_kwargs["load_in_8bit"] = True

    device_map = CFG["device_map"] if importlib.util.find_spec("accelerate") else None

    hf_pipeline = pipeline(
        "text-generation", model=model_name, model_kwargs=model_kwargs,
        device_map=device_map, tokenizer=tokenizer, trust_remote_code=CFG['transformers_trust_remote_code'],
        token=HF_TOKEN
    )
    configure_pipeline_for_batching(hf_pipeline, tokenizer)
    return hf_pipeline

def resolve_input_path(dataset, input_dir=None):
    """Retorna o input canônico de ``pt`` ou ``pten``.

    ``input_dir`` permite uma sobreposição local, mas os valores padrão são
    sempre os arquivos versionados no repositório.
    """
    if input_dir is not None:
        candidate = Path(input_dir) / IO["input_template"].format(dataset=dataset)
        if candidate.exists():
            return candidate

    input_path = DATASET_INPUTS[dataset]
    if input_path.exists():
        return input_path
    raise FileNotFoundError(f"Dataset '{dataset}' não encontrado: {input_path}")


def describe_datasets(datasets, input_dir=None):
    """Valida e descreve os inputs sem carregar um modelo."""
    for dataset in datasets:
        input_path = resolve_input_path(dataset, input_dir)
        with input_path.open(encoding="utf-8") as input_file:
            prompt_count = sum(1 for line in input_file if line.strip())
        print(f"{dataset}: {prompt_count} prompts | input: {input_path}")

# --- FUNÇÃO PRINCIPAL ---
def run_multilang_inference(
    model_name,
    gpu_util,
    max_len,
    output_dir=DEFAULT_OUTPUT_DIR,
    input_dir=None,
    use_8bit=CFG['load_in_8bit'],
    datasets=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if datasets is None:
        datasets = CFG["datasets_default"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[WORKER] Iniciando inferência para: {model_name}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Saída: {output_dir}")

    os.environ[CFG["allocator_env"]] = CFG["allocator_config"]
    if gpu_util > CFG['gpu_util_cap']: gpu_util = CFG['gpu_util_cap']
    gc.collect(); torch.cuda.empty_cache()

    # --- DETECÇÃO ---
    model_lower = model_name.lower()
    is_qwen_35 = any(marker in model_lower for marker in CFG["qwen35_markers"])
    is_qwen_3  = any(marker in model_lower for marker in CFG["qwen3_markers"]) and not is_qwen_35
    is_gemma = "gemma" in model_lower
    is_llama_32 = "llama-3.2" in model_lower
    prefer_transformers = CFG['prefer_transformers']

    llm_engine = None; hf_pipeline = None; tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=CFG['tokenizer_trust_remote_code'], token=HF_TOKEN)

        # --- LÓGICA DE "SEM LIMITES" (AUTO-DETECT) ---
        # Tenta pegar o limite do config do modelo. Se não achar, usa 32k.
        # Qwen 3 geralmente suporta 32768.
        model_max_context = getattr(tokenizer, "model_max_length", CFG['maximum_context_tokens'])

        # Limite mais realista (evita overflow mas não estrangula)
        if model_max_context > CFG['context_cap'] or model_max_context < 0:
            model_max_context = CFG['context_cap']

        max_generation_tokens = min(max_new_tokens, model_max_context)

        print(f"📏 Limite de Geração Definido para: {model_max_context} tokens")

        if prefer_transformers:
            print(f"🚀 [MODO TRANSFORMERS] Usando backend compatível...")
            hf_pipeline = build_transformers_pipeline(model_name, tokenizer, use_8bit)

        elif VLLM_AVAILABLE:
            print(f"⚡ [MODO vLLM] Carregando engine...")
            quant_method = None
            if "gptq" in model_lower: quant_method = "gptq"
            elif "awq" in model_lower: quant_method = "awq"

            # No vLLM, max_model_len define o contexto total (Prompt + Resposta)
            # Se o usuário não passou argumento, usamos o detectado
            vllm_len = min(max_len, model_max_context) if max_len > 0 else model_max_context

            try:
                llm_engine = LLM(
                    model=model_name, trust_remote_code=CFG['vllm_trust_remote_code'], gpu_memory_utilization=gpu_util,
                    max_model_len=vllm_len, enforce_eager=CFG['vllm_enforce_eager'], tensor_parallel_size=CFG['tensor_parallel_size'],
                    quantization=quant_method, dtype=CFG['vllm_dtype']
                )
            except Exception as vllm_error:
                print(f"⚠️ vLLM falhou ({vllm_error}). Fazendo fallback para Transformers...")
                hf_pipeline = build_transformers_pipeline(model_name, tokenizer, use_8bit)
                llm_engine = None

        else:
            print(f"🚀 [MODO TRANSFORMERS] vLLM não disponível, usando Transformers...")
            hf_pipeline = build_transformers_pipeline(model_name, tokenizer, use_8bit)

    except Exception as e:
        sys.exit(f"❌ Erro Fatal: {e}")

    failures = []
    for dataset in datasets:
        print(f"\n>>> Processando: {dataset.upper()}")
        try:
            input_path = resolve_input_path(dataset, input_dir)
        except FileNotFoundError as error:
            print(f"⚠️ {error}")
            failures.append((dataset, error))
            continue

        try:
            ds = load_dataset(
                "json", data_files={"train": os.fspath(input_path)}, split="train"
            )
            prompt_col = "prompt"
            if "prompt" not in ds.column_names:
                for c in CFG["prompt_column_candidates"]:
                    if c in ds.column_names: prompt_col = c; break

            raw_prompts = [item[prompt_col] for item in ds]
            final_prompts = []

            print(f"   ⚙️ Aplicando Chat Template...")
            for prompt in raw_prompts:
                msgs = [{"role": CFG['message_role'], "content": prompt}]
                formatted = prompt
                if tokenizer.chat_template:
                    try:
                        if is_qwen_3:
                            # Qwen3: thinking ativado
                            formatted = tokenizer.apply_chat_template(
                                msgs, tokenize=CFG['chat_template_tokenize'],
                                add_generation_prompt=CFG['chat_template_add_generation_prompt'],
                                enable_thinking=CFG['qwen3_enable_thinking']
                            )
                        elif is_qwen_35:
                            formatted = tokenizer.apply_chat_template(
                                msgs, tokenize=CFG['chat_template_tokenize'],
                                add_generation_prompt=CFG['chat_template_add_generation_prompt'],
                                enable_thinking=CFG['qwen35_enable_thinking']
                            )
                        else:
                            formatted = tokenizer.apply_chat_template(
                                msgs, tokenize=CFG['chat_template_tokenize'],
                                add_generation_prompt=CFG['chat_template_add_generation_prompt']
                            )
                    except TypeError:
                        try: formatted = tokenizer.apply_chat_template(msgs, tokenize=CFG['chat_template_tokenize'], add_generation_prompt=CFG['chat_template_add_generation_prompt'])
                        except: formatted = CFG["fallback_chat_template"].format(prompt=prompt)
                    except Exception: formatted = CFG["fallback_chat_template"].format(prompt=prompt)
                final_prompts.append(formatted)
            print(f"   ⚡ Gerando respostas...")
            generated_text = []

            # === GERAÇÃO (Transformers) ===
            if hf_pipeline:
                from tqdm import tqdm

                gen_cfg = build_generation_config(tokenizer, is_qwen_3, max_generation_tokens)
                gen_kwargs = {
                    "return_full_text": CFG["return_full_text"],
                    "generation_config": gen_cfg,
                }

                total_prompts = len(final_prompts)
                n_batches = (total_prompts + batch_size - 1) // batch_size
                print(f"   📝 {total_prompts} prompts | batch_size={batch_size} | {n_batches} batches")
                print(f"   ⏳ Gerando... (cada batch pode demorar alguns minutos)")

                raw_text = ""  # garante que existe para o fallback
                with tqdm(total=total_prompts, desc="   ⚡ Gerando respostas", unit="amostra") as pbar:
                    for outputs in hf_pipeline(final_prompts, batch_size=batch_size, **gen_kwargs):
                        raw_text = outputs[0]["generated_text"]
                        clean_txt = clean_response(raw_text, [])
                        # Fallback bruto se limpeza removeu conteúdo demais
                        if len(clean_txt.split()) < CFG["minimum_clean_words"]:
                            clean_txt = raw_text[:CFG["fallback_characters"]]
                        generated_text.append(clean_txt)
                        pbar.update(1)

            # === GERAÇÃO (vLLM) ===
            elif llm_engine:
                # Qwen3/Qwen3.5: sampling com temperatura (thinking mode)
                # Demais modelos: greedy (temperatura 0)
                if is_qwen_3:
                    sampling_params = SamplingParams(
                        temperature=CFG['vllm_qwen3_temperature'],
                        top_p=CFG['vllm_qwen3_top_p'],
                        max_tokens=max_generation_tokens,
                        stop=CFG['vllm_stop_strings']
                    )
                else:
                    sampling_params = SamplingParams(
                        temperature=CFG['vllm_default_temperature'],
                        max_tokens=max_generation_tokens,
                        stop=CFG['vllm_stop_strings']
                    )

                outputs = llm_engine.generate(final_prompts, sampling_params)
                outputs.sort(key=lambda x: int(x.request_id))
                total = len(outputs)
                for idx, output in enumerate(outputs, start=1):
                    generated_text.append(clean_response(output.outputs[0].text, []))
                    if idx == total or idx % CFG["progress_interval"] == 0:
                        pct = int((idx / total) * 100)
                        print(f"   ⚡ Gerando respostas: {pct}% ({idx}/{total})")

            # Salvar
            safe_model = model_name.replace('/', '__')
            output_filename = output_dir / IO["response_template"].format(dataset=dataset, model=safe_model)
            if "response" in ds.column_names: ds = ds.remove_columns("response")
            ds = ds.add_column("response", generated_text)
            ds.select_columns([prompt_col, "response"]).to_json(output_filename, force_ascii=False)
            print(f"   ✅ Salvo: {output_filename}")
            torch.cuda.empty_cache()

        except Exception as e:
            failures.append((dataset, e))
            print(f"❌ Erro {dataset}: {e}"); traceback.print_exc()

    if failures:
        failed_datasets = ", ".join(dataset for dataset, _ in failures)
        raise RuntimeError(
            f"A inferência falhou para: {failed_datasets}. "
            "Verifique os logs; nem todas as respostas foram produzidas."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera respostas para os datasets canônicos pt (535) e pten (541)."
    )
    parser.add_argument("--config", help="Stage YAML configuration file.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=CFG['gpu_memory_utilization'])
    parser.add_argument("--max_model_len", type=int, default=CFG['max_model_len'])
    parser.add_argument(
        "--output-dir", "--data_dir", dest="output_dir", type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta para respostas; não é usada como fonte dos inputs.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Sobrepõe os inputs com <pasta>/<dataset>_input_data.jsonl.",
    )
    parser.add_argument("--load_in_8bit", action=argparse.BooleanOptionalAction, default=CFG['load_in_8bit'])
    parser.add_argument(
        "--datasets", "--languages", dest="datasets", nargs="+", choices=tuple(DATASET_INPUTS),
        default=CFG['datasets_default'], help="pt e/ou pten.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Valida os datasets sem carregar o modelo.")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Modelo: {args.model_name}")
        print(f"Saída prevista: {args.output_dir}")
        describe_datasets(args.datasets, args.input_dir)
        sys.exit(0)
    run_multilang_inference(
        args.model_name, args.gpu_memory_utilization, args.max_model_len,
        args.output_dir, args.input_dir, args.load_in_8bit, args.datasets,
        args.max_new_tokens, args.batch_size
    )
