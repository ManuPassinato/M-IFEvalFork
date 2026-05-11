import os
import argparse
import torch
import gc
import sys
import traceback
import re
import importlib.util
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, pipeline

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
DEFAULT_MAX_NEW_TOKENS = 8192   # ⚡ Era 32768 — reserva KV-cache gigante desnecessariamente.
                                #    8192 é mais que suficiente para IFEval/benchmarks de instrução.
                                #    Mude para 32768 só se seu benchmark exigir respostas muito longas.
DEFAULT_BATCH_SIZE = 32         # ⚡ Era 4. GPUs modernas aguentam muito mais em paralelo.

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
    gen_cfg = GenerationConfig()
    gen_cfg.max_new_tokens = max_generation_tokens
    gen_cfg.max_length = None
    gen_cfg.eos_token_id = tokenizer.eos_token_id
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    if is_qwen_3:
        gen_cfg.do_sample = True
        gen_cfg.temperature = 0.7
        gen_cfg.top_p = 0.8
    else:
        gen_cfg.do_sample = False
        gen_cfg.repetition_penalty = 1.2
    return gen_cfg

# Tenta importar vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ Aviso: vLLM não encontrado. O script rodará apenas em modo Transformers.")

def clean_response(text, stop_tokens_str):
    if not text: return ""

    # 1. REMOVE BLOCO DE PENSAMENTO FECHADO (<think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. SEGURANÇA PARA PENSAMENTO NÃO FINALIZADO
    if "<think>" in text:
        parts = text.split("<think>")
        if len(parts[0].strip()) > 0:
            text = parts[0]
        else:
            return ""

    # 3. REMOVE MARCADORES DE PENSAMENTO EM INGLÊS
    markers = [
        "Okay, I need to", "Okay, the user", "Here is the blog post",
        "Sure, here", "Let me start by", "I need to write", "Analysis:",
        "First, I will", "To write this essay", "First, I'll outline"
    ]
    for m in markers:
        if m in text:
            index = text.find(m)
            if index < 100:
                parts = text.split("\n\n")
                if len(parts) > 1:
                    text = "\n\n".join(parts[1:])
                else:
                    return ""
            else:
                text = text[:index]

    # 4. LIMPEZA FINAL
    text = text.lstrip("\n").lstrip("e\n").lstrip("va\n").strip()
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'</?s>', '', text)
    text = text.replace("<pad>", "")
    text = text.replace("<eos>", "")
    text = re.sub(r'(\s*<pad>\s*)+$', '', text)
    text = re.sub(r'(\s*</s>\s*)+$', '', text)

    return text.strip()

def build_transformers_pipeline(model_name, tokenizer, use_8bit):
    model_kwargs = {"dtype": torch.bfloat16}
    if use_8bit:
        model_kwargs["load_in_8bit"] = True

    device_map = "auto" if importlib.util.find_spec("accelerate") else None

    hf_pipeline = pipeline(
        "text-generation", model=model_name, model_kwargs=model_kwargs,
        device_map=device_map, tokenizer=tokenizer, trust_remote_code=True,
        token=HF_TOKEN
    )
    configure_pipeline_for_batching(hf_pipeline, tokenizer)
    return hf_pipeline

def resolve_data_dir(data_dir):
    if data_dir:
        return data_dir
    possible = [
        "M-IFEvalFork/data", "data", os.path.join(os.getcwd(), "data"),
        "M-IFEvalFork/data_new", "data_new", os.path.join(os.getcwd(), "data_new"),
    ]
    for path in possible:
        if os.path.exists(path):
            return path
    return None

# --- FUNÇÃO PRINCIPAL ---
def run_multilang_inference(
    model_name,
    gpu_util,
    max_len,
    data_dir=None,
    use_8bit=False,
    languages=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    if languages is None:
        languages = ["pt"]
    print(f"\n[WORKER] Iniciando Ciclo Multi-Idioma para: {model_name}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpu_util > 0.95: gpu_util = 0.95
    gc.collect(); torch.cuda.empty_cache()

    data_dir = resolve_data_dir(data_dir)
    if not data_dir:
        sys.exit("❌ Pasta data não encontrada")

    # --- DETECÇÃO ---
    model_lower = model_name.lower()
    is_qwen_3 = "qwen3" in model_lower or "qwen-3" in model_lower
    is_qwen_35 = "qwen3.5" in model_lower or "qwen3-5" in model_lower
    is_gemma = "gemma" in model_lower
    is_llama_32 = "llama-3.2" in model_lower

    # Modelos grandes numa GPU de 24GB nao tem VRAM suficiente para o modelo
    # + CUDA Graphs simultaneamente. enforce_eager=True desativa CUDA Graphs
    # e libera ~2-3GB, permitindo que o KV-cache seja alocado.
    # Threshold conservador: >7B em bfloat16 = >14GB so para pesos.
    _large_patterns = ["9b", "8b", "7b", "13b", "30b", "32b", "34b", "70b"]
    is_large_for_24gb = any(p in model_lower for p in _large_patterns)

    use_eager = is_large_for_24gb
    prefer_transformers = not VLLM_AVAILABLE

    llm_engine = None; hf_pipeline = None; tokenizer = None

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
        except Exception as tok_err:
            if "vocab_size" in str(tok_err):
                # Qwen3.5 e outros modelos novos: transformers ainda nao tem
                # suporte completo, falta vocab_size no config gerado.
                # Forcamos use_fast=False para usar o tokenizer Python puro
                # que nao depende do config para vocab_size.
                print(f"⚠️ Tokenizer rapido falhou ({tok_err}). Tentando use_fast=False...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, token=HF_TOKEN, use_fast=False
                )
            else:
                raise

        model_max_context = getattr(tokenizer, "model_max_length", 32768)
        if model_max_context > 131072 or model_max_context < 0:
            # Valores absurdos (ex: 1e30) que alguns tokenizers retornam
            model_max_context = 32768

        max_generation_tokens = min(max_new_tokens, model_max_context)
        print(f"📏 Contexto máximo do modelo: {model_max_context} | Geração: {max_generation_tokens} tokens")

        # Patch para Qwen3.5 e outros modelos novos onde transformers nao carregou
        # vocab_size no config (bug em transformers 5.x com arquiteturas recentes).
        # Registra um hook global que injeta vocab_size em qualquer config que nao tenha,
        # cobrindo tanto o pipeline HuggingFace quanto o vLLM.
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
            if not hasattr(_cfg, "vocab_size") or _cfg.vocab_size is None:
                _vocab_size = len(tokenizer)
                _cfg.vocab_size = _vocab_size
                # Salva no config class para que instancias futuras tambem tenham
                type(_cfg).vocab_size = _vocab_size
                print(f"🔧 Patch vocab_size aplicado: {_vocab_size}")
        except Exception as _ve:
            print(f"⚠️ Patch vocab_size falhou: {_ve}")

        if prefer_transformers:
            print(f"🚀 [MODO TRANSFORMERS] vLLM não disponível, usando HuggingFace pipeline...")
            hf_pipeline = build_transformers_pipeline(model_name, tokenizer, use_8bit)

        else:
            print(f"⚡ [MODO vLLM] Carregando engine...")
            quant_method = None
            if "gptq" in model_lower: quant_method = "gptq"
            elif "awq" in model_lower: quant_method = "awq"

            # ⚡ max_model_len = contexto total (prompt + resposta).
            #    Sempre usa o contexto real do modelo — nunca ultrapassa o limite
            #    definido em config.json, evitando o erro de validação do vLLM.
            #    O argumento --max_model_len só reduz, nunca aumenta além do suportado.
            vllm_len = min(max_len if max_len > 0 else model_max_context, model_max_context)

            # Token HF deve estar no env ANTES de qualquer chamada do vLLM
            if HF_TOKEN:
                os.environ["HF_TOKEN"] = HF_TOKEN
                os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

            vllm_kwargs = dict(
                model=model_name,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_util,
                max_model_len=vllm_len,
                # enforce_eager=True para modelos grandes (>7B) em GPUs de 24GB:
                # pesos >14GB + CUDA Graphs nao cabem juntos no KV-cache.
                # Para modelos pequenos, False e mais rapido (CUDA Graphs ativos).
                enforce_eager=is_large_for_24gb,
                tensor_parallel_size=1,
                quantization=quant_method,
                dtype="auto",
                enable_prefix_caching=True,
                # max_num_batched_tokens removido no vLLM 0.19.x (V1 engine)
                disable_log_stats=True,
            )

            try:
                llm_engine = LLM(**vllm_kwargs)
            except Exception as vllm_error:
                print(f"⚠️ vLLM falhou ({vllm_error}). Fazendo fallback para Transformers...")
                # Libera qualquer memoria residual do processo filho do vLLM
                # antes de carregar o pipeline Transformers
                try:
                    import vllm.distributed.parallel_state as _vps
                    _vps.destroy_model_parallel()
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                hf_pipeline = build_transformers_pipeline(model_name, tokenizer, use_8bit)
                llm_engine = None

    except Exception as e:
        if "vocab_size" in str(e):
            sys.exit(
                f"❌ Erro Fatal: {e}\n"
                f"💡 Dica: Qwen3.5 requer transformers mais recente. Rode:\n"
                f"   pip install --upgrade git+https://github.com/huggingface/transformers.git"
            )
        sys.exit(f"❌ Erro Fatal: {e}")

    for lang in languages:
        print(f"\n>>> Processando: {lang.upper()}")
        input_file = f"{lang}_input_data.jsonl"
        input_path = os.path.join(data_dir, input_file)

        if not os.path.exists(input_path):
            project_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_inputs = [
                os.path.join(project_dir, "data", input_file),
                os.path.join(os.getcwd(), "data", input_file),
            ]
            input_path = next((p for p in fallback_inputs if os.path.exists(p)), input_path)

        if not os.path.exists(input_path):
            print(f"⚠️ Sem input para {lang}: {input_file} não encontrado em {data_dir} nem em data/")
            continue

        try:
            ds = load_dataset("json", data_files={"train": input_path}, split="train")
            prompt_col = "prompt"
            if "prompt" not in ds.column_names:
                for c in ["instruction", "input"]:
                    if c in ds.column_names: prompt_col = c; break

            raw_prompts = [item[prompt_col] for item in ds]
            final_prompts = []

            print(f"   ⚙️ Aplicando Chat Template...")
            for prompt in raw_prompts:
                msgs = [{"role": "user", "content": prompt}]
                formatted = prompt
                if tokenizer.chat_template:
                    try:
                        if is_qwen_3:
                            formatted = tokenizer.apply_chat_template(
                                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                            )
                        else:
                            formatted = tokenizer.apply_chat_template(
                                msgs, tokenize=False, add_generation_prompt=True
                            )
                    except TypeError:
                        try: formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        except: formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    except Exception:
                        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                final_prompts.append(formatted)

            print(f"   ⚡ Gerando respostas ({len(final_prompts)} amostras)...")
            generated_text = []

            # === GERAÇÃO (Transformers) ===
            if hf_pipeline:
                from tqdm import tqdm

                stop_strs = ["<|endoftext|>", "<|im_end|>", "<|end|>", "</s>"]
                gen_cfg = build_generation_config(tokenizer, is_qwen_3, max_generation_tokens)
                gen_kwargs = {
                    "return_full_text": False,
                    "generation_config": gen_cfg,
                }

                # ⚡ batch_size passado dinamicamente (era fixo em 8 no runner)
                progress = tqdm(
                    hf_pipeline(final_prompts, batch_size=batch_size, **gen_kwargs),
                    total=len(final_prompts),
                    desc="   ⚡ Gerando respostas",
                    unit="amostra",
                )
                for outputs in progress:
                    raw_text = outputs[0]["generated_text"]
                    clean_txt = clean_response(raw_text, stop_strs)
                    # ⚡ BUG CORRIGIDO: fallback estava fora do loop, referenciando
                    #    clean_txt da iteração anterior. Agora está dentro.
                    if len(clean_txt.split()) < 3:
                        clean_txt = raw_text[:2000]
                    generated_text.append(clean_txt)

            # === GERAÇÃO (vLLM) ===
            elif llm_engine:
                # ⚡ temperature=0 + uso de beam implícito = geração mais rápida e determinística
                sampling_params = SamplingParams(
                    temperature=0.7 if is_qwen_3 else 0.0,
                    top_p=0.8 if is_qwen_3 else 1.0,
                    max_tokens=max_generation_tokens,
                    stop=["</s>", "<|end|>", "<|im_end|>"],
                    # ⚡ Parâmetro novo: ignora EOS até min_tokens para evitar
                    #    respostas degeneradas de 1-2 tokens
                    min_tokens=10,
                )
                # ⚡ vLLM faz batching interno automaticamente (continuous batching).
                #    Não precisa de loop manual por chunk — gerar tudo de uma vez
                #    é sempre mais eficiente.
                outputs = llm_engine.generate(final_prompts, sampling_params)
                outputs.sort(key=lambda x: int(x.request_id))
                total = len(outputs)
                for idx, output in enumerate(outputs, start=1):
                    generated_text.append(clean_response(output.outputs[0].text, []))
                    if idx == total or idx % 50 == 0:
                        pct = int((idx / total) * 100)
                        print(f"   ⚡ Processando saídas: {pct}% ({idx}/{total})")

            # Salvar
            safe_model = model_name.replace('/', '__')
            output_filename = os.path.join(data_dir, f"{lang}_input_response_data_{safe_model}.jsonl")
            if "response" in ds.column_names: ds = ds.remove_columns("response")
            ds = ds.add_column("response", generated_text)
            ds.select_columns([prompt_col, "response"]).to_json(output_filename, force_ascii=False)
            print(f"   ✅ Salvo: {output_filename}")
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Erro {lang}: {e}"); traceback.print_exc()

    # Destroi engine ao final para garantir que VRAM seja liberada
    # antes que o processo pai (benchmark_runner) inicie o proximo modelo
    if llm_engine is not None:
        _destroy_vllm_engine(llm_engine)
    if hf_pipeline is not None:
        del hf_pipeline
        gc.collect()
        torch.cuda.empty_cache()

def _destroy_vllm_engine(llm_engine):
    """Destroi o engine vLLM e libera VRAM explicitamente."""
    if llm_engine is None:
        return
    try:
        import vllm.distributed.parallel_state as _vps
        _vps.destroy_model_parallel()
    except Exception:
        pass
    try:
        del llm_engine
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--languages", type=str, nargs="+", default=["pt"])
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    run_multilang_inference(
        args.model_name, args.gpu_memory_utilization, args.max_model_len,
        args.data_dir, args.load_in_8bit, args.languages, args.max_new_tokens, args.batch_size
    )
