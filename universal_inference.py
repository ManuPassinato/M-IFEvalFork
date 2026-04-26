import os
import argparse
import torch
import gc
import sys
import traceback
import json
import re
import importlib.util
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, pipeline

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "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"

def configure_pipeline_for_batching(hf_pipeline, tokenizer, is_qwen_3, model_max_context):
    """Configura pad token e generation_config para evitar warnings/deprecações."""
    if tokenizer.pad_token_id is None:
        eos_id = tokenizer.eos_token_id
        if eos_id is None and hasattr(hf_pipeline.model.config, "eos_token_id"):
            eos_id = hf_pipeline.model.config.eos_token_id
        if eos_id is not None:
            tokenizer.pad_token_id = eos_id
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = getattr(hf_pipeline.model, "generation_config", None)
    if gen_cfg is not None:
        # Evita warning de conflito entre max_new_tokens e max_length.
        gen_cfg.max_length = None
        gen_cfg.max_new_tokens = model_max_context
        if is_qwen_3:
            gen_cfg.do_sample = True
            gen_cfg.temperature = 0.7
            gen_cfg.top_p = 0.8
        else:
            gen_cfg.do_sample = False
            gen_cfg.repetition_penalty = 1.2

# Tenta importar vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ Aviso: vLLM não encontrado. O script rodará apenas em modo Transformers.")

# --- A FUNÇÃO DE LIMPEZA DEFINITIVA (V8 - NUCLEAR) ---
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

    # Remove lixo repetido no final
    text = re.sub(r'(\s*<pad>\s*)+$', '', text)
    text = re.sub(r'(\s*</s>\s*)+$', '', text)
    
    return text.strip()

# --- FUNÇÃO PRINCIPAL ---
def run_multilang_inference(model_name, gpu_util, max_len, data_dir=None, use_8bit=False, languages=None):
    if languages is None:
        languages = ["pt"]
    print(f"\n[WORKER] Iniciando Ciclo Multi-Idioma para: {model_name}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpu_util > 0.95: gpu_util = 0.95
    gc.collect(); torch.cuda.empty_cache()

    if not data_dir:
        possible = [
            "M-IFEvalFork/data",
            "data",
            os.path.join(os.getcwd(), "data"),
            "M-IFEvalFork/data_new",
            "data_new",
            os.path.join(os.getcwd(), "data_new"),
        ]
        for p in possible:
            if os.path.exists(p): data_dir = p; break
    if not data_dir: sys.exit("❌ Pasta data não encontrada")

    # --- DETECÇÃO ---
    model_lower = model_name.lower()
    is_qwen_3 = "qwen3" in model_lower or "qwen-3" in model_lower
    is_gemma_3 = "gemma-3" in model_lower
    is_gemma = "gemma" in model_lower
    is_llama_32 = "llama-3.2" in model_lower
    prefer_transformers = is_qwen_3 or is_gemma or is_llama_32
    
    llm_engine = None; hf_pipeline = None; tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
        
        # --- LÓGICA DE "SEM LIMITES" (AUTO-DETECT) ---
        # Tenta pegar o limite do config do modelo. Se não achar, usa 32k.
        # Qwen 3 geralmente suporta 32768.
        model_max_context = getattr(tokenizer, "model_max_length", 32768)

        # Limite mais realista (evita overflow mas não estrangula)
        if model_max_context > 32768 or model_max_context < 0:
            model_max_context = 32768

        # NOVO: limite separado de geração
        max_generation_tokens = 8096
            
        print(f"📏 Limite de Geração Definido para: {model_max_context} tokens")

        if prefer_transformers:
            print(f"🚀 [MODO TRANSFORMERS] Usando backend compatível...")
            model_kwargs = {"dtype": torch.bfloat16}
            if use_8bit: model_kwargs["load_in_8bit"] = True

            device_map = "auto" if importlib.util.find_spec("accelerate") else None

            hf_pipeline = pipeline(
                "text-generation", model=model_name, model_kwargs=model_kwargs,
                device_map=device_map, tokenizer=tokenizer, trust_remote_code=True,
                token=HF_TOKEN
            )
            configure_pipeline_for_batching(hf_pipeline, tokenizer, is_qwen_3, model_max_context)
        
        elif VLLM_AVAILABLE:
            print(f"⚡ [MODO vLLM] Carregando engine...")
            quant_method = None
            if "gptq" in model_lower: quant_method = "gptq"
            elif "awq" in model_lower: quant_method = "awq"
            
            # No vLLM, max_model_len define o contexto total (Prompt + Resposta)
            # Se o usuário não passou argumento, usamos o detectado
            vllm_len = max_len if max_len > 4096 else model_max_context

            try:
                llm_engine = LLM(
                    model=model_name, trust_remote_code=True, gpu_memory_utilization=gpu_util,
                    max_model_len=vllm_len, enforce_eager=True, tensor_parallel_size=1,
                    quantization=quant_method, dtype="auto"
                )
            except Exception as vllm_error:
                print(f"⚠️ vLLM falhou ({vllm_error}). Fazendo fallback para Transformers...")
                model_kwargs = {"dtype": torch.bfloat16}
                if use_8bit:
                    model_kwargs["load_in_8bit"] = True

                device_map = "auto" if importlib.util.find_spec("accelerate") else None

                hf_pipeline = pipeline(
                    "text-generation", model=model_name, model_kwargs=model_kwargs,
                    device_map=device_map, tokenizer=tokenizer, trust_remote_code=True,
                    token=HF_TOKEN
                )
                configure_pipeline_for_batching(hf_pipeline, tokenizer, is_qwen_3, model_max_context)
                llm_engine = None

    except Exception as e:
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
                            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                        else:
                            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    except TypeError:
                        try: formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        except: formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    except Exception: formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                final_prompts.append(formatted)

            print(f"   ⚡ Gerando respostas...")
            generated_text = []

            # === GERAÇÃO (Transformers) ===
            if hf_pipeline:
                from tqdm import tqdm
                
                stop_strs = ["<|endoftext|>", "<|im_end|>", "<|end|>", "</s>"]
                
                gen_kwargs = {
                    "return_full_text": False,
                    "max_new_tokens": max_generation_tokens,
                    "do_sample": False,
                    "temperature": 0.0,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                }

                for outputs in tqdm(hf_pipeline(final_prompts, batch_size=8, **gen_kwargs), total=len(final_prompts)):
                    raw_text = outputs[0]["generated_text"]
                    clean_txt = clean_response(raw_text, stop_strs)
                    generated_text.append(clean_txt)

                if len(clean_txt.split()) < 3:
                    clean_txt = raw_text[:2000]  # fallback bruto

            # === GERAÇÃO (vLLM) ===
            elif llm_engine:
                # vLLM: Definimos max_tokens alto
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=8096,
                    stop=["</s>", "<|end|>", "<|im_end|>"]
                )
                outputs = llm_engine.generate(final_prompts, sampling_params)
                outputs.sort(key=lambda x: int(x.request_id))
                generated_text = [clean_response(o.outputs[0].text, []) for o in outputs]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8096)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--languages", type=str, nargs="+", default=["en"])  # <-- NOVO
    args = parser.parse_args()
    run_multilang_inference(
        args.model_name, args.gpu_memory_utilization, args.max_model_len,
        args.data_dir, args.load_in_8bit, args.languages              # <-- NOVO
    )