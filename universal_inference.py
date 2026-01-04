import os
import argparse
import torch
import gc
import sys
import traceback
import json
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, pipeline

# Tenta importar vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ Aviso: vLLM não encontrado. O script rodará apenas em modo Transformers.")

LANGUAGES = ["pt", "en", "ja", "es", "fr"] 

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
    
    return text.strip()

# --- FUNÇÃO PRINCIPAL ---
def run_multilang_inference(model_name, gpu_util, max_len, data_dir=None, use_8bit=False):
    print(f"\n[WORKER] Iniciando Ciclo Multi-Idioma para: {model_name}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpu_util > 0.95: gpu_util = 0.95
    gc.collect(); torch.cuda.empty_cache()

    if not data_dir:
        possible = ["M-IFEvalFork/data", "data", os.path.join(os.getcwd(), "data")]
        for p in possible:
            if os.path.exists(p): data_dir = p; break
    if not data_dir: sys.exit("❌ Pasta data não encontrada")

    # --- DETECÇÃO ---
    model_lower = model_name.lower()
    is_qwen_3 = "qwen3" in model_lower or "qwen-3" in model_lower
    is_gemma_3 = "gemma-3" in model_lower
    use_transformers_fallback = is_qwen_3 or is_gemma_3
    
    llm_engine = None; hf_pipeline = None; tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # --- LÓGICA DE "SEM LIMITES" (AUTO-DETECT) ---
        # Tenta pegar o limite do config do modelo. Se não achar, usa 32k.
        # Qwen 3 geralmente suporta 32768.
        model_max_context = getattr(tokenizer, "model_max_length", 8096)
        
        # Proteção: Alguns tokenizers retornam números absurdos (int max).
        # Limitamos a 32768 para não estourar a VRAM da 4090.
        if model_max_context > 8096 or model_max_context < 0:
            model_max_context = 8096
            
        print(f"📏 Limite de Geração Definido para: {model_max_context} tokens")

        if use_transformers_fallback:
            print(f"🚀 [MODO QWEN 3/GEMMA] Usando Transformers...")
            model_kwargs = {"torch_dtype": torch.bfloat16}
            if use_8bit: model_kwargs["load_in_8bit"] = True

            hf_pipeline = pipeline(
                "text-generation", model=model_name, model_kwargs=model_kwargs,
                device_map="auto", tokenizer=tokenizer, trust_remote_code=True
            )
        
        elif VLLM_AVAILABLE:
            print(f"⚡ [MODO vLLM] Carregando engine...")
            quant_method = None
            if "gptq" in model_lower: quant_method = "gptq"
            elif "awq" in model_lower: quant_method = "awq"
            
            # No vLLM, max_model_len define o contexto total (Prompt + Resposta)
            # Se o usuário não passou argumento, usamos o detectado
            vllm_len = max_len if max_len > 4096 else model_max_context

            llm_engine = LLM(
                model=model_name, trust_remote_code=True, gpu_memory_utilization=gpu_util,
                max_model_len=vllm_len, enforce_eager=True, tensor_parallel_size=1,
                quantization=quant_method, dtype="auto"
            )

    except Exception as e:
        sys.exit(f"❌ Erro Fatal: {e}")

    for lang in LANGUAGES:
        print(f"\n>>> Processando: {lang.upper()}")
        input_file = f"{lang}_input_data_FINAL_CLEAN.jsonl"
        input_path = os.path.join(data_dir, input_file)
        if not os.path.exists(input_path): input_path = os.path.join(data_dir, f"{lang}_input_data.jsonl")
        if not os.path.exists(input_path): continue

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
            if use_transformers_fallback and hf_pipeline:
                from tqdm import tqdm
                
                terminators = [tokenizer.eos_token_id]
                stop_strs = ["<|endoftext|>", "<|im_end|>", "<|end|>", "</s>"]
                if hasattr(tokenizer, "convert_tokens_to_ids"):
                    for t in stop_strs:
                        tid = tokenizer.convert_tokens_to_ids(t)
                        if isinstance(tid, int): terminators.append(tid)
                
                gen_kwargs = {
                    "max_new_tokens": model_max_context, # <--- USA O MÁXIMO DETECTADO (ex: 32k)
                    "return_full_text": False,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": terminators
                }

                if is_qwen_3:
                    gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.8})
                else:
                    gen_kwargs.update({"do_sample": False, "repetition_penalty": 1.2})

                for outputs in tqdm(hf_pipeline(final_prompts, batch_size=8, **gen_kwargs), total=len(final_prompts)):
                    raw_text = outputs[0]["generated_text"]
                    clean_txt = clean_response(raw_text, stop_strs)
                    generated_text.append(clean_txt)

            # === GERAÇÃO (vLLM) ===
            elif llm_engine:
                # vLLM: Definimos max_tokens alto
                sampling_params = SamplingParams(temperature=0.0, max_tokens=model_max_context)
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
    # Define padrão alto para vLLM, mas o script tenta detectar auto
    parser.add_argument("--max_model_len", type=int, default=8096) 
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()
    run_multilang_inference(args.model_name, args.gpu_memory_utilization, args.max_model_len, args.data_dir, args.load_in_8bit)
