import os
import argparse
import torch
import gc
import sys
import traceback
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def run_model_inference(model_name, gpu_util, max_len):
    print(f"\n[WORKER] Iniciando: {model_name}")
    print(f"[WORKER] Config: GPU Util={gpu_util}, Max Len={max_len}")

    # 1. Limpeza de Memória
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Identificar arquivo de dados (Prioridade para o CLEAN)
    # Ajuste o caminho base se necessário. Assumindo execução da raiz do projeto.
    # Se estiver rodando de dentro de M-IFEvalFork, use apenas "data"
    possible_paths = [
        "M-IFEvalFork/data", 
        "data", 
        "../data"
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = path
            break
    
    if not data_dir:
        print("❌ [ERRO] Pasta 'data' não encontrada.")
        sys.exit(1)

    input_file = None
    if os.path.exists(os.path.join(data_dir, "pt_input_data_FINAL_CLEAN.jsonl")):
        input_file = "pt_input_data_FINAL_CLEAN.jsonl"
    elif os.path.exists(os.path.join(data_dir, "pt_input_data_clean.jsonl")):
        input_file = "pt_input_data_clean.jsonl"
    elif os.path.exists(os.path.join(data_dir, "pt_input_data.jsonl")):
        input_file = "pt_input_data.jsonl"

    if not input_file:
        print(f"❌ Nenhum arquivo de input encontrado em {data_dir}")
        sys.exit(1)

    input_path = os.path.join(data_dir, input_file)
    print(f"[WORKER] Usando arquivo de entrada: {input_path}")

    # 3. Carregar Tokenizer (Para aplicar Chat Template)
    try:
        print("[WORKER] Carregando Tokenizer para Chat Template...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível carregar tokenizer separadamente ({e}). Usaremos prompt cru.")
        tokenizer = None

    # 4. Carregar Modelo vLLM
    try:
        print(f"[WORKER] Carregando vLLM...")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_util, # Dinâmico
            max_model_len=max_len,           # Dinâmico
            enforce_eager=True,
            tensor_parallel_size=1,
            device="cuda"
        )
    except Exception:
        print("❌ [ERRO FATAL] Falha ao carregar o modelo vLLM.")
        traceback.print_exc()
        sys.exit(1)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # 5. Processamento e Formatação
    try:
        ds = load_dataset("json", data_files={"train": input_path}, split="train")

        # Detecta coluna de prompt
        col_names = ds.column_names
        prompt_col = "prompt"
        if "prompt" not in col_names:
            for c in ["instruction", "pergunta", "input"]:
                if c in col_names:
                    prompt_col = c; break
        
        print(f"[WORKER] Coluna de prompt detectada: '{prompt_col}'")
        
        # --- APLICAÇÃO DO CHAT TEMPLATE (A Mágica) ---
        raw_prompts = [item[prompt_col] for item in ds]
        final_prompts = []

        if tokenizer and tokenizer.chat_template:
            print("[WORKER] Aplicando Chat Template (Llama 3 / Qwen / Mistral)...")
            for prompt in raw_prompts:
                # Cria a estrutura de conversa padrão
                messages = [{"role": "user", "content": prompt}]
                # Aplica o template sem tokenizar (retorna string formatada)
                formatted = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                final_prompts.append(formatted)
        else:
            print("⚠️ [AVISO] Modelo sem chat_template detectado ou tokenizer falhou. Usando prompts crus.")
            final_prompts = raw_prompts

        # 6. Geração
        print(f"[WORKER] Gerando respostas para {len(final_prompts)} prompts...")
        outputs = llm.generate(final_prompts, sampling_params)
        generated_text = [output.outputs[0].text for output in outputs]

        # 7. Salvar Saída
        safe_model = model_name.replace('/', '__')
        output_filename = os.path.join(data_dir, f"pt_input_response_data_{safe_model}.jsonl")

        # Remove coluna antiga se existir para evitar duplicação
        if "response" in ds.column_names:
            ds = ds.remove_columns("response")

        ds = ds.add_column("response", generated_text)
        # Salva mantendo o prompt original (sem formatação) para a avaliação bater
        ds.select_columns([prompt_col, "response"]).to_json(output_filename)
        print(f"✅ [SUCESSO] Arquivo salvo: {output_filename}")

    except Exception as e:
        print(f"❌ [ERRO] Falha durante geração: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    # Novos argumentos para controle fino de memória
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    
    args = parser.parse_args()
    
    run_model_inference(
        args.model_name, 
        args.gpu_memory_utilization, 
        args.max_model_len
    )
