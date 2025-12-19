import os
import argparse
import torch
import gc
import sys
import traceback
from datasets import load_dataset
from vllm import LLM, SamplingParams

def run_model_inference(model_name):
    print(f"\n[WORKER] Iniciando: {model_name}")

    # 1. Limpeza
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Carregar Modelo
    try:
        print(f"[WORKER] Carregando vLLM...")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            tensor_parallel_size=1,
            device="cuda"
        )
    except Exception:
        print("❌ [ERRO FATAL] Falha ao carregar o modelo.")
        traceback.print_exc()
        sys.exit(1)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # 3. Identificar arquivo de dados
    data_dir = "./data"
    input_file = None

    # Prioridade para o arquivo limpo
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
    print(f"[WORKER] Usando arquivo de entrada: {input_file}")

    # 4. Processamento
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
        prompts = [item[prompt_col] for item in ds]

        # Geração
        outputs = llm.generate(prompts, sampling_params)
        generated_text = [output.outputs[0].text for output in outputs]

        # Salva Saída (SEM O _new)
        safe_model = model_name.replace('/', '__')
        output_filename = os.path.join(data_dir, f"pt_input_response_data_{safe_model}.jsonl")

        ds = ds.add_column("response", generated_text)
        ds.select_columns([prompt_col, "response"]).to_json(output_filename)
        print(f"✅ [SUCESSO] Arquivo salvo: {output_filename}")

    except Exception as e:
        print(f"❌ [ERRO] Falha durante geração: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    run_model_inference(args.model_name)
