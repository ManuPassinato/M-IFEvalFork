import os

# --- CORREÇÃO DE PASTA TEMPORÁRIA ---
# Cria uma pasta 'tmp_local' no diretório atual e força o Python a usá-la
# Isso resolve o erro "No usable temporary directory found"
local_tmp = os.path.join(os.getcwd(), "tmp_local")
if not os.path.exists(local_tmp):
    try:
        os.makedirs(local_tmp)
    except OSError:
        pass # Ignora se já existir ou der erro (o print abaixo avisa)

os.environ["TMPDIR"] = local_tmp
os.environ["TEMP"] = local_tmp
os.environ["TMP"] = local_tmp
print(f"🔧 Pasta temporária redirecionada para: {local_tmp}")

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache_local")
os.environ["HF_HUB_CACHE"] = os.path.join(os.getcwd(), "hf_cache_local")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "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")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

import torch
import sys
import shutil
import subprocess
import time
import json
import argparse
from datetime import datetime, timedelta
from huggingface_hub import scan_cache_dir

# --- CONFIGURAÇÃO ---
MODELS_TO_BENCHMARK = [
    # "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.2-1B-evals",
    # "meta-llama/Llama-3.2-3B-evals",
    # "meta-llama/Llama-3.2-1B-Instruct-evals",
    # "meta-llama/Llama-3.2-3B-Instruct-evals",
    # "Qwen/Qwen3.5-0.8B",
    # "Qwen/Qwen3.5-2B",
    #"Qwen/Qwen3.5-4B",
    #"Qwen/Qwen3.5-9B"
    # "google/gemma-4-E2B-it",
    # "google/gemma-4-E4B-it",
    #"CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    # "Polygl0t/Tucano2-0.6B-Base",
    # "Polygl0t/Tucano2-qwen-0.5B-Base",
    # "Polygl0t/Tucano2-qwen-1.5B-Base",
    # "Polygl0t/Tucano2-qwen-3.7B-Base",
]

# Idiomas que queremos testar
TARGET_LANGUAGES = ["pt"]

def install_dependencies():
    """Instala as dependências necessárias."""
    print("📦 Verificando dependências...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", 
        "vllm==0.7.1", "bitsandbytes==0.45.1", "hf-transfer==0.1.9", 
        "langdetect", "janome", "ja_sentence_segmenter", "spacy", "nltk", "accelerate"
    ])
    
    import nltk
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')

    # Modelos Spacy necessários para avaliação
    spacy_models = ["en_core_web_sm", "es_core_news_sm", "fr_core_news_sm", "ja_core_news_sm", "pt_core_news_sm", "xx_sent_ud_sm"]
    print("Verificando Spacy...")
    for model in spacy_models:
        try:
            # Check rápido se carrega
            import spacy
            if not spacy.util.is_package(model):
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        except Exception:
            pass

def delete_model_cache(model_id):
    try:
        shutil.rmtree(f"/root/.cache/huggingface/hub/models--{model_id.replace('/', '--')}", ignore_errors=True)
    except: pass

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def force_gpu_cleanup():
    subprocess.run(["pkill", "-f", "universal_inference.py"], check=False)
    torch.cuda.empty_cache()
    time.sleep(2)

    print(f"\n🎉 BENCHMARK GERAL")

def run_benchmark():
    benchmark_start = time.time()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_inference = os.path.join(project_dir, "data_new")
    os.makedirs(data_dir_inference, exist_ok=True)

    for model in MODELS_TO_BENCHMARK:
        force_gpu_cleanup()
        model_start = time.time()
        safe_model_name = model.replace('/', '__')

        print(f"\n{'='*50}\n🚀 INICIANDO: {model}\n{'='*50}")

        # --- 1. INFERÊNCIA ---
        print(">> Passo 1: Inferência Multi-Língua")
        t0_inf = time.time()
        
        try:
                    current_dir = project_dir
                    script_path = os.path.join(current_dir, "universal_inference.py")

                    # Verifica se TODOS os idiomas já têm resposta gerada
                    all_exist = all(
                        os.path.exists(
                            os.path.join(data_dir_inference, f"{lang}_input_response_data_{safe_model_name}.jsonl")
                        )
                        for lang in TARGET_LANGUAGES
                    )

                    if all_exist:
                        print(f"   ♻️ Respostas já existem para todos os idiomas, pulando inferência.")
                    else:
                        # Monta o comando base
                        cmd = [sys.executable, script_path, 
                               "--model_name", model, 
                               "--data_dir", data_dir_inference, 
                               "--languages", *TARGET_LANGUAGES,]

                        if any(size in model.lower() for size in ["13b", "30b", "32b", "34b", "70b"]):
                            print(f"   ⚖️ Modelo grande detectado ({model}). Ativando 8-bit...")
                            cmd.append("--load_in_8bit")
                            cmd.extend(["--gpu_memory_utilization", "0.9"])
                        else:
                            cmd.extend(["--gpu_memory_utilization", "0.75"])

                        subprocess.run(cmd, check=True)
                        print(f"⏱️ Tempo Inferência Total: {format_time(time.time() - t0_inf)}")

        except subprocess.CalledProcessError:
            print(f"❌ Falha crítica na inferência do modelo {model}. Tentando avaliar respostas já existentes.")

        # --- 2. AVALIAÇÃO (O resto do código permanece igual) ---
        print("\n>> Passo 2: Avaliação")
        
        for lang in TARGET_LANGUAGES:
            print(f"\n📊 Avaliando: {lang.upper()}")
            input_data = os.path.join(project_dir, "data", f"{lang}_input_data.jsonl")
  
            resp_candidates = [
                os.path.join(project_dir, "data_new", f"{lang}_input_response_data_{safe_model_name}.jsonl"),
                os.path.join(project_dir, "data", f"{lang}_input_response_data_{safe_model_name}.jsonl"),
            ]
            resp_file = next((p for p in resp_candidates if os.path.exists(p)), None)
            output_dir = os.path.join(project_dir, "eval_new", f"{lang}_input_response_data_{safe_model_name}")

            if resp_file:
                try:
                    # Verifica se evaluation_main existe no caminho certo
                    eval_script = os.path.join(project_dir, "evaluation_main.py")
                    if not os.path.exists(eval_script):
                         # Tenta no diretório atual se não achar no absoluto
                         eval_script = "evaluation_main.py"

                    cmd_eval = [
                        sys.executable, eval_script,
                        "--input_data", input_data,
                        "--input_response_data", resp_file,
                        "--output_dir", output_dir
                    ]
                    
                    subprocess.run(cmd_eval, check=True)
                    print(f"   ✅ Sucesso: {lang}")
                except Exception as e:
                    print(f"   ❌ Falha na avaliação {lang}: {e}")
            else:
                print(f"   ⚠️ Arquivo de resposta não encontrado para {safe_model_name} em data_new/data")

        # --- 3. LIMPEZA ---
        print(f"\n>> Passo 3: Limpeza")
        # delete_model_cache(model) # Cuidado com isso se não quiser baixar de novo
        force_gpu_cleanup() 
        print(f"⏱️ Tempo total modelo: {format_time(time.time() - model_start)}")

    print(f"\n🎉 BENCHMARK GERAL FINALIZADO: {format_time(time.time() - benchmark_start)}")
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    install_dependencies()
    #prepare_data()
    run_benchmark()
