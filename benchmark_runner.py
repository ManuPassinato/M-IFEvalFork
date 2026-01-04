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
os.environ["HF_TOKEN"] = ""
os.environ["HF_HUB_CACHE"] = os.path.join(os.getcwd(), "hf_cache_local")

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
    #"Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    #"google/gemma-3-1b-it",
]

# Idiomas que queremos testar
TARGET_LANGUAGES = ["pt", "en", "ja", "es", "fr"]

def install_dependencies():
    """Instala as dependências necessárias."""
    print("📦 Verificando dependências...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", 
        "vllm==0.7.1", "bitsandbytes==0.45.1", "hf-transfer==0.1.9", 
        "langdetect", "janome", "ja_sentence_segmenter", "spacy", "nltk"
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

# def prepare_data():
#     """Limpa os dados de entrada para TODOS os idiomas."""
#     print("\n🛠️ Preparando dados Multi-Idioma...")
    
#     base_dir = "/workspace/M-IFEvalFork/data"
#     KILL_LIST = ["detectable_format:constrained_response", "combination:repeat_prompt"]
    
#     # Mapeia codigo 'ja' para arquivo que pode estar como 'jp' ou 'ja'
#     # Vamos padronizar a limpeza
    
#     for lang in TARGET_LANGUAGES:
#         # Tenta achar o arquivo original
#         filename = f"{lang}_input_data.jsonl"
#         # Correção para caso o arquivo japonês esteja como 'jp'
#         if lang == "ja" and not os.path.exists(os.path.join(base_dir, filename)):
#             if os.path.exists(os.path.join(base_dir, "jp_input_data.jsonl")):
#                 filename = "jp_input_data.jsonl" # Usa o que tem
#             elif os.path.exists(os.path.join(base_dir, "jp_input_data.json")):
#                  # Converte json -> jsonl se precisar
#                  old = os.path.join(base_dir, "jp_input_data.json")
#                  new = os.path.join(base_dir, "jp_input_data.jsonl")
#                  os.rename(old, new)
#                  filename = "jp_input_data.jsonl"

#         input_path = os.path.join(base_dir, filename)
#         output_path = os.path.join(base_dir, f"{lang}_input_data_FINAL_CLEAN.jsonl")

#         if not os.path.exists(input_path):
#             print(f"⚠️ Pular {lang}: {filename} não encontrado.")
#             continue

#         if os.path.exists(output_path):
#             print(f"  -> {lang.upper()} já limpo.")
#             continue

#         print(f"🧹 Limpando {lang.upper()}...")
        
#         total = 0
#         kept = 0
        
#         with open(input_path, "r", encoding="utf-8") as fin, \
#              open(output_path, "w", encoding="utf-8") as fout:
#             for line in fin:
#                 total += 1
#                 try:
#                     data = json.loads(line)
#                     ids = data.get("instruction_id_list", [])
#                     # Verifica se tem algum ID proibido (adaptado para verificar substring)
#                     is_bad = False
#                     for bad_id in KILL_LIST:
#                         # O ID no json pode ser "pt:combination..." ou só "combination..."
#                         if any(bad_id in curr_id for curr_id in ids):
#                             is_bad = True
#                             break
                    
#                     if not is_bad:
#                         fout.write(line)
#                         kept += 1
#                 except: pass
#         print(f"     Mantidos: {kept}/{total}")

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

    for model in MODELS_TO_BENCHMARK:
        force_gpu_cleanup()
        model_start = time.time()
        safe_model_name = model.replace('/', '__')

        print(f"\n{'='*50}\n🚀 INICIANDO: {model}\n{'='*50}")

        # --- 1. INFERÊNCIA ---
        print(">> Passo 1: Inferência Multi-Língua")
        t0_inf = time.time()
        
        try:
            # Caminho do script
            script_path = "/home/emanuel/workspace/M-IFEvalFork/universal_inference.py"
            
            # Monta o comando base
            cmd = [sys.executable, script_path, "--model_name", model]

            # --- LÓGICA INTELIGENTE AQUI ---
            # Se for modelo pesado (13B+), ativa 8-bit e aumenta uso de GPU
            if any(size in model.lower() for size in ["13b", "30b", "32b", "34b", "70b"]):
                print(f"   ⚖️ Modelo grande detectado ({model}). Ativando 8-bit...")
                cmd.append("--load_in_8bit")
                cmd.extend(["--gpu_memory_utilization", "0.9"])
            else:
                # Para modelos menores, usa o padrão
                cmd.extend(["--gpu_memory_utilization", "0.75"])
            
            # Executa
            subprocess.run(cmd, check=True)
            print(f"⏱️ Tempo Inferência Total: {format_time(time.time() - t0_inf)}")

        except subprocess.CalledProcessError:
            print(f"❌ Falha crítica na inferência do modelo {model}. Pulando.")
            continue

        # --- 2. AVALIAÇÃO (O resto do código permanece igual) ---
        print("\n>> Passo 2: Avaliação")
        
        for lang in TARGET_LANGUAGES:
            print(f"\n📊 Avaliando: {lang.upper()}")
            input_data = f"/home/emanuel/workspace/M-IFEvalFork/data/{lang}_input_data_FINAL_CLEAN.jsonl"
            if not os.path.exists(input_data):
                 input_data = f"/home/emanuel/workspace/M-IFEvalFork/data/{lang}_input_data.jsonl"

            resp_file = f"/home/emanuel/workspace/M-IFEvalFork/data/{lang}_input_response_data_{safe_model_name}.jsonl"
            output_dir = f"/home/emanuel/workspace/M-IFEvalFork/evaluations/{lang}_input_response_data_{safe_model_name}"

            if os.path.exists(resp_file):
                try:
                    # Verifica se evaluation_main existe no caminho certo
                    eval_script = "/home/emanuel/workspace/M-IFEvalFork/evaluation_main.py"
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
                print(f"   ⚠️ Arquivo de resposta não encontrado: {resp_file}")

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
