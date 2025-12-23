import os
os.environ['HF_HUB_CACHE'] = '/workspace/M-IFEvalFork/.cache/huggingface/hub'

import sys
import shutil
import subprocess
import time
import json
import argparse
from datetime import datetime, timedelta
from huggingface_hub import scan_cache_dir


# --- CONFIGURAÇÃO DA ESCALA ---
MODELS_TO_BENCHMARK = [
    # 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
    # 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4'
    # 'deepseek-ai/deepseek-llm-7b-chat',
    #'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',
    'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'
]

def install_dependencies():
    """Instala as dependências necessárias se ainda não estiverem instaladas."""
    print("📦 Verificando e instalando dependências...")
    
    # Dependências Python
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", 
        "vllm==0.7.1", "bitsandbytes==0.45.1", "hf-transfer==0.1.9", 
        "langdetect", "janome", "ja_sentence_segmenter", "spacy", "nltk"
    ])
    
    # Dependências do repositório
    if os.path.exists("requirements.txt"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    # Setup NLTK
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Setup Spacy
    spacy_models = [
        "en_core_web_sm", "es_core_news_sm", "fr_core_news_sm", 
        "ja_core_news_sm", "pt_core_news_sm", "xx_sent_ud_sm"
    ]
    
    print("Instalando modelos do Spacy...")
    for model in spacy_models:
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        except subprocess.CalledProcessError:
            print(f"⚠️ Aviso: Falha ao baixar modelo spacy {model}")

    os.makedirs("instruction_utils", exist_ok=True)
    with open("instruction_utils/__init__.py", "a") as f:
        pass 

    print("\n✅ Instalação de dependências concluída.")

def prepare_data():
    """Renomeia e limpa os dados de entrada."""
    print("\n🛠️ Preparando dados...")
    
    # 1. Renomear JSON para JSONL se necessário
    old_path = os.path.join("/workspace/M-IFEvalFork/data", "pt_input_data.json")
    new_path = os.path.join("/workspace/M-IFEvalFork/data", "pt_input_data.jsonl")

    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print("✅ Arquivo renomeado para .jsonl")

    # 2. Limpeza Cirúrgica
    input_path = "/workspace/M-IFEvalFork/data/pt_input_data.jsonl"
    output_path = "/workspace/M-IFEvalFork/data/pt_input_data_FINAL_CLEAN.jsonl"

    # Verifique se o arquivo de saída já existe
    if os.path.exists(output_path):
        print("📁 O arquivo de saída FINAL_CLEAN já existe. Pulando limpeza...")
        return

    # Se o arquivo FINAL_CLEAN não existir, inicie a limpeza
    print("🧹 INICIANDO LIMPEZA CIRÚRGICA...")
    
    KILL_LIST = ["pt:detectable_format:constrained_response", "pt:combination:repeat_prompt"]
    
    total = 0
    kept = 0
    removed = 0

    if not os.path.exists(input_path):
        print("❌ Arquivo original pt_input_data.jsonl não encontrado!")
        return

    # Abrir o arquivo de entrada e de saída
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            try:
                data = json.loads(line)
                ids = data.get("instruction_id_list", [])
                is_bad_line = any(bad_id in ids for bad_id in KILL_LIST)

                if is_bad_line:
                    removed += 1
                else:
                    fout.write(line)
                    kept += 1
            except json.JSONDecodeError:
                pass

    print(f"📊 RESULTADO LIMPEZA: Total: {total} | Mantidos: {kept} | Removidos: {removed}")
    if kept > 0:
        print(f"✅ Arquivo limpo gerado: {output_path}")

def delete_model_cache(model_id):
    """Limpa o cache do HuggingFace para liberar espaço."""
    print(f"🧹 Tentando limpar cache para: {model_id}...")
    try:
        hf_cache_info = scan_cache_dir()
        found = False
        for repo in hf_cache_info.repos:
            if repo.repo_id == model_id:
                shutil.rmtree(repo.repo_path)
                found = True
        if found: print("   -> Cache removido.")
        else: print("   -> Nada no cache para remover.")
    except Exception as e:
        print(f"   -> Erro não fatal na limpeza: {e}")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))
def force_gpu_cleanup():
    print("🧹 [SISTEMA] Forçando limpeza de processos Zumbis na GPU...")
    try:
        subprocess.run(["pkill", "-f", "universal_inference.py"], check=False)
        time.sleep(5) # Dá um tempo para a memória liberar
    except Exception as e:
        print(f"⚠️ Aviso na limpeza: {e}")

def run_benchmark():
    benchmark_start_time = time.time()

    for model in MODELS_TO_BENCHMARK:
        force_gpu_cleanup()
        model_start_time = time.time()
        safe_model_name = model.replace('/', '__')

        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO: {model}")
        print(f"{'='*60}")

        # --- PASSO 1: INFERÊNCIA ---
        print(">> Passo 1: Inferência (Processo Isolado)")
        t0_inf = time.time()
        inferencia_sucesso = False

        try:
            cmd = [sys.executable, "/workspace/M-IFEvalFork/universal_inference.py", "--model_name", model]
            process = subprocess.run(cmd, check=True, text=True)
            
            inference_time = time.time() - t0_inf
            print(f"   ⏱️ Tempo de Inferência: {format_time(inference_time)}")
            inferencia_sucesso = True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ FALHA NO MODELO {model} (Código {e.returncode})")
            print("   Ação: Pulando avaliação e limpando recursos.")
            delete_model_cache(model)
            continue

        # --- PASSO 2: AVALIAÇÃO ---
        if inferencia_sucesso:
            print("\n>> Passo 2: Avaliação")
            t0_eval = time.time()
            lang = "pt"
            
            # Caminhos absolutos garantem que o script não se perca
            base_path = "/workspace/M-IFEvalFork"
            input_data = os.path.join(base_path, f"data/{lang}_input_data_FINAL_CLEAN.jsonl")
            resp_file = os.path.join(base_path, f"data/{lang}_input_response_data_{safe_model_name}.jsonl")
            out_dir = os.path.join(base_path, f"evaluations/{lang}_input_response_data_{safe_model_name}")

            if os.path.exists(resp_file):
                os.makedirs(out_dir, exist_ok=True)
                try:
                    cmd_eval = [
                        sys.executable, 
                        "evaluation_main.py", # Removido o "-m" pois evaluation_main costuma ser script raiz
                        "--input_data", input_data,
                        "--input_response_data", resp_file,
                        "--output_dir", out_dir
                    ]
                    
                    print(f"   [DEBUG] Executando: {' '.join(cmd_eval)}") # Para você conferir o comando
                    
                    subprocess.run(
                        cmd_eval, 
                        check=True, 
                        cwd=base_path # <--- CRUCIAL: define onde o comando roda
                    )
                    print(f"   ✅ {lang.upper()}: Métricas calculadas com sucesso.")
                    
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ {lang.upper()}: Falhou na etapa de cálculo de métricas (Código {e.returncode}).")
            else:
                print(f"   ⚠️ {lang.upper()}: Arquivo de resposta não encontrado: {resp_file}")

            print(f"   ⏱️ Tempo Avaliação: {format_time(time.time() - t0_eval)}")

        # --- PASSO 3: LIMPEZA ---
        print(f"\n>> Passo 3: Limpeza Pós-Ciclo")
        delete_model_cache(model)
        
        print(f"✅ Ciclo finalizado para {model}")
        print(f"⏱️ Tempo total deste modelo: {format_time(time.time() - model_start_time)}")

    total_benchmark_time = time.time() - benchmark_start_time
    print(f"\n{'='*60}")
    print(f"🎉 BENCHMARK COMPLETO! Tempo total: {format_time(total_benchmark_time)}")

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
    install_dependencies()
    prepare_data()
    run_benchmark()
