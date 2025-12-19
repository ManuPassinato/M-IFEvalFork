import os
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
    'gpt-4o-mini-2024-07-18',
    'gpt-4o-2024-08-06',
    'o1-preview-2024-09-12',
    'o1-mini-2024-09-12',
    'claude-3-haiku-20240307',
    'claude-3-5-sonnet-20240620',
    'claude-3-opus-20240229',
    'gemini-1.5-pro-002',
    'gemini-1.5-flash-002',
    'CohereForAI/c4ai-command-r-plus-4bit',
    'CohereForAI/c4ai-command-r-v01-4bit',
    'CohereForAI/aya-23-8B',
    'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4'',
    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
    'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'deepseek-ai/deepseek-llm-7b-chat''
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
    old_path = os.path.join("data", "pt_input_data.json")
    new_path = os.path.join("data", "pt_input_data.jsonl")

    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print("✅ Arquivo renomeado para .jsonl")

    # 2. Limpeza Cirúrgica
    print("🧹 INICIANDO LIMPEZA CIRÚRGICA...")
    KILL_LIST = ["pt:detectable_format:constrained_response"]
    
    input_path = "data/pt_input_data.jsonl"
    output_path = "data/pt_input_data_FINAL_CLEAN.jsonl"

    total = 0
    kept = 0
    removed = 0

    if not os.path.exists(input_path):
        print("❌ Arquivo original pt_input_data.jsonl não encontrado!")
        return

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

def run_benchmark():
    benchmark_start_time = time.time()

    for model in MODELS_TO_BENCHMARK:
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
            cmd = [sys.executable, "universal_inference.py", "--model_name", model]
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
            
            resp_file = f"data/{lang}_input_response_data_{safe_model_name}.jsonl"
            out_dir = f"evaluations/{lang}_input_response_data_{safe_model_name}"

            if os.path.exists(resp_file):
                os.makedirs(out_dir, exist_ok=True)
                try:
                    subprocess.run([
                        sys.executable, "-m", "evaluation_main",
                        "--input_data", f"data/{lang}_input_data_FINAL_CLEAN.jsonl",
                        "--input_response_data", resp_file,
                        "--output_dir", out_dir
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    print(f"   ✅ {lang.upper()}: Métricas calculadas com sucesso.")
                except subprocess.CalledProcessError:
                    print(f"   ❌ {lang.upper()}: Falhou na etapa de cálculo de métricas.")
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

def zip_results():
    """Empacota os resultados (JSONs e Evaluations) localmente."""
    if not MODELS_TO_BENCHMARK: return

    model_name = MODELS_TO_BENCHMARK[0]
    safe_model_name = model_name.replace('/', '__')
    timestamp = datetime.now().strftime("%H%M")
    
    # 1. Zipar JSONs gerados
    nome_zip_json = f"novos_jsons_{safe_model_name}_{timestamp}"
    pasta_temp_json = "download_temp_jsons"
    os.makedirs(pasta_temp_json, exist_ok=True)
    
    # ATENÇÃO: Nome sem _new
    file_path = f"data/pt_input_response_data_{safe_model_name}.jsonl"
    
    if os.path.exists(file_path):
        shutil.copy(file_path, pasta_temp_json)
        shutil.make_archive(nome_zip_json, 'zip', pasta_temp_json)
        print(f"\n📦 Arquivo ZIP de respostas gerado: {nome_zip_json}.zip")
    shutil.rmtree(pasta_temp_json)

    # 2. Zipar Resultados da Avaliação
    nome_zip_eval = f"resultados_evaluation_{timestamp}"
    pasta_temp_eval = "download_temp_evals"
    if os.path.exists(pasta_temp_eval): shutil.rmtree(pasta_temp_eval)
    os.makedirs(pasta_temp_eval, exist_ok=True)
    
    # ATENÇÃO: Nome sem _new
    src_dir = f"evaluations/pt_input_response_data_{safe_model_name}"
    dst_dir = os.path.join(pasta_temp_eval, "pt_results")
    
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, dst_dir)
        shutil.make_archive(nome_zip_eval, 'zip', pasta_temp_eval)
        print(f"📦 Arquivo ZIP de avaliações gerado: {nome_zip_eval}.zip")
    shutil.rmtree(pasta_temp_eval)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
    install_dependencies()
    prepare_data()
    run_benchmark()
    zip_results()
