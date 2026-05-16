import os
import torch
import gc
import sys
import subprocess
import time
from datetime import timedelta

local_tmp = os.path.join(os.getcwd(), "tmp_local")
if not os.path.exists(local_tmp):
    try:
        os.makedirs(local_tmp)
    except OSError:
        pass

os.environ["TMPDIR"] = local_tmp
os.environ["TEMP"] = local_tmp
os.environ["TMP"] = local_tmp
print(f"🔧 Pasta temporária redirecionada para: {local_tmp}")

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache_local")
os.environ["HF_HUB_CACHE"] = os.path.join(os.getcwd(), "hf_cache_local")
# ⚡ Ativa download paralelo de shards via hf-transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# --- CONFIGURAÇÃO ---
MODELS_TO_BENCHMARK = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3.5-0.8B",
    # "Qwen/Qwen3.5-2B",
    # "Qwen/Qwen3.5-4B",
    # "Qwen/Qwen3.5-9B",
    # "google/gemma-3-1b-pt",
    # "google/gemma-3-1b-it",
    # "google/gemma-4-E2B-it",
    # "google/gemma-4-E4B-it",
    # "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    # "Polygl0t/Tucano2-0.6B-Base",
    # "Polygl0t/Tucano2-qwen-0.5B-Base",
    # "Polygl0t/Tucano2-qwen-1.5B-Base",
    # "Polygl0t/Tucano2-qwen-3.7B-Base",
    # "Polygl0t/Tucano2-qwen-0.5B-Instruct",
    # "Polygl0t/Tucano2-qwen-1.5B-Instruct",
    # "Polygl0t/Tucano2-qwen-3.7B-Instruct",
    # "TucanoBR/Tucano-1b1-Instruct",
    # "TucanoBR/Tucano-2b4-Instruct",  
    #"nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
]

#TARGET_LANGUAGES = ["en", "es", "fr", "ja", "pt"]
TARGET_LANGUAGES = ["pt"]


# ⚡ GPU_UTIL alto = mais KV-cache disponível = batches maiores = mais throughput.
#    Era 0.65 para modelos pequenos — deixava 35% da VRAM parada.
GPU_UTIL_DEFAULT = 0.90
GPU_UTIL_LARGE   = 0.88   # modelos >=13B com 8bit ainda precisam de margem

def install_dependencies():
    """Instala as dependências necessárias."""
    print("📦 Verificando dependências...")
    # vLLM 0.19.x = primeira versao estavel com suporte a transformers>=5.5
    # (necessario para Gemma 4). vLLM 0.7.1 e incompativel com transformers v5.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "vllm==0.19.1", "bitsandbytes==0.45.1", "hf-transfer==0.1.9", "psutil",
        "langdetect", "janome", "ja_sentence_segmenter", "spacy", "nltk", "accelerate",
    ])
    # vLLM 0.19.x pina transformers<=4.57.6 mas Gemma 4 exige >=5.5.0.
    # Instalamos transformers separadamente apos o vLLM para forcar a versao correta.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade", "transformers>=5.5.0",
    ])

    # Verifica se o transformers instalado suporta Qwen3.5.
    # Se nao (vocab_size ausente no Qwen3_5Config), instala do git HEAD.
    try:
        from transformers import AutoConfig
        AutoConfig.for_model("qwen3_5")  # lanca erro se nao suportado
    except (ValueError, AttributeError):
        print("🔧 transformers nao suporta Qwen3.5 ainda. Instalando do git HEAD...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "git+https://github.com/huggingface/transformers.git",
        ])
    except Exception:
        pass  # qualquer outro erro: ignora e deixa falhar no runtime

    import nltk
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')

    spacy_models = [
        "en_core_web_sm", "es_core_news_sm", "fr_core_news_sm",
        "ja_core_news_sm", "pt_core_news_sm", "xx_sent_ud_sm"
    ]
    print("Verificando Spacy...")
    for model in spacy_models:
        try:
            import spacy
            if not spacy.util.is_package(model):
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        except Exception:
            pass

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def force_gpu_cleanup(wait_for_free=False):
    """Limpeza de VRAM.

    wait_for_free=True: aguarda ate >85% da VRAM estar livre.
    Use entre modelos para garantir que o processo filho do vLLM morreu.
    """
    import subprocess as _sp
    # pkill e mais confiavel que psutil.children() porque o EngineCore e
    # filho do universal_inference.py (neto do benchmark_runner), entao
    # psutil.children() do benchmark_runner nao o enxerga.
    _sp.run(["pkill", "-9", "-f", "vllm.v1.engine.core"], check=False)
    _sp.run(["pkill", "-9", "-f", "EngineCore"], check=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if wait_for_free and torch.cuda.is_available():
        deadline = time.time() + 30
        while time.time() < deadline:
            free, total = torch.cuda.mem_get_info()
            free_gb, total_gb = free / 1024**3, total / 1024**3
            if free_gb > total_gb * 0.85:
                print(f"   ✅ VRAM pronta: {free_gb:.1f}/{total_gb:.1f} GiB livres")
                break
            print(f"   ⏳ Aguardando VRAM liberada... {free_gb:.1f}/{total_gb:.1f} GiB livres")
            time.sleep(2)
            _sp.run(["pkill", "-9", "-f", "vllm.v1.engine.core"], check=False)
            torch.cuda.empty_cache()
        else:
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"   ⚠️ Timeout aguardando VRAM ({free_gb:.1f} GiB livres). Continuando.")
    else:
        time.sleep(1)

def responses_exist(data_dir, safe_model_name, languages):
    return all(
        os.path.exists(
            os.path.join(data_dir, f"{lang}_input_response_data_{safe_model_name}.jsonl")
        )
        for lang in languages
    )

def build_inference_cmd(script_path, model, data_dir, languages):
    is_large = any(size in model.lower() for size in ["13b", "30b", "32b", "34b", "70b"])

    gpu_util = GPU_UTIL_LARGE if is_large else GPU_UTIL_DEFAULT

    cmd = [
        sys.executable, script_path,
        "--model_name", model,
        "--data_dir", data_dir,
        "--languages", *languages,
        # ⚡ batch_size maior: vLLM faz continuous batching, então este valor
        #    controla apenas o fallback Transformers. No vLLM não tem efeito direto.
        "--batch_size", "64" if not is_large else "16",
        "--gpu_memory_utilization", str(gpu_util),
    ]

    if is_large:
        print(f"   ⚖️ Modelo grande detectado ({model}). Ativando 8-bit...")
        cmd.append("--load_in_8bit")

    return cmd

def resolve_eval_script(project_dir):
    eval_script = os.path.join(project_dir, "evaluation_main.py")
    if not os.path.exists(eval_script):
        return "evaluation_main.py"
    return eval_script

def run_benchmark():
    benchmark_start = time.time()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_inference = os.path.join(project_dir, "data_nemo")
    os.makedirs(data_dir_inference, exist_ok=True)

    for model in MODELS_TO_BENCHMARK:
        print("\n🧹 Limpeza pré-modelo...")
        force_gpu_cleanup(wait_for_free=True)
        model_start = time.time()
        safe_model_name = model.replace('/', '__')

        print(f"\n{'='*50}\n🚀 INICIANDO: {model}\n{'='*50}")

        # --- 1. INFERÊNCIA ---
        print(">> Passo 1: Inferência Multi-Língua")

        # ⚡ Pula inferência se arquivo já existe (útil ao retomar após falha)
        if responses_exist(data_dir_inference, safe_model_name, TARGET_LANGUAGES):
            print(f"   ⏭️ Respostas já existem para {safe_model_name}. Pulando inferência.")
        else:
            force_gpu_cleanup()
            t0_inf = time.time()
            try:
                script_path = os.path.join(project_dir, "universal_inference.py")
                cmd = build_inference_cmd(script_path, model, data_dir_inference, TARGET_LANGUAGES)
                subprocess.run(cmd, check=True)
                print(f"⏱️ Tempo Inferência Total: {format_time(time.time() - t0_inf)}")
            except subprocess.CalledProcessError:
                print(f"❌ Falha crítica na inferência do modelo {model}. Tentando avaliar respostas já existentes.")
            finally:
                print("   🧹 Limpeza pós-inferência...")
                force_gpu_cleanup()

        # --- 2. AVALIAÇÃO ---
        print("\n>> Passo 2: Avaliação")
        force_gpu_cleanup()

        eval_script = resolve_eval_script(project_dir)
        for lang in TARGET_LANGUAGES:
            print(f"\n📊 Avaliando: {lang.upper()}")
            input_data = os.path.join(project_dir, "data", f"{lang}_input_data.jsonl")

            resp_candidates = [
                os.path.join(project_dir, "data_nemo", f"{lang}_input_response_data_{safe_model_name}.jsonl"),
                os.path.join(project_dir, "data", f"{lang}_input_response_data_{safe_model_name}.jsonl"),
            ]
            resp_file = next((p for p in resp_candidates if os.path.exists(p)), None)
            output_dir = os.path.join(project_dir, "eval_nemo", f"{lang}_input_response_data_{safe_model_name}")

            if resp_file:
                try:
                    force_gpu_cleanup()
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
                finally:
                    force_gpu_cleanup()
            else:
                print(f"   ⚠️ Arquivo de resposta não encontrado para {safe_model_name} em data_nemo/data")

        # --- 3. LIMPEZA ---
        print(f"\n>> Passo 3: Limpeza")
        force_gpu_cleanup()
        print(f"⏱️ Tempo total modelo: {format_time(time.time() - model_start)}")

    print(f"\n🎉 BENCHMARK GERAL FINALIZADO: {format_time(time.time() - benchmark_start)}")

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    install_dependencies()
    run_benchmark()
