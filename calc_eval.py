import os
import json
import glob

# Caminho raiz das avaliações
EVAL_ROOT = "/workspace/M-IFEvalFork/evaluations/"

def calculate_accuracy(file_path):
    """Lê um arquivo JSONL e calcula a % de acertos"""
    if not os.path.exists(file_path):
        return None
    
    total = 0
    passed = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                total += 1
                
                # --- CORREÇÃO AQUI ---
                # Seus arquivos usam "follow_all_instructions" (True/False)
                if data.get("follow_all_instructions", False):
                    passed += 1
        
        if total == 0: return 0.0
        return (passed / total) * 100
    except Exception as e:
        print(f"❌ Erro ao ler {file_path}: {e}")
        return 0.0

def main():
    print(f"{'MODELO / IDIOMA':<65} | {'STRICT %':<10} | {'LOOSE %':<10}")
    print("-" * 95)
    
    # Encontra todas as subpastas dentro de evaluations
    subdirs = [d for d in glob.glob(os.path.join(EVAL_ROOT, "*")) if os.path.isdir(d)]
    subdirs.sort()

    for folder in subdirs:
        folder_name = os.path.basename(folder)
        
        # Caminhos dos arquivos
        strict_path = os.path.join(folder, "eval_results_strict.jsonl")
        loose_path = os.path.join(folder, "eval_results_loose.jsonl")
        
        # Calcula apenas se os arquivos existirem
        strict_score = calculate_accuracy(strict_path)
        loose_score = calculate_accuracy(loose_path)
        
        # Formata a saída
        s_str = f"{strict_score:.2f}" if strict_score is not None else "N/A"
        l_str = f"{loose_score:.2f}" if loose_score is not None else "N/A"
        
        # Só imprime se tiver algum resultado
        if strict_score is not None or loose_score is not None:
            print(f"{folder_name[:65]:<65} | {s_str:<10} | {l_str:<10}")

if __name__ == "__main__":
    main()