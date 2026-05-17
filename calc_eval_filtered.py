import os
import json
import glob


# Caminho raiz das avaliacoes
EVAL_ROOT = "M-IFEvalFork/eval_new_3"

# Arquivo base de chaves validas
PT_INPUT_DATA = "M-IFEvalFork/data/pt_input_data.jsonl"

# Keys a serem ignoradas
IGNORED_KEYS = {
    211, 231, 244, 245, 286, 289, 303, 310, 316, 348, 358, 373, 383, 390,
    391, 392, 415, 418, 423, 424, 459, 464, 469, 476, 482, 498, 509, 511,
    521, 524, 526, 527, 534, 539, 541, 542, 544, 549, 550, 553, 557, 561,
    571, 573, 585, 592, 604, 613, 616, 620, 627, 631, 650, 651, 657, 672,
    674, 685, 686, 701, 702,
}


def normalize_key(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
    return None


def normalize_prompt(value):
    if not isinstance(value, str):
        return None
    dash_map = {
        ord("\u2010"): "-",
        ord("\u2011"): "-",
        ord("\u2012"): "-",
        ord("\u2013"): "-",
        ord("\u2014"): "-",
    }
    value = value.translate(dash_map)
    return " ".join(value.strip().split())


def load_valid_keys(file_path):
    valid_keys = set()
    prompt_to_key = {}
    if not os.path.exists(file_path):
        return valid_keys, prompt_to_key

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize_key(data.get("key"))
            prompt = normalize_prompt(data.get("prompt"))
            if key is not None:
                valid_keys.add(key)
                if prompt:
                    prompt_to_key.setdefault(prompt, key)
    valid_keys = valid_keys - IGNORED_KEYS
    prompt_to_key = {p: k for p, k in prompt_to_key.items() if k in valid_keys}
    return valid_keys, prompt_to_key


def calculate_accuracy(file_path, valid_keys, prompt_to_key):
    """Le um arquivo JSONL e calcula a % de acertos filtrando chaves."""
    if not os.path.exists(file_path):
        return None

    total = 0
    passed = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                prompt = normalize_prompt(data.get("prompt"))
                key = prompt_to_key.get(prompt)

                if key is None or key not in valid_keys:
                    continue

                total += 1
                if data.get("follow_all_instructions", False):
                    passed += 1

        if total == 0:
            return 0.0
        return (passed / total) * 100
    except Exception as e:
        print(f"❌ Erro ao ler {file_path}: {e}")
        return 0.0


def main():
    valid_keys, prompt_to_key = load_valid_keys(PT_INPUT_DATA)
    if not valid_keys:
        print("❌ Nenhuma key valida encontrada no arquivo base.")
        return

    print(f"{'MODELO / IDIOMA':<65} | {'STRICT %':<10} | {'LOOSE %':<10}")
    print("-" * 95)

    subdirs = [d for d in glob.glob(os.path.join(EVAL_ROOT, "*")) if os.path.isdir(d)]
    subdirs.sort()

    for folder in subdirs:
        folder_name = os.path.basename(folder)

        strict_path = os.path.join(folder, "eval_results_strict.jsonl")
        loose_path = os.path.join(folder, "eval_results_loose.jsonl")

        strict_score = calculate_accuracy(strict_path, valid_keys, prompt_to_key)
        loose_score = calculate_accuracy(loose_path, valid_keys, prompt_to_key)

        s_str = f"{strict_score:.2f}" if strict_score is not None else "N/A"
        l_str = f"{loose_score:.2f}" if loose_score is not None else "N/A"

        if strict_score is not None or loose_score is not None:
            print(f"{folder_name[:65]:<65} | {s_str:<10} | {l_str:<10}")


if __name__ == "__main__":
    main()
