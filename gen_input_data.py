
from config import load_config, project_path
CONFIG = load_config('data_gen')
CFG = CONFIG['gen_input_data']

import instructions_registry
import secrets
import random
import json
from model_handler import safe_chat_call, get_model_id

# Adicionado: lista de opções extraídas manualmente (exemplos de 20+ prompts do diretório data)
EXTRACTED_OPTIONS = CFG['extracted_options']

CHECK_PROMPT = CFG['check_prompt']

REWRITE_PROMPT = CFG['rewrite_prompt']

def write_file(dict):
    with open(project_path(CFG['output_file']), CFG['output_mode'], encoding="utf-8") as fout:
        try:
            fout.write(json.dumps(dict, ensure_ascii=False) + "\n")
        except Exception:
            pass


def main():
    model_id = get_model_id()
    print(f'[START] Starting generating process, using model: {model_id}')

    # Substitui lógica anterior por iteração sobre todas as chaves que começam com 'pt:'
    pt_keys = [k for k in instructions_registry.INSTRUCTION_DICT.keys() if k.startswith(CFG['instruction_prefix'])]
    print(pt_keys)
    i = CFG['initial_key']
    if not pt_keys:
        print("Nenhuma chave 'pt:' encontrada no registro.")
    else:
        # Substitui impressão para prefixar uma instrução aleatória a cada descrição
        for key in pt_keys:
            try:
                instruction_cls = instructions_registry.INSTRUCTION_DICT[key]
                instruction = instruction_cls(key)
                desc = instruction.build_description()
                args = instruction.get_instruction_args()

                # prefixa a descrição principal com uma instrução aleatória
                prefix_main = secrets.choice(EXTRACTED_OPTIONS)
                prefixed_desc = f"{prefix_main}\n {desc}"

                # prepara lista de outras chaves disponíveis (exclui a atual)
                others = [k for k in pt_keys if k != key]
                if not others:
                    print("Sem outras instruções para combinar.")
                    continue

                # escolhe aleatoriamente 1 ou 2 outras instruções, respeitando a quantidade disponível
                n_choose = secrets.choice(CFG['combination_sizes'])
                n_choose = min(n_choose, len(others))
                sampled = random.sample(others, n_choose)

                # coleta descrições das instruções selecionadas e prefixa cada uma
                key_list = [key]
                args_list = [args]
                prefix_main = secrets.choice(EXTRACTED_OPTIONS)
                other_prefixed_desc = f"{prefix_main}\n {desc}"
                combo_descs = [other_prefixed_desc]
                for other_key in sampled:
                    try:
                        key_list.append(other_key)
                        other_cls = instructions_registry.INSTRUCTION_DICT[other_key]
                        other_inst = other_cls(other_key)
                        other_desc = other_inst.build_description()
                        other_prefixed = f"\n {other_desc}"
                        other_args = other_inst.get_instruction_args()
                        args_list.append(other_args)
                        combo_descs.append(other_prefixed)
                    except Exception as e:
                        print(f"Erro ao processar {other_key}: {e}")
                combo = "\n".join(combo_descs)

                message = [{"role": CFG['message_role'], "content": CHECK_PROMPT + '"'+ prefixed_desc+'"'}]
                result = safe_chat_call(message, model_id, None)
                last_line = result.strip().splitlines()[-1]
                # if last_line.lower().find("impossivel") != -1:
                #     print(prefixed_desc)
                #     print('impossivel')
                #     print(key)
                #     print('----------------------------')
                # else:
                #     print('Before:')
                #     print(prefixed_desc)
                #     message = [{"role": "system", "content": REWRITE_PROMPT + '"'+ prefixed_desc+'"'}]
                #     result = safe_chat_call(message, model_id, None)
                #     print('After:')
                #     print(result)
                #     print('----------------------------')
                #     data = {'key': i, 'instruction_id_list': [key], 'prompt': result, "kwargs": [args]}
                #     write_file(data)
                # i+=1

                message = [{"role": CFG['message_role'], "content": CHECK_PROMPT + '"'+ combo +'"'}]
                result = safe_chat_call(message, model_id, None)
                last_line = result.strip().splitlines()[-1]
                if last_line.lower().find(CFG['rejection_marker']) != -1:
                    print(combo)
                    print('impossivel')
                    print(key_list)
                    print('----------------------------')
                else:
                    print('Before:')
                    print(combo)
                    message = [{"role": CFG['message_role'], "content": REWRITE_PROMPT + '"'+ combo +'"'}]
                    result = safe_chat_call(message, model_id, None)
                    print('After:')
                    print(result)
                    print('----------------------------')
                    data = {'key': i, 'instruction_id_list': key_list, 'prompt': combo, "kwargs": args_list}
                    write_file(data)
                # print(data)
                i+=1

            except Exception as e:
                # Erro em uma entrada não interrompe as demais
                print(f"Erro ao processar {key}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate candidate Portuguese benchmark data.')
    parser.add_argument('--config', help='Data-generation YAML file.')
    parser.parse_args()
    main()
