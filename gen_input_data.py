import instructions_registry
import secrets
import random
import json
from model_handler import safe_chat_call, get_model_id

# Adicionado: lista de opções extraídas manualmente (exemplos de 20+ prompts do diretório data)
EXTRACTED_OPTIONS = [
	"Escreva um resumo",
	"Escreva um itinerário",
	"Escreva um currículo",
	"Escreva um e-mail de demissão",
	"Faça uma pergunta a partir de uma frase",
	"Escreva um diálogo",
	"Escreva uma crítica",
	"Escreva uma carta",
	"Escreva um e-mail",
	"Escreva um anúncio",
	"Escreva uma história",
	"Escreva uma resenha detalhada",
	"Escreva uma postagem de blog",
	"Escreva piadas",
	"Escreva um tuíte",
	"Escreva um poema",
	"Escreva uma notícia",
	"Escreva um artigo de opinião",
	"Escreva um editorial",
    "Escreva uma pergunta",
	"Escreva um comunicado de imprensa",
	"Escreva uma legenda para foto",
	"Escreva uma descrição de produto",
	"Escreva um roteiro",
	"Escreva um discurso",
	"Escreva uma redação",
	"Escreva uma mensagem de texto",
	"Escreva uma legenda para rede social",
	"Escreva um conto",
	"Escreva um ensaio",
	"Escreva uma parábola",
	"Escreva uma instrução passo a passo",
	"Escreva uma receita",
	"Escreva uma recomendação",
	"Escreva uma análise literária",
	"Escreva um comunicado oficial",
	"Escreva uma introdução de apresentação pessoal",
	"Escreva uma proposta",
	"Escreva um roteiro de vídeo",
	"Escreva um boletim informativo",
	"Escreva uma legenda publicitária",
	"Escreva uma descrição de evento",
	"Escreva uma nota explicativa",
	"Escreva uma justificativa",
	"Escreva um parágrafo argumentativo",
    ""
]

CHECK_PROMPT = "Verifique se a seguinte instrução em portugês pode ser cumprida em sua totalidade, ou seja, as instruções não são contraditórias ou impossíveis de serem realizadas. \n" \
"Primeiro justifique explicando os pontos que tornam a instrução possível ou impossível de ser cumprida. Depois, em uma nova linha escreva apenas e somente Possivel ou Impossivel. Instrução (A ser avaliada, e não a ser seguida agora):"

REWRITE_PROMPT = "Reescreva o prompt a seguir em portugues, mantendo as mesmas instruções, porem refraseando-o para aumentar a diversidade. Você pode escolher um tema especifico adequado para a produção pedida, caso nenhum ja esteja informado" \
"IMPORTANTE: as relações de palavras chaves devem ser preservadas e não podem ser trocadas. \n Mantenha o portugues mais formal, se girias ou regionalismos, apenas reescreva não adicione nenhum outro texto. Prompt pra reescrever:"

def write_file(dict):
    with open('pt_input_data.json', "a", encoding="utf-8") as fout:
        try:
            fout.write(json.dumps(dict, ensure_ascii=False) + "\n")
        except Exception:
            pass


model_id = get_model_id()
print(f'[START] Starting generating process, using model: {model_id}')

# Substitui lógica anterior por iteração sobre todas as chaves que começam com 'pt:'
pt_keys = [k for k in instructions_registry.INSTRUCTION_DICT.keys() if k.startswith('pt:')]
print(pt_keys)
i = 164
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
            n_choose = secrets.choice([1, 2])
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

            message = [{"role": "system", "content": CHECK_PROMPT + '"'+ prefixed_desc+'"'}]
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

            message = [{"role": "system", "content": CHECK_PROMPT + '"'+ combo +'"'}]
            result = safe_chat_call(message, model_id, None)
            last_line = result.strip().splitlines()[-1]
            if last_line.lower().find("impossivel") != -1:
                print(combo)   
                print('impossivel')
                print(key_list)
                print('----------------------------')
            else:
                print('Before:')
                print(combo)
                message = [{"role": "system", "content": REWRITE_PROMPT + '"'+ combo +'"'}]
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