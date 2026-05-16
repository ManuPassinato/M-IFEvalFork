import unittest
import sys
import re
from unittest.mock import MagicMock

try:
    from instruction_utils import pt_instructions_util
    from instructions import pt_instructions
except ImportError:
    print("\nERRO CRÍTICO: Não foi possível importar 'instructions' ou 'instruction_utils'.")
    print("Certifique-se de estar rodando este arquivo na raiz do repositório.")
    sys.exit(1)


def fast_count_sentences(text):
    return len(re.findall(r'[.!?]+', text))

def fast_count_words(text):
    return len(text.split())

def fast_split_into_sentences(text):
    text = text.strip()
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def fast_word_tokenize(text):
    return text.split()
pt_instructions_util.count_sentences = fast_count_sentences
pt_instructions_util.count_words = fast_count_words
pt_instructions_util.split_into_sentences = fast_split_into_sentences
pt_instructions_util.nltk = MagicMock()
pt_instructions_util.nltk.word_tokenize = fast_word_tokenize
pt_instructions.langdetect = MagicMock()
pt_instructions.langdetect.detect.return_value = "pt"

class TestAllInstructionsPT(unittest.TestCase):
    
    # 1. ResponseLanguageChecker
    def test_01_response_language_checker(self):
        instruction = pt_instructions.ResponseLanguageChecker("id")
        instruction.build_description(language="pt")
        # Mockamos o retorno para "pt" no setup, então deve passar
        self.assertTrue(instruction.check_following("Texto em português."))

    # 2. NumberOfSentences
    def test_02_number_of_sentences(self):
        instruction = pt_instructions.NumberOfSentences("id")
        instruction.build_description(num_sentences=3, relation="menos que")
        self.assertTrue(instruction.check_following("Uma. Duas."), "Deveria aceitar 2 (< 3)")
        self.assertFalse(instruction.check_following("Uma. Duas. Três."), "Deveria rejeitar 3")

    # 3. PlaceholderChecker
    def test_03_placeholder_checker(self):
        instruction = pt_instructions.PlaceholderChecker("id")
        instruction.build_description(num_placeholders=1)
        self.assertTrue(instruction.check_following("Olá [nome]."))
        self.assertFalse(instruction.check_following("Olá nome."))

    # 4. BulletListChecker
    def test_04_bullet_list_checker(self):
        instruction = pt_instructions.BulletListChecker("id")
        instruction.build_description(num_bullets=2)
        # Regex original procura linhas começando com * ou -
        self.assertTrue(instruction.check_following("* Item 1\n* Item 2"))
        self.assertFalse(instruction.check_following("Item 1\nItem 2"))

    # 5. ConstrainedResponseChecker
    def test_05_constrained_response_checker(self):
        instruction = pt_instructions.ConstrainedResponseChecker("id")
        instruction.build_description() # Opções padrão: Sim., Não., Talvez.
        self.assertTrue(instruction.check_following("Sim."))
        self.assertFalse(instruction.check_following("Com certeza."))

    # 6. ConstrainedStartChecker
    def test_06_constrained_start_checker(self):
        instruction = pt_instructions.ConstrainedStartChecker("id")
        instruction.build_description(starter="Eu acredito")
        self.assertTrue(instruction.check_following("Eu acredito que vai chover."))
        self.assertFalse(instruction.check_following("Vai chover."))

    # 7. HighlightSectionChecker
    def test_07_highlight_section_checker(self):
        instruction = pt_instructions.HighlightSectionChecker("id")
        instruction.build_description(num_highlights=1)
        self.assertTrue(instruction.check_following("Texto *destacado*."))
        self.assertFalse(instruction.check_following("Texto normal."))

    # 8. SectionChecker
    def test_08_section_checker(self):
        instruction = pt_instructions.SectionChecker("id")
        instruction.build_description(num_sections=2, section_spliter="Sessão")
        # O padrão regex espera algo como "Sessão 1", "Sessão 2"
        self.assertTrue(instruction.check_following("Intro Sessão 1 Conteúdo Sessão 2 Fim"))
        self.assertFalse(instruction.check_following("Texto corrido sem divisão."))

    # 9. ParagraphChecker
    def test_09_paragraph_checker(self):
        instruction = pt_instructions.ParagraphChecker("id")
        instruction.build_description(num_paragraphs=2)
        # Divisor padrão é ***
        self.assertTrue(instruction.check_following("Parágrafo 1\n***\nParágrafo 2"))
        self.assertFalse(instruction.check_following("Apenas um bloco."))

    # 10. PostscriptChecker
    def test_10_postscript_checker(self):
        instruction = pt_instructions.PostscriptChecker("id")
        instruction.build_description(postscript_marker="P.S.")
        self.assertTrue(instruction.check_following("Texto principal.\nP.S. esqueci isso."))
        self.assertFalse(instruction.check_following("Texto sem pós-escrito."))

    # 11. RephraseChecker
    def test_11_rephrase_checker(self):
        instruction = pt_instructions.RephraseChecker("id")
        # O texto original deve ter *mudança*
        original = "O céu é *azul*."
        instruction.build_description(original_message=original)
        
        # O check_following verifica se a parte FORA dos asteriscos é igual
        # e se a resposta também tem a estrutura de mudança
        self.assertTrue(instruction.check_following("O céu é *vermelho*.")) 
        self.assertFalse(instruction.check_following("O mar é *azul*.")) # Mudou fora do asterisco

    # 12. KeywordChecker
    def test_12_keyword_checker(self):
        instruction = pt_instructions.KeywordChecker("id")
        instruction.build_description(keywords=["obrigado"])
        self.assertTrue(instruction.check_following("Muito obrigado!"))
        self.assertFalse(instruction.check_following("Valeu!"))

    # 13. KeywordFrequencyChecker
    def test_13_keyword_frequency_checker(self):
        instruction = pt_instructions.KeywordFrequencyChecker("id")
        instruction.build_description(keyword="teste", frequency=2, relation="ao menos")
        self.assertTrue(instruction.check_following("Isso é um teste de teste."))
        self.assertFalse(instruction.check_following("Apenas um teste."))

    # 14. NumberOfWords
    def test_14_number_of_words(self):
        instruction = pt_instructions.NumberOfWords("id")
        instruction.build_description(num_words=5, relation="menos que")
        self.assertTrue(instruction.check_following("Um dois três."))
        self.assertFalse(instruction.check_following("Um dois três quatro cinco seis."))

    # 15. JsonFormat
    def test_15_json_format(self):
        instruction = pt_instructions.JsonFormat("id")
        instruction.build_description()
        self.assertTrue(instruction.check_following('```json\n{"chave": "valor"}\n```'))
        self.assertFalse(instruction.check_following('Chave: valor'))

    # 16. ParagraphFirstWordCheck
    def test_16_paragraph_first_word_check(self):
        instruction = pt_instructions.ParagraphFirstWordCheck("id")
        # Espera 2 parágrafos (sep por \n\n), o 1º deve começar com "Olá"
        instruction.build_description(num_paragraphs=2, nth_paragraph=1, first_word="Olá")
        self.assertTrue(instruction.check_following("Olá mundo.\n\nSegundo parágrafo."))
        self.assertFalse(instruction.check_following("Tchau mundo.\n\nSegundo parágrafo."))

    # 17. KeySentenceChecker
    def test_17_key_sentence_checker(self):
        instruction = pt_instructions.KeySentenceChecker("id")
        instruction.build_description(key_sentences=["O dia está lindo"], num_sentences=1)
        self.assertTrue(instruction.check_following("Olá. O dia está lindo."))
        self.assertFalse(instruction.check_following("Olá. O dia está feio."))

    # 18. ForbiddenWords
    def test_18_forbidden_words(self):
        instruction = pt_instructions.ForbiddenWords("id")
        instruction.build_description(forbidden_words=["senha"])
        self.assertTrue(instruction.check_following("Meu acesso."))
        self.assertFalse(instruction.check_following("Minha senha é 123."))

    # 19. RephraseParagraph
    def test_19_rephrase_paragraph(self):
        instruction = pt_instructions.RephraseParagraph("id")
        # Verifica se a resposta tem entre Low e High palavras em comum com o original
        original = "Gato come rato"
        instruction.build_description(original_paragraph=original, low=1, high=2)
        
        self.assertTrue(instruction.check_following("Gato dorme")) # 1 palavra igual (Gato) -> OK
        self.assertFalse(instruction.check_following("Cachorro dorme")) # 0 palavras iguais -> Fail

    # 20. TwoResponsesChecker
    def test_20_two_responses_checker(self):
        instruction = pt_instructions.TwoResponsesChecker("id")
        instruction.build_description()
        # Separador é ******
        self.assertTrue(instruction.check_following("Resposta A ****** Resposta B"))
        self.assertFalse(instruction.check_following("Resposta única"))

    # 21. RepeatPromptThenAnswer
    def test_21_repeat_prompt_then_answer(self):
        instruction = pt_instructions.RepeatPromptThenAnswer("id")
        instruction.build_description(prompt_to_repeat="Pergunta")
        self.assertTrue(instruction.check_following("Pergunta resposta"))
        self.assertFalse(instruction.check_following("Apenas resposta"))

    # 22. EndChecker
    def test_22_end_checker(self):
        instruction = pt_instructions.EndChecker("id")
        instruction.build_description(end_phrase="Fim.")
        self.assertTrue(instruction.check_following("Texto texto. Fim."))
        self.assertFalse(instruction.check_following("Texto texto."))

    # 23. TitleChecker
    def test_23_title_checker(self):
        instruction = pt_instructions.TitleChecker("id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("<<Meu Título>>\nTexto."))
        self.assertFalse(instruction.check_following("Meu Título\nTexto."))

    # 24. LetterFrequencyChecker
    def test_24_letter_frequency_checker(self):
        instruction = pt_instructions.LetterFrequencyChecker("id")
        instruction.build_description(letter="a", let_frequency=2, let_relation="ao menos")
        self.assertTrue(instruction.check_following("Casa")) # Tem 2 'a's
        self.assertFalse(instruction.check_following("Bo"))   # Tem 0 'a's

    # 25. CapitalLettersPortugueseChecker
    def test_25_capital_letters_pt(self):
        instruction = pt_instructions.CapitalLettersPortugueseChecker("id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("TUDO MAIÚSCULO"))
        self.assertFalse(instruction.check_following("Tudo Maiúsculo"))

    # 26. LowercaseLettersPortugueseChecker
    def test_26_lowercase_letters_pt(self):
        instruction = pt_instructions.LowercaseLettersPortugueseChecker("id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("tudo minúsculo"))
        self.assertFalse(instruction.check_following("Tudo Minúsculo"))

    # 27. CommaChecker
    def test_27_comma_checker(self):
        instruction = pt_instructions.CommaChecker("id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("Sem vírgulas aqui"))
        self.assertFalse(instruction.check_following("Com, vírgula"))

    # 28. CapitalWordFrequencyChecker
    def test_28_capital_word_frequency(self):
        instruction = pt_instructions.CapitalWordFrequencyChecker("id")
        instruction.build_description(capital_frequency=1, capital_relation="ao menos")
        # Nosso mock de tokenização separa por espaço. "TESTE" é isupper() -> True
        self.assertTrue(instruction.check_following("Este é um TESTE"))
        self.assertFalse(instruction.check_following("Este é um teste"))

    # 29. QuotationChecker
    def test_29_quotation_checker(self):
        instruction = pt_instructions.QuotationChecker("id")
        instruction.build_description()
        self.assertTrue(instruction.check_following('"Texto entre aspas"'))
        self.assertFalse(instruction.check_following('Texto sem aspas'))

    # 30. CedilhaFrequencyChecker
    def test_30_cedilha_frequency_checker(self):
        instruction = pt_instructions.CedilhaFrequencyChecker("test_cedilha_id")
        instruction.build_description(count=2)
        self.assertTrue(instruction.check_following("Ação e reação."))
        self.assertFalse(instruction.check_following("Ação apenas."))

    # 31. NoTildeChecker
    def test_31_no_tilde_checker(self):
        instruction = pt_instructions.NoTildeChecker("test_tilde_id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("Tudo bem?"))
        self.assertTrue(instruction.check_following("A crase à noite está correta."))
        self.assertTrue(instruction.check_following("À tarde, há café e música clássica."))
        self.assertTrue(instruction.check_following("Açaí e café combinam bem."))
        self.assertFalse(instruction.check_following("Não."))
        self.assertFalse(instruction.check_following("Põe isso aqui."))
        self.assertFalse(instruction.check_following("Senõrita."))

    # 32. CrasePresenceChecker
    def test_32_crase_presence_checker(self):
        instruction = pt_instructions.CrasePresenceChecker("test_crase_id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("Fui à feira."))
        self.assertFalse(instruction.check_following("Fui a feira."))

    # 33. MesocliseChecker
    def test_33_mesoclise_checker(self):
        instruction = pt_instructions.MesocliseChecker("test_mesoclise_id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("Comprar-te-ei um carro."))
        self.assertFalse(instruction.check_following("Vou te comprar."))

    # 34. VOSAddressChecker
    def test_34_vos_address_checker(self):
        instruction = pt_instructions.VOSAddressChecker("test_vos_id")
        instruction.build_description()
        self.assertTrue(instruction.check_following("Vós sabeis a verdade."))
        self.assertFalse(instruction.check_following("Você sabe a verdade."))

    # 35. SpecificPorqueGrammarChecker
    def test_35_specific_porque_grammar(self):
        instruction = pt_instructions.SpecificPorqueGrammarChecker("test_single_porque_id")

        with self.assertRaises(ValueError):
            instruction.build_description(porque_type="por qué")

        # 1. "porque" como conjunção causal/explicativa
        instruction.build_description(porque_type="porque")
        self.assertTrue(instruction.check_following("O evento foi cancelado porque choveu muito."))
        self.assertTrue(instruction.check_following("Você saiu porque quis?"))
        self.assertTrue(instruction.check_following("Porque estava cansado, ele foi embora mais cedo."))
        self.assertFalse(instruction.check_following("Porque o evento foi cancelado?"))
        self.assertFalse(instruction.check_following("Não entendi o porque do cancelamento."))
        self.assertFalse(instruction.check_following("O evento foi cancelado por que choveu muito."))

        # 2. "porquê" como substantivo, singular ou plural
        instruction.build_description(porque_type="porquê")
        self.assertTrue(instruction.check_following("Gostaria de compreender o porquê de tanta confusão."))
        self.assertTrue(instruction.check_following("Não existe nenhum porquê para essa atitude."))
        self.assertTrue(instruction.check_following("Todo porquê tem uma resposta."))
        self.assertTrue(instruction.check_following("Seus porquês continuam obscuros para nós."))
        self.assertTrue(instruction.check_following("Esses porquês ainda me intrigam."))
        self.assertFalse(instruction.check_following("Gostaria de compreender porquê de tanta confusão."))
        self.assertFalse(instruction.check_following("Gostaria de compreender o porque de tanta confusão."))

        # 3. "por que" em pergunta direta, indireta ou valor relativo
        instruction.build_description(porque_type="por que")
        self.assertTrue(instruction.check_following("Por que o céu é azul?"))
        self.assertTrue(instruction.check_following("Gostaria de saber por que o céu é azul."))
        self.assertTrue(instruction.check_following("Não sei por que o céu é azul."))
        self.assertTrue(instruction.check_following("Desconheço o motivo por que ela partiu."))
        self.assertTrue(instruction.check_following("Essa é a razão por que não fomos ao cinema."))
        self.assertFalse(instruction.check_following("Porque o céu é azul?"))
        self.assertFalse(instruction.check_following("Vocês saíram por que?"))
        self.assertFalse(instruction.check_following("Quero entender o porquê da escolha."))

        # 4. "por quê" em posição final antes de pontuação
        instruction.build_description(porque_type="por quê")
        self.assertTrue(instruction.check_following("Vocês estão a rir de por quê?"))
        self.assertTrue(instruction.check_following("A rua está fechada, por quê, alguém sabe?"))
        self.assertTrue(instruction.check_following("Ele faltou de novo (e eu não sei por quê)."))
        self.assertTrue(instruction.check_following("Ninguém explicou por quê."))
        self.assertFalse(instruction.check_following("Por quê vocês estão a rir?"))
        self.assertFalse(instruction.check_following("Vocês estão a rir de por que?"))

        instruction.build_description(porque_type="porque")
        self.assertFalse(instruction.check_following("O evento foi cancelado devido à chuva."))
        self.assertFalse(instruction.check_following("Por que choveu? Porque era inverno."))



if __name__ == '__main__':
    # Executa todos os testes e mostra detalhe
    print("Iniciando execução...")
    unittest.main(verbosity=2)
