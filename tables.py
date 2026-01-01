import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
pd.options.display.max_colwidth = 100
pd.options.display.float_format = '{:.1f}'.format

class MIFEvalAnalyzer:
    def __init__(self, folder_name='evaluations', target_lang='en'):
        script_location = os.path.dirname(os.path.abspath(__file__))
        self.evaluations_path = os.path.join(script_location, folder_name)
        self.target_lang = target_lang
        self.df_all_prompts = pd.DataFrame()
        self.df_all_instructions = pd.DataFrame()

    def _extract_category(self, instruction_id):
        try:
            parts = instruction_id.split(':')
            if len(parts) >= 2:
                return parts[1].replace('_', ' ').title()
        except:
            pass
        return "Other"

    def _extract_instruction_name(self, instruction_id):
        try:
            parts = instruction_id.split(':')
            if len(parts) >= 3:
                return parts[2].replace('_', ' ').title()
        except:
            pass
        return instruction_id

    def load_data(self):
        print(f"Local do Script:     {os.path.dirname(self.evaluations_path)}")
        print(f"Pasta Alvo:          {self.evaluations_path}")

        if not os.path.exists(self.evaluations_path):
             print(f"\nERRO: pasta não encontrada.")
             return

        found_files = []
        for root, dirs, files in os.walk(self.evaluations_path):
            for file in files:
                if file.lower().endswith('.jsonl') or file.lower().endswith('.json'):
                    found_files.append(os.path.join(root, file))

        if not found_files:
            print("\nNENHUM ARQUIVO DE DADOS ENCONTRADO!")
            return

        print(f"len(found_files) arquivos encontrados.")
        
        prompt_records = []
        instruction_records = []

        for filepath in found_files:
            folder_name = os.path.basename(os.path.dirname(filepath))
            file_name = os.path.basename(filepath)
            raw_name = os.path.splitext(file_name)[0]
            if folder_name != os.path.basename(self.evaluations_path):
                raw_name = folder_name

            if "input_response_data_" in raw_name:
                model_name = raw_name.split("input_response_data_")[-1]
            else:
                model_name = raw_name.replace('_results', '').replace('results_', '')
            
            if len(model_name) > 30: model_name = model_name[:27] + "..."

            try:
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data.append(json.loads(line))
                            except: pass
                    if not data:
                        f.seek(0)
                        try:
                            content = json.load(f)
                            if isinstance(content, list): data = content
                        except: pass

                for entry in data:
                    if not isinstance(entry, dict) or 'instruction_id_list' not in entry: continue
                    
                    first_id = entry['instruction_id_list'][0]
                    lang_code = first_id.split(':')[0] 
                    
                    # PROMPT LEVEL METRICS (Tabelas 1 e 7)
                    strict_pass = entry.get('follow_all_instructions', False)
                    loose_pass = entry.get('follow_all_instructions_loose', None)
                    if loose_pass is None:
                        loose_list = entry.get('follow_instruction_list_loose', [])
                        if loose_list: loose_pass = all(loose_list)
                        else: loose_pass = strict_pass

                    prompt_records.append({
                        'Model': model_name,
                        'Language': lang_code,
                        'Strict Pass': 1 if strict_pass else 0,
                        'Loose Pass': 1 if loose_pass else 0
                    })

                    # INSTRUCTION LEVEL METRICS (Tabelas 2, 6 e 8)
                    instr_ids = entry.get('instruction_id_list', [])
                    strict_results = entry.get('follow_instruction_list', [])
                    loose_results = entry.get('follow_instruction_list_loose', strict_results)

                    for i, inst_id in enumerate(instr_ids):
                        cat = self._extract_category(inst_id)
                        inst_name = self._extract_instruction_name(inst_id)
                        
                        s_res = strict_results[i] if i < len(strict_results) else False
                        l_res = loose_results[i] if i < len(loose_results) else s_res

                        instruction_records.append({
                            'Model': model_name,
                            'Language': lang_code,
                            'Category': cat,
                            'Instruction Type': inst_name,
                            'Strict Passed': 1 if s_res else 0,
                            'Loose Passed': 1 if l_res else 0
                        })
                
            except Exception as e:
                print(f"Erro ao ler {file_name}: {e}")

        self.df_all_prompts = pd.DataFrame(prompt_records)
        self.df_all_instructions = pd.DataFrame(instruction_records)

        print(f"\nBase Carregada: {len(self.df_all_prompts)} prompts e {len(self.df_all_instructions)} instruções individuais.")

    def _generate_pivot_table(self, df, value_col, filename, title):
        """Função auxiliar para criar e salvar tabelas Model x Language"""
        if df.empty: return
        pivot = df.groupby(['Model', 'Language'])[value_col].mean() * 100
        pivot = pivot.unstack(level='Language')
        pivot['Mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Mean', ascending=False)
        
        print(f"\n[{title.upper()}]")
        print(pivot.to_string())
        pivot.to_csv(filename)
        print(f"Salvo: {filename}")

    def generate_outputs(self):
        if self.df_all_prompts.empty: return

        # Tabela 1: Prompt-level Strict
        self._generate_pivot_table(
            self.df_all_prompts, 'Strict Pass', 
            'table_1_prompt_strict.csv', 'Table 1: Prompt-level Strict Accuracy'
        )

        # Tabela 2: Instruction-level Strict (A Média de acerto das instruções individuais)
        self._generate_pivot_table(
            self.df_all_instructions, 'Strict Passed', 
            'table_2_instruction_strict.csv', 'Table 2: Instruction-level Strict Accuracy'
        )

        # Tabela 6: Instruction-level Loose (A Média de acerto loose das instruções)
        self._generate_pivot_table(
            self.df_all_instructions, 'Loose Passed', 
            'table_6_instruction_loose.csv', 'Table 6: Instruction-level Loose Accuracy'
        )

        # Tabela 7: Prompt-level Loose
        self._generate_pivot_table(
            self.df_all_prompts, 'Loose Pass', 
            'table_7_prompt_loose.csv', 'Table 7: Prompt-level Loose Accuracy'
        )

        # Tabela 8: Detalhada por Tipo de Instrução
        if not self.df_all_instructions.empty:
            t8 = self.df_all_instructions.groupby(['Instruction Type', 'Language'])['Strict Passed'].mean() * 100
            t8_pivot = t8.unstack(level='Language')
            t8_pivot['Mean'] = t8_pivot.mean(axis=1)
            t8_pivot = t8_pivot.sort_values('Mean', ascending=False)

            print(f"\n[TABLE 8: Detailed Instruction Scores]")
            print(t8_pivot.head(10).to_string())
            t8_pivot.to_csv('table_8_detailed_instructions.csv')
            print("Salvo: table_8_detailed_instructions.csv")

        # GRÁFICOS VISUAIS (FOCADOS NO IDIOMA ALVO)

        print(f"GERANDO GRÁFICOS PARA: {self.target_lang.upper()}")
        
        df_chart_prompts = self.df_all_prompts[self.df_all_prompts['Language'] == self.target_lang]
        df_chart_inst = self.df_all_instructions[self.df_all_instructions['Language'] == self.target_lang]

        if df_chart_prompts.empty:
            print(f"Sem dados para '{self.target_lang.upper()}'.")
            return

        # 1. Pie Chart
        categories = df_chart_inst['Category'].tolist()
        if categories:
            counts = Counter(categories)
            labels, sizes = zip(*counts.most_common())
            plt.figure(figsize=(10, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
            plt.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
            plt.title(f'Distribuição de Tarefas ({self.target_lang.upper()})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'pie_distribution_{self.target_lang}.png')
            plt.close()

        # 2. Bar Chart (Category Strict)
        if not df_chart_inst.empty:
            cat_strict = df_chart_inst.groupby(['Model', 'Category'])['Strict Passed'].mean() * 100
            cat_strict = cat_strict.reset_index()
            plt.figure(figsize=(14, 7))
            sns.barplot(data=cat_strict, x='Category', y='Strict Passed', hue='Model', palette='viridis')
            plt.title(f'Category Accuracy - STRICT ({self.target_lang.upper()})', fontsize=14)
            plt.ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'category_strict_{self.target_lang}.png')
            plt.close()

        # 3. Bar Chart (Category Loose)
            cat_loose = df_chart_inst.groupby(['Model', 'Category'])['Loose Passed'].mean() * 100
            cat_loose = cat_loose.reset_index()
            plt.figure(figsize=(14, 7))
            sns.barplot(data=cat_loose, x='Category', y='Loose Passed', hue='Model', palette='rocket')
            plt.title(f'Category Accuracy - LOOSE ({self.target_lang.upper()})', fontsize=14)
            plt.ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'category_loose_{self.target_lang}.png')
            plt.close()

        # 4. Leaderboard Strict (Gráfico)
        lb_strict = df_chart_prompts.groupby('Model')['Strict Pass'].mean() * 100
        lb_strict = lb_strict.sort_values(ascending=False).reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=lb_strict, x='Model', y='Strict Pass', hue='Model', palette='Blues_d', legend=False)
        plt.title(f'Leaderboard Strict - {self.target_lang.upper()}', fontsize=14)
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'leaderboard_{self.target_lang}.png')
        plt.close()
        
        print("Todos os gráficos e tabelas foram gerados.")

if __name__ == "__main__":
    analyzer = MIFEvalAnalyzer(folder_name='evaluations', target_lang='pt')
    analyzer.load_data()
    analyzer.generate_outputs()