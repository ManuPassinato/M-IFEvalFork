
from config import load_config, project_path
CONFIG = load_config('metrics')
CFG = CONFIG['tables']

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_theme(style=CFG['seaborn_theme'])
plt.rcParams.update({'font.size': CFG['font_size']})
pd.options.display.max_colwidth = CFG['dataframe_max_column_width']
pd.options.display.float_format = CFG['float_format'].format

class MIFEvalAnalyzer:
    def __init__(self, folder_name=None, target_lang=CFG['constructor_language']):
        if folder_name is None:
            folder_name = project_path(CONFIG['paths']['report_evaluations'])
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
                if any(file.lower().endswith(ext) for ext in CFG['accepted_extensions']):
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

            if CONFIG["io"]["response_marker"].lstrip("_") in raw_name:
                model_name = raw_name.split(CONFIG["io"]["response_marker"].lstrip("_"))[-1]
            else:
                model_name = raw_name
                for token in CFG['removable_model_tokens']:
                    model_name = model_name.replace(token, '')
            
            if len(model_name) > CFG["maximum_model_length"]: model_name = model_name[:CFG["truncated_model_length"]] + CFG["model_suffix"]

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
        pivot.to_csv(project_path(CONFIG["paths"]["report_output_dir"]) / filename)
        print(f"Salvo: {filename}")

    def generate_outputs(self):
        if self.df_all_prompts.empty: return

        # Tabela 1: Prompt-level Strict
        self._generate_pivot_table(
            self.df_all_prompts, 'Strict Pass', 
            CFG['prompt_strict_csv'], 'Table 1: Prompt-level Strict Accuracy'
        )

        # Tabela 2: Instruction-level Strict (A Média de acerto das instruções individuais)
        self._generate_pivot_table(
            self.df_all_instructions, 'Strict Passed', 
            CFG['instruction_strict_csv'], 'Table 2: Instruction-level Strict Accuracy'
        )

        # Tabela 6: Instruction-level Loose (A Média de acerto loose das instruções)
        self._generate_pivot_table(
            self.df_all_instructions, 'Loose Passed', 
            CFG['instruction_loose_csv'], 'Table 6: Instruction-level Loose Accuracy'
        )

        # Tabela 7: Prompt-level Loose
        self._generate_pivot_table(
            self.df_all_prompts, 'Loose Pass', 
            CFG['prompt_loose_csv'], 'Table 7: Prompt-level Loose Accuracy'
        )

        # Tabela 8: Detalhada por Tipo de Instrução
        if not self.df_all_instructions.empty:
            t8 = self.df_all_instructions.groupby(['Instruction Type', 'Language'])['Strict Passed'].mean() * 100
            t8_pivot = t8.unstack(level='Language')
            t8_pivot['Mean'] = t8_pivot.mean(axis=1)
            t8_pivot = t8_pivot.sort_values('Mean', ascending=False)

            print(f"\n[TABLE 8: Detailed Instruction Scores]")
            print(t8_pivot.head(CFG['detailed_row_limit']).to_string())
            t8_pivot.to_csv(project_path(CONFIG["paths"]["report_output_dir"]) / CFG['detailed_instructions_csv'])
            print(f"Salvo: {CFG['detailed_instructions_csv']}")

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
            plt.figure(figsize=CFG['pie_size'])
            plt.pie(sizes, labels=labels, autopct=CFG['pie_percentage_format'], startangle=CFG['pie_start_angle'], pctdistance=CFG['pie_percentage_distance'])
            plt.gca().add_artist(plt.Circle((0,0),CFG["center_radius"],fc=CFG['center_color']))
            plt.title(f'Distribuição de Tarefas ({self.target_lang.upper()})', fontsize=CFG['title_font_size'])
            plt.tight_layout()
            plt.savefig(project_path(CONFIG["paths"]["report_output_dir"]) / CFG['output_pie_distribution'].format(language=self.target_lang))
            plt.close()

        # 2. Bar Chart (Category Strict)
        if not df_chart_inst.empty:
            cat_strict = df_chart_inst.groupby(['Model', 'Category'])['Strict Passed'].mean() * 100
            cat_strict = cat_strict.reset_index()
            plt.figure(figsize=CFG['category_size'])
            sns.barplot(data=cat_strict, x='Category', y='Strict Passed', hue='Model', palette=CFG['strict_palette'])
            plt.title(f'Category Accuracy - STRICT ({self.target_lang.upper()})', fontsize=CFG['title_font_size'])
            plt.ylim(CFG['y_axis_min'], CFG['y_axis_max'])
            plt.xticks(rotation=CFG["x_axis_rotation"], ha=CFG['x_axis_alignment'])
            plt.legend(bbox_to_anchor=CFG['legend_bbox'], loc=CFG['legend_location'])
            plt.tight_layout()
            plt.savefig(project_path(CONFIG["paths"]["report_output_dir"]) / CFG['output_category_strict'].format(language=self.target_lang))
            plt.close()

        # 3. Bar Chart (Category Loose)
            cat_loose = df_chart_inst.groupby(['Model', 'Category'])['Loose Passed'].mean() * 100
            cat_loose = cat_loose.reset_index()
            plt.figure(figsize=CFG['category_size'])
            sns.barplot(data=cat_loose, x='Category', y='Loose Passed', hue='Model', palette=CFG['loose_palette'])
            plt.title(f'Category Accuracy - LOOSE ({self.target_lang.upper()})', fontsize=CFG['title_font_size'])
            plt.ylim(CFG['y_axis_min'], CFG['y_axis_max'])
            plt.xticks(rotation=CFG["x_axis_rotation"], ha=CFG['x_axis_alignment'])
            plt.legend(bbox_to_anchor=CFG['legend_bbox'], loc=CFG['legend_location'])
            plt.tight_layout()
            plt.savefig(project_path(CONFIG["paths"]["report_output_dir"]) / CFG['output_category_loose'].format(language=self.target_lang))
            plt.close()

        # 4. Leaderboard Strict (Gráfico)
        lb_strict = df_chart_prompts.groupby('Model')['Strict Pass'].mean() * 100
        lb_strict = lb_strict.sort_values(ascending=False).reset_index()
        plt.figure(figsize=CFG['leaderboard_size'])
        sns.barplot(data=lb_strict, x='Model', y='Strict Pass', hue='Model', palette=CFG['leaderboard_palette'], legend=False)
        plt.title(f'Leaderboard Strict - {self.target_lang.upper()}', fontsize=CFG['title_font_size'])
        plt.ylim(CFG['y_axis_min'], CFG['y_axis_max'])
        plt.xticks(rotation=CFG["x_axis_rotation"], ha=CFG['x_axis_alignment'])
        plt.tight_layout()
        plt.savefig(project_path(CONFIG["paths"]["report_output_dir"]) / CFG['output_leaderboard'].format(language=self.target_lang))
        plt.close()
        
        print("Todos os gráficos e tabelas foram gerados.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("--config", help="Metrics YAML configuration file.")
    parser.parse_args()
    project_path(CONFIG["paths"]["report_output_dir"]).mkdir(parents=True, exist_ok=True)
    analyzer = MIFEvalAnalyzer(folder_name=project_path(CONFIG['paths']['report_evaluations']), target_lang=CFG['target_language'])
    analyzer.load_data()
    analyzer.generate_outputs()
