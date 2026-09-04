"""Offline configuration and differential regression tests.

Baseline: the unmodified repository commit before the YAML migration. Set
IFEVAL_BASELINE_DIR to an extracted copy when Git history is unavailable.
Model backends and API calls are simulated; no model downloads are performed.
"""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import yaml

BASELINE = 'cd3b32eb8400434e424eb214c96db6c952239880'


def original(relative):
    if os.getenv('IFEVAL_BASELINE_DIR'):
        return (Path(os.environ['IFEVAL_BASELINE_DIR']) / relative).read_text(encoding='utf-8')
    result = subprocess.run(['git', 'show', f'{BASELINE}:{relative}'], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise unittest.SkipTest('Baseline unavailable; set IFEVAL_BASELINE_DIR')
    return result.stdout.decode('utf-8')


def module(source, name, filename):
    obj = types.ModuleType(name)
    obj.__file__ = str(filename)
    with patch.dict(sys.modules, {name: obj}), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, str(filename), 'exec'), obj.__dict__)
    return obj


class ConfigurationTests(unittest.TestCase):
    def test_reports_follow_generated_evaluations(self):
        data = config.load_config('metrics')
        self.assertEqual(data['paths']['report_evaluations'], data['paths']['generated_evaluations'])
        paths = {'generated_evaluations': 'experiments/custom', 'report_evaluations': None, 'report_output_dir': '.'}
        with patch.object(config, '_read', return_value={'paths': paths}):
            self.assertEqual(config.load_section('metrics', 'paths')['report_evaluations'], 'experiments/custom')
        paths['report_evaluations'] = 'experiments/history'
        with patch.object(config, '_read', return_value={'paths': paths}):
            self.assertEqual(config.load_section('metrics', 'paths')['report_evaluations'], 'experiments/history')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            metrics = config.load_config('metrics')
            metrics.pop('io')
            metrics['source']['config'] = str(ROOT / 'inference.yml')
            metrics['paths']['generated_evaluations'] = str(folder / 'evaluations')
            metrics['paths']['report_evaluations'] = None
            target = folder / 'evaluations/pt_test'
            target.mkdir(parents=True)
            for mode, scores in [('strict', [True, False]), ('loose', [True, True])]:
                filename = metrics['evaluation_main'][mode + '_filename']
                (target / filename).write_text('\n'.join(json.dumps({'follow_all_instructions': score}) for score in scores), encoding='utf-8')
            selected = folder / 'metrics.yml'
            selected.write_text(yaml.safe_dump(metrics), encoding='utf-8')
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'calc_eval.py'), '--config', str(selected)], cwd=folder, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('pt_test', result.stdout)
            self.assertIn('50.00', result.stdout)
            self.assertIn('100.00', result.stdout)

    def test_section_loading_does_not_resolve_other_stages(self):
        nlp = config.load_section('metrics', 'nlp')
        with patch.object(config, '_read', return_value={'nlp': nlp}) as reader:
            self.assertEqual(config.load_section('metrics', 'nlp'), nlp)
            reader.assert_called_once()
            self.assertTrue(reader.call_args.args[0].endswith('metrics.yml'))
        defaults = config.load_section('data_gen', 'instruction_defaults', 'pt')
        with patch.object(config, '_read', return_value={'instruction_defaults': {'pt': defaults}, 'model_handler': 'invalid but unused'}):
            selected = config.load_section('data_gen', 'instruction_defaults', 'pt')
            selected['num_bullets'] = 999
            self.assertNotEqual(selected, defaults)

    def test_operational_validation(self):
        cases = [
            ('benchmark_runner', 'batch_size', 0),
            ('benchmark_runner', 'gpu_util_large', 2),
            ('universal_inference', 'prefer_transformers', 'false'),
            ('universal_inference', 'qwen3_temperature', -1),
            ('universal_inference', 'qwen3_temperature', float('nan')),
            ('universal_inference', 'repetition_penalty', 0),
            ('universal_inference', 'transformers_stop_strings', ['unused']),
            ('get_responses', 'use_progress_bar', 'true'),
        ]
        for section, key, invalid in cases:
            data = config.load_config('inference')
            data[section][key] = invalid
            with self.subTest(section=section, key=key), patch.object(config, '_read', return_value=data), self.assertRaises(ValueError):
                config.load_config('inference')
        for key, invalid in [('retry_attempts', 0), ('retry_wait_seconds', -1), ('enable_thinking', 'false'), ('resp_max_tokens_default', 0)]:
            data = config.load_config('data_gen')
            data['model_handler'][key] = invalid
            with self.subTest(key=key), patch.object(config, '_read', return_value=data), self.assertRaises(ValueError):
                config.load_section('data_gen', 'model_handler')

    def test_instruction_range_validation(self):
        value = config.load_section('data_gen', 'instruction_defaults', 'pt')
        value['num_words_lower_limit'] = value['num_words_upper_limit'] + 1
        with patch.object(config, '_read', return_value={'instruction_defaults': {'pt': value}}), self.assertRaises(ValueError):
            config.load_section('data_gen', 'instruction_defaults', 'pt')

    def test_metrics_section_validation(self):
        cases = [('calc_eval', 'score_decimals', 1.5), ('paths', 'report_output_dir', False), ('nlp', 'portuguese_disable', 'parser')]
        for section, key, invalid in cases:
            value = config.load_section('metrics', section)
            value[key] = invalid
            with self.subTest(section=section), patch.object(config, '_read', return_value={section: value}), self.assertRaises(ValueError):
                config.load_section('metrics', section)
        value = config.load_section('metrics', 'evaluation_main')
        value['loose_filename'] = value['strict_filename']
        with patch.object(config, '_read', return_value={'evaluation_main': value}), self.assertRaises(ValueError):
            config.load_section('metrics', 'evaluation_main')

    def test_three_files_and_canonical_paths(self):
        self.assertEqual({p.name for p in ROOT.glob('*.yml')}, {'data_gen.yml', 'inference.yml', 'metrics.yml'})
        inference = config.load_config('inference')
        self.assertEqual(inference['io'], config.load_config('metrics')['io'])
        for lang, expected in [('pt', 535), ('pten', 541)]:
            path = config.project_path(inference['io']['input_files'][lang])
            rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({r['prompt'] for r in rows}), expected)

    def test_isolation_and_cache(self):
        first = config.load_config('inference')
        first['io']['input_files']['pt'] = 'mutated'
        self.assertNotEqual(first, config.load_config('inference'))
        self.assertGreater(config._read.cache_info().hits, 0)

    def test_config_cli_precedence(self):
        with patch.dict(os.environ, {'IFEVAL_INFERENCE_CONFIG': 'environment.yml'}):
            with patch.object(sys, 'argv', ['universal_inference.py', '--config', 'cli.yml']):
                self.assertEqual(config.config_path('inference'), Path('cli.yml').resolve())
                self.assertEqual(config.config_path('metrics').name, 'metrics.yml')
            with patch.object(sys, 'argv', ['benchmark_runner.py', '--metrics-config=report.yml']):
                self.assertEqual(config.config_path('metrics'), Path('report.yml').resolve())

    def test_duplicates_bad_types_and_io_contract(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            yaml.load('a: 1\na: 2', Loader=config.UniqueKeyLoader)
        with self.assertRaises(ValueError):
            config.load_config('unknown', ROOT / 'inference.yml')
        for key, invalid in [('batch_size', 0), ('max_new_tokens', True), ('gpu_memory_utilization', 2)]:
            data = config.load_config('inference')
            data['universal_inference'][key] = invalid
            with patch.object(config, '_read', return_value=data), self.assertRaises(ValueError):
                config.load_config('inference')
        data = config.load_config('inference')
        data['io']['response_template'] = '{dataset}_different_{model}.jsonl'
        with patch.object(config, '_read', return_value=data), self.assertRaises(ValueError):
            config.load_config('inference')

    def test_metrics_explicit_source(self):
        data = config.load_config('metrics')
        expected = data.pop('io')
        data['source'] = {'mode': 'explicit', 'io': expected}
        with patch.object(config, '_read', return_value=data):
            self.assertEqual(config.load_config('metrics')['io'], expected)

    def test_original_automatic_context_mode(self):
        for length in (0, -1):
            data = config.load_config('inference')
            data['universal_inference']['max_model_len'] = length
            with patch.object(config, '_read', return_value=data):
                self.assertEqual(config.load_config('inference')['universal_inference']['max_model_len'], length)

    def test_alternative_config_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            settings = config.load_config('inference')
            settings['io']['responses_dir'] = 'experiments/custom_responses'
            settings['io']['response_marker'] = '_answers_'
            settings['io']['response_template'] = '{dataset}_answers_{model}.jsonl'
            settings['universal_inference']['max_model_len'] = 0
            selected = folder / 'custom.yml'
            selected.write_text(yaml.safe_dump(settings), encoding='utf-8')
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'universal_inference.py'), '--config', str(selected), '--model_name', 'test', '--datasets', 'pt', 'pten', '--dry-run'], cwd=folder, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('custom_responses', result.stdout)
            self.assertIn('535 prompts', result.stdout)
            self.assertIn('541 prompts', result.stdout)
            metrics = config.load_config('metrics')
            metrics.pop('io')
            metrics['source']['config'] = selected.name
            report = folder / 'metrics.yml'
            report.write_text(yaml.safe_dump(metrics), encoding='utf-8')
            self.assertEqual(config.load_config('metrics', report)['io'], settings['io'])
            records = [json.loads(line) for line in config.project_path(settings['io']['input_files']['pt']).read_text(encoding='utf-8').splitlines()]
            (folder / 'pt_answers_test.jsonl').write_text('\n'.join(json.dumps({'prompt': row['prompt'], 'response': 'Teste'}) for row in records), encoding='utf-8')
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'run_eval_only.py'), '--config', str(ROOT / 'metrics.yml'), '--inference-config', str(selected), '--responses-dir', str(folder), '--evaluations-dir', str(folder / 'scores'), '--languages', 'pt', '--dry-run'], cwd=folder, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Evaluated files: 1/1', result.stdout)

    def test_scientific_checkers_are_unchanged(self):
        for path in (ROOT / 'instructions').glob('*_instructions.py'):
            dumps = []
            for source in [original(path.relative_to(ROOT).as_posix()), path.read_text(encoding='utf-8')]:
                methods = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == 'check_following']
                dumps.append([ast.dump(node, include_attributes=False) for node in methods])
            self.assertEqual(*dumps, path.name)

    def test_registry_compatibility_matches_original(self):
        tree = ast.parse((ROOT / 'instructions_registry.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        # Resolve registry keys without importing NLP libraries or loading models.
        stub = type('InstructionModule', (), {'__getattr__': lambda self, name: None})()
        namespace = {lang + '_instructions': stub for lang in ('en', 'es', 'fr', 'ja', 'pt')}
        exec(compile(tree, '<registry>', 'exec'), namespace)
        keys = namespace['INSTRUCTION_DICT'].keys()
        for lang, relative in config.load_config('inference')['io']['input_files'].items():
            findings = []
            for source in [original(relative), (ROOT / relative).read_text(encoding='utf-8')]:
                rows = [json.loads(line) for line in source.splitlines() if line.strip()]
                unknown = {key for row in rows for key in row['instruction_id_list']} - keys
                affected = sum(any(key in unknown for key in row['instruction_id_list']) for row in rows)
                findings.append((unknown, affected))
            self.assertEqual(*findings, lang)
            if lang == 'pt':
                self.assertEqual(findings[0], (set(), 0))
            else:
                self.assertEqual((len(findings[0][0]), findings[0][1]), (8, 251))

    def test_benchmark_propagates_paths(self):
        import benchmark_runner as runner
        command = runner.build_evaluation_cmd(Path('responses'), Path('scores'), ['pt'], True)
        self.assertEqual(command[command.index('--pt-input-data') + 1], str(config.project_path(runner.IO['input_files']['pt'])))
        self.assertIn('--config', command)
        self.assertIn('--inference-config', command)
        self.assertIn('--dry-run', command)

    def test_all_constant_config_references_exist(self):
        paths = list(ROOT.glob('*.py')) + list((ROOT / 'instructions').glob('*.py')) + list((ROOT / 'instruction_utils').glob('*.py'))
        for path in paths:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            namespace = {'load_config': config.load_config, 'load_section': config.load_section}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {'CONFIG', 'CFG', 'IO', 'NLP_CONFIG'}:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in namespace and isinstance(node.slice, ast.Constant):
                    with self.subTest(file=path.name, key=node.slice.value):
                        self.assertIn(node.slice.value, namespace[node.value.id])

    def test_instruction_constants_preserve_values_and_types(self):
        for path in list((ROOT / 'instructions').glob('*_instructions.py')) + list((ROOT / 'instruction_utils').glob('*_instructions_util.py')):
            lang = path.name.split('_')[0]
            settings = config.load_config('data_gen')['instruction_defaults'][lang]
            before = ast.parse(original(path.relative_to(ROOT).as_posix()))
            after = ast.parse(path.read_text(encoding='utf-8'))
            old_values = {}
            for node in before.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        old_values[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
            for node in after.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in old_values and 'CFG[' in ast.unparse(node.value):
                        current = eval(compile(ast.Expression(node.value), str(path), 'eval'), {'CFG': settings})
                        self.assertEqual(current, old_values[name], f'{path.name}:{name}')
                        self.assertIs(type(current), type(old_values[name]), f'{path.name}:{name}')


class DifferentialTests(unittest.TestCase):
    def test_configured_retries_and_exhaustion(self):
        class BadRequest(Exception): pass
        fake = types.SimpleNamespace(OpenAI=None, AsyncOpenAI=None, BadRequestError=BadRequest)
        with patch.dict(sys.modules, {'openai': fake}):
            obj = module((ROOT / 'model_handler.py').read_text(encoding='utf-8'), 'retry_test', ROOT / 'model_handler.py')
        with patch.object(obj, 'chat_call', side_effect=RuntimeError('failure')) as call, patch.object(obj.time, 'sleep') as sleep:
            self.assertIsNone(obj.safe_chat_call([], 'model'))
            self.assertEqual(call.call_count, 1)
            sleep.assert_not_called()
        obj.CFG['retry_attempts'] = 3
        with patch.object(obj, 'chat_call', side_effect=[RuntimeError('temporary'), 'recovered']) as call, patch.object(obj.time, 'sleep') as sleep:
            self.assertEqual(obj.safe_chat_call([], 'model'), 'recovered')
            self.assertEqual(call.call_count, 2)
            sleep.assert_called_once_with(obj.CFG['retry_wait_seconds'])
        with patch.object(obj, 'chat_call', side_effect=RuntimeError('failure')) as call, patch.object(obj.time, 'sleep') as sleep:
            self.assertIsNone(obj.safe_chat_call([], 'model'))
            self.assertEqual(call.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_api_response_generators(self):
        for kind in ['OpenaiResponseGenerator', 'AnthropicResponseGenerator', 'VllmResponseGenerator']:
            results = []
            for source in [original('get_responses.py'), (ROOT / 'get_responses.py').read_text(encoding='utf-8')]:
                calls = []
                response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='Resposta'))], content=[types.SimpleNamespace(text='Resposta')])
                def create(**kwargs):
                    calls.append(('create', kwargs))
                    return response
                def client(**kwargs):
                    calls.append(('client', kwargs))
                    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)), messages=types.SimpleNamespace(create=create))
                class LLM:
                    def __init__(self, **kwargs): calls.append(('engine', kwargs))
                    def chat(self, messages, **kwargs):
                        calls.append(('chat', messages, kwargs))
                        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='Resposta')]) for _ in messages]
                fake = {'datasets': types.SimpleNamespace(load_dataset=None), 'openai': types.SimpleNamespace(OpenAI=client), 'anthropic': types.SimpleNamespace(Anthropic=client), 'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **k: k), 'tqdm.auto': types.SimpleNamespace(tqdm=lambda seq: seq)}
                with patch.dict(sys.modules, fake), patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY':'test-key'}):
                    obj = module(source, 'responses_test', ROOT / 'get_responses.py')
                    output = getattr(obj, kind)('test/model').get_response(['Primeiro', 'Segundo'])
                    results.append((calls, output))
            self.assertEqual(*results, kind)

    def test_model_handler_requests_and_fallback(self):
        results = []
        for source in [original('model_handler.py'), (ROOT / 'model_handler.py').read_text(encoding='utf-8')]:
            calls = []
            fail_next = [False]
            class BadRequest(Exception): pass
            def create(**kwargs):
                calls.append(('request', copy.deepcopy(kwargs)))
                if fail_next[0]:
                    fail_next[0] = False
                    raise BadRequest('simulated fallback')
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='Resposta'))])
            def client(**kwargs):
                calls.append(('client', kwargs))
                return types.SimpleNamespace(models=types.SimpleNamespace(list=lambda: types.SimpleNamespace(data=[types.SimpleNamespace(id='test-model')])), chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
            def retry(**kwargs):
                return lambda f: f
            fake = {'openai': types.SimpleNamespace(OpenAI=client, AsyncOpenAI=client, BadRequestError=BadRequest), 'tenacity': types.SimpleNamespace(retry=retry, stop_after_attempt=lambda x: x, wait_fixed=lambda x: x)}
            with patch.dict(sys.modules, fake), contextlib.redirect_stdout(io.StringIO()):
                obj = module(source, 'model_test', ROOT / 'model_handler.py')
                model = obj.get_model_id()
                self.assertEqual(obj.safe_chat_call([{'role':'system', 'content':'Teste'}], model), 'Resposta')
                fail_next[0] = True
                self.assertEqual(obj.safe_chat_call([{'role':'system', 'content':'Fallback'}], model), 'Resposta')
            results.append(calls)
        self.assertEqual(*results)

    def test_score_calculator(self):
        results = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eval.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in [
                {'prompt':'A — B', 'follow_all_instructions': True},
                {'prompt':'C', 'follow_all_instructions': False},
                {'prompt':'excluded', 'follow_all_instructions': True},
            ]), encoding='utf-8')
            for source in [original('calc_eval.py'), (ROOT / 'calc_eval.py').read_text(encoding='utf-8')]:
                obj = module(source, 'calc_test', ROOT / 'calc_eval.py')
                scores = [obj.calculate_accuracy(path), obj.calculate_accuracy(path, {'A - B','C'})]
                results.append([(s.passed, s.total, obj.format_score(s)) for s in scores])
        self.assertEqual(*results)

    def test_clean_response(self):
        sources = [original('universal_inference.py'), (ROOT / 'universal_inference.py').read_text(encoding='utf-8')]
        functions = []
        for source in sources:
            tree = ast.parse(source)
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clean_response')
            namespace = {'re': re, 'CFG': config.load_config('inference')['universal_inference']}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<clean>', 'exec'), namespace)
            functions.append(namespace['clean_response'])
        samples = ['', None, '<think>raciocinio</think>Uma resposta final.', '<think>unfinished', 'answer<think>unfinished', 'e\nva\nUma resposta</s>', 'Let me think\n\nResposta em portugues.', '1. **Analyze the request** sem resposta']
        # Include every archived closed-model response, not just synthetic cases.
        for path in (ROOT / 'experiments/data_close').glob('pt*_input_response*.jsonl'):
            samples.extend(json.loads(line).get('response') for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        for index, sample in enumerate(samples):
            self.assertEqual(functions[0](sample, []), functions[1](sample, []), f'response {index}')

    def test_inference_backends_and_payloads(self):
        for backend in ['transformers', 'vllm', 'fallback']:
            for model_name in ['Qwen/Qwen3-8B', 'Qwen/Qwen3.5-4B', 'other/model']:
                results = []
                with tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory)
                    (folder / 'pt_input_data.jsonl').write_text('{"prompt":"Um teste"}\n', encoding='utf-8')
                    for source in [original('universal_inference.py'), (ROOT / 'universal_inference.py').read_text(encoding='utf-8')]:
                        calls = []
                        class Dataset(list):
                            column_names = ['prompt']
                            def add_column(self, name, values):
                                return Dataset([{**r, name: value} for r, value in zip(self, values)])
                            def select_columns(self, names):
                                return self
                            def to_json(self, path, **kwargs):
                                calls.append(('output', Path(path).name, list(self), kwargs))
                        class Tokenizer:
                            pad_token_id = 2
                            eos_token_id = 2
                            chat_template = 'template'
                            model_max_length = 4096
                            def apply_chat_template(self, messages, **kwargs):
                                calls.append(('template', messages, kwargs))
                                return 'formatted prompt'
                        def sampling(**kwargs):
                            calls.append(('sampling', kwargs))
                            return kwargs
                        class LLM:
                            def __init__(self, **kwargs):
                                calls.append(('engine', kwargs))
                                if backend == 'fallback':
                                    raise RuntimeError('simulated backend failure')
                            def generate(self, prompts, params):
                                calls.append(('generate', prompts, params))
                                return [types.SimpleNamespace(request_id='0', outputs=[types.SimpleNamespace(text='<think>hidden</think>Uma resposta de teste.')])]
                        def pipeline(*args, **kwargs):
                            calls.append(('pipeline', args, {k: v for k,v in kwargs.items() if k != 'tokenizer'}))
                            def generate(prompts, **options):
                                options['generation_config'] = vars(options['generation_config'])
                                calls.append(('hf_generate', prompts, options))
                                return [[{'generated_text':'<think>hidden</think>Uma resposta de teste.'}]]
                            return generate
                        transformer = types.SimpleNamespace(GenerationConfig=types.SimpleNamespace, AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()), pipeline=pipeline)
                        fake = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None), bfloat16='bfloat16'), 'datasets': types.SimpleNamespace(load_dataset=lambda *a, **k: Dataset([{'prompt':'Um teste'}])), 'transformers': transformer, 'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=sampling)}
                        with patch.dict(sys.modules, fake), patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
                            worker = module(source, 'inference_test', ROOT / 'universal_inference.py')
                            worker.VLLM_AVAILABLE = backend != 'transformers'
                            worker.run_multilang_inference(model_name, .9, 8096, folder, folder, False, ['pt'], 32768, 4)
                        self.assertTrue(any(c[0] == 'output' for c in calls))
                        results.append(calls)
                self.assertEqual(results[0], results[1], (backend, model_name))

        failing_modules = dict(fake)
        failing_modules['datasets'] = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError('simulated dataset failure')
            )
        )
        source = (ROOT / 'universal_inference.py').read_text(encoding='utf-8')
        with patch.dict(sys.modules, failing_modules), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            worker = module(source, 'inference_failure_test', ROOT / 'universal_inference.py')
            worker.VLLM_AVAILABLE = False
            with self.assertRaisesRegex(RuntimeError, 'pt'):
                worker.run_multilang_inference('test-model', .9, 8096, folder, folder, False, ['pt'], 16, 1)

    def test_inference_uses_string_paths_and_propagates_dataset_failures(self):
        source = (ROOT / 'universal_inference.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        load_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == 'load_dataset']
        self.assertEqual(len(load_calls), 1)
        data_files = next(keyword.value for keyword in load_calls[0].keywords
                          if keyword.arg == 'data_files')
        train_path = data_files.values[0]
        self.assertIsInstance(train_path, ast.Call)
        self.assertIsInstance(train_path.func, ast.Attribute)
        self.assertEqual((train_path.func.value.id, train_path.func.attr), ('os', 'fspath'))
        self.assertIn('if failures:', source)
        self.assertIn('raise RuntimeError(', source)

    def test_candidate_generation_records_and_api_messages(self):
        results = []
        class Instruction:
            def __init__(self, key): self.key = key
            def build_description(self): return 'Instrucao ' + self.key
            def get_instruction_args(self): return {'value': 1}
        for source in [original('gen_input_data.py'), (ROOT / 'gen_input_data.py').read_text(encoding='utf-8')]:
            calls, records = [], []
            def chat(messages, model, style):
                calls.append((copy.deepcopy(messages), model, style))
                return 'Possivel'
            fake = {'instructions_registry': types.SimpleNamespace(INSTRUCTION_DICT={f'pt:test:{i}': Instruction for i in range(3)}), 'model_handler': types.SimpleNamespace(safe_chat_call=chat, get_model_id=lambda: 'test-model')}
            tree = ast.parse(source)
            # Capture records instead of writing the default candidate output.
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'write_file')]
            namespace = {'__name__': '__main__', '__file__': str(ROOT / 'gen_input_data.py'), 'write_file': lambda row: records.append(copy.deepcopy(row))}
            with patch.dict(sys.modules, fake), patch('secrets.choice', side_effect=lambda seq: seq[0]), patch('random.sample', side_effect=lambda seq, k: seq[:k]), patch.object(sys, 'argv', ['gen_input_data.py']), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(tree, '<data-gen>', 'exec'), namespace)
            self.assertEqual(len(records), 3)
            results.append((calls, records))
        self.assertEqual(*results)


if __name__ == '__main__':
    unittest.main()
