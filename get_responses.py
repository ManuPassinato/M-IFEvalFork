# coding=utf-8
# Copyright 2025 The Lightblue Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from glob import glob
from tqdm.auto import tqdm
from datasets import load_dataset

class ResponseGenerator:
    def __init__(self, model_name):
        raise NotImplementedError
    
    def get_response(self, input_texts):
        raise NotImplementedError

######## Anthropic ########

class AnthropicResponseGenerator(ResponseGenerator):

    def __init__(self, model_name):
        import anthropic
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        self.model_name = model_name
    
    def get_response(self, input_texts):
        return [
            self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": input_text
                            }
                        ]
                    }
                ]
            ).content[0].text for input_text in tqdm(input_texts)
        ]

######## OpenAI ########

class OpenaiResponseGenerator(ResponseGenerator):
    def __init__(self, model_name):
        from openai import OpenAI

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name
    
    def get_single_response(self, input_text):
        try:
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": input_text
                        }
                    ]
                    }
                ],
                # temperature=0,
                # # max_tokens=None if "o1" in self.model_name else 2048,
                # # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0,
                # response_format={"type": "text"}
            ).choices[0].message.content
        except Exception as e:
            print(e)
            return None
    
    def get_response(self, input_texts):
        return [
            self.get_single_response(input_text) for input_text in tqdm(input_texts)
        ]

######## VertexAI ########

# TO DO: Add Support for VertexAI
# class VertexResponseGenerator(ResponseGenerator):
#     def __init__(self, model_name):
#         self.model_name = model_name
    
#     def get_response(self, input_texts):
#         import vertexai
#         from vertexai.generative_models import GenerativeModel

#         generation_config = {
#             "max_output_tokens": 2048,
#             "temperature": 0,
#         }

#         safety_settings = [
#         ]

#         vertexai.init(project="dev-llab", location="asia-south1")
#         model = GenerativeModel(
#             self.model_name,
#         )

#         def get_vertex_response(input_text):
#             chat = model.start_chat(response_validation=False)

#             return chat.send_message(
#                 [input_text],
#                 generation_config=generation_config,
#                 safety_settings=safety_settings
#             ).candidates[0].content.parts[0].text

#         return [get_vertex_response(input_text) for input_text in tqdm(input_texts)]
        


######## vLLM ########

class VllmResponseGenerator(ResponseGenerator):
    def __init__(self, model_name):
        from vllm import LLM, SamplingParams
        self.model_name = model_name
        self.llm = LLM(model=self.model_name, max_model_len=os.environ.get("MAX_MODEL_LEN", 4096))
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    def get_response(self, input_texts):
        input_conversations = [[{
            "role": "user",
            "content": input_text
        }] for input_text in input_texts]

        outputs = self.llm.chat(input_conversations,
                   sampling_params=self.sampling_params,
                   use_tqdm=True)
        return [output.outputs[0].text for output in outputs]

######## Main ########

SUPPORTED_MODELS = {
    'gpt-4o-mini-2024-07-18': 'openai',
    'gpt-4o-2024-08-06': 'openai',
    'o1-preview-2024-09-12': 'openai',
    'o1-mini-2024-09-12': 'openai',
    'claude-3-haiku-20240307': 'anthropic',
    'claude-3-5-sonnet-20240620': 'anthropic',
    'claude-3-opus-20240229': 'anthropic',
    # 'gemini-1.5-pro-002': 'gemini',
    # 'gemini-1.5-flash-002': 'gemini',
    'CohereForAI/c4ai-command-r-plus-4bit': 'vllm',
    'CohereForAI/c4ai-command-r-v01-4bit': 'vllm',
    'CohereForAI/aya-23-8B': 'vllm',
    'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4': 'vllm',
    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4': 'vllm',
    'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4': 'vllm',
    'mistralai/Mistral-7B-Instruct-v0.3': 'vllm',
    'deepseek-ai/deepseek-llm-7b-chat': 'vllm'
}

MODEL_CLASS_DICT = {
    "openai": OpenaiResponseGenerator,
    "anthropic": AnthropicResponseGenerator,
    # "gemini": VertexResponseGenerator,
    "vllm": VllmResponseGenerator,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_dir", default="./data")
    parser.add_argument("--output_dir", default="./experiments/generated_responses")
    args = parser.parse_args()

    model_name = args.model_name

    assert model_name in SUPPORTED_MODELS, f"Model {model_name} not supported, update SUPPORTED_MODELS dictionary in get_responses.py to support it."

    paths = sorted(glob(os.path.join(args.input_dir, "*_input_data.jsonl")))
    os.makedirs(args.output_dir, exist_ok=True)

    model_class = MODEL_CLASS_DICT[SUPPORTED_MODELS[model_name]]
    response_generator = model_class(model_name)

    for path in paths:
        print(path + " - " + model_name)
        ds = load_dataset("json", data_files={"train": path}, split="train")
        ds = ds.add_column("response", response_generator.get_response(ds["prompt"]))
        input_name = os.path.basename(path)
        output_name = (
            input_name[:-10]
            + "response_data_"
            + model_name.replace("/", "__")
            + ".jsonl"
        )
        ds.select_columns(["prompt", "response"]).to_json(
            os.path.join(args.output_dir, output_name)
        )
