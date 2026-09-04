
from config import load_config, project_path
CONFIG = load_config('inference')
CFG = CONFIG['get_responses']
IO = CONFIG["io"]

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
            api_key=os.environ[CFG['anthropic_key_env']],
        )
        self.model_name = model_name
    
    def get_response(self, input_texts):
        return [
            self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=CFG['anthropic_max_tokens'],
                temperature=CFG['anthropic_temperature'],
                messages=[
                    {
                        "role": CFG['message_role'],
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

        self.openai_client = OpenAI(api_key=os.environ[CFG['openai_key_env']])
        self.model_name = model_name
    
    def get_single_response(self, input_text):
        try:
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": CFG['message_role'],
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
        self.llm = LLM(model=self.model_name, max_model_len=os.environ.get(CFG['max_model_len_env'], CFG['vllm_max_model_len']))
        self.sampling_params = SamplingParams(temperature=CFG['vllm_temperature'], max_tokens=CFG['vllm_max_tokens'])

    def get_response(self, input_texts):
        input_conversations = [[{
            "role": CFG['message_role'],
            "content": input_text
        }] for input_text in input_texts]

        outputs = self.llm.chat(input_conversations,
                   sampling_params=self.sampling_params,
                   use_tqdm=CFG['use_progress_bar'])
        return [output.outputs[0].text for output in outputs]

######## Main ########

SUPPORTED_MODELS = CFG['supported_models']

MODEL_CLASS_DICT = {
    "openai": OpenaiResponseGenerator,
    "anthropic": AnthropicResponseGenerator,
    # "gemini": VertexResponseGenerator,
    "vllm": VllmResponseGenerator,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Stage YAML configuration file.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_dir", default=project_path(IO["data_dir"]))
    parser.add_argument("--output_dir", default=project_path(IO["responses_dir"]))
    args = parser.parse_args()

    model_name = args.model_name

    assert model_name in SUPPORTED_MODELS, f"Model {model_name} not supported; update get_responses.supported_models in inference.yml."

    paths = sorted(glob(os.path.join(args.input_dir, CFG["input_glob"])))
    os.makedirs(args.output_dir, exist_ok=True)

    model_class = MODEL_CLASS_DICT[SUPPORTED_MODELS[model_name]]
    response_generator = model_class(model_name)

    for path in paths:
        print(path + " - " + model_name)
        ds = load_dataset("json", data_files={"train": path}, split="train")
        ds = ds.add_column("response", response_generator.get_response(ds["prompt"]))
        input_name = os.path.basename(path)
        output_name = IO["response_template"].format(
            dataset=input_name[:-len(IO["input_template"].format(dataset=""))],
            model=model_name.replace("/", "__"),
        )
        ds.select_columns(["prompt", "response"]).to_json(
            os.path.join(args.output_dir, output_name)
        )
