# M-IFEval: Multilingual Instruction Following Evaluation
<span style="display: inline; gap: 5px;">
<a href="https://github.com/lightblue-tech/M-IFEval/fork"><img src="https://img.shields.io/badge/PRs-Welcome-purple?color=%23b304d6" height="20"/></a>
<a href="https://colab.research.google.com/github/lightblue-tech/M-IFEval/blob/main/colab_mifeval_run.ipynb"><img src="https://img.shields.io/badge/Colab-Demo-gray?logo=googlecolab&color=%23F9AB00" height="20"/></a> 
<a href="https://www.arxiv.org/abs/2502.04688"><img src="https://img.shields.io/badge/ArXiv-Preprint-gray?logo=arxiv&labelColor=%23B31B1B" height="20"/></a>    
</span>   


> This repository contains source code and data for [M-IFEval: Multilingual
Instruction-Following Evaluation](https://www.arxiv.org/abs/2502.04688)


Large language models (LLMs) are increasingly used in real-world applications that require understanding and executing user instructions accurately. Evaluating instruction following is crucial to ensure these models perform reliably across different tasks and languages.

Building upon the [Instruction Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) benchmark, which first underscored the importance of instruction-following evaluation,  we introduce **M-IFEval**,  a benchmark designed to extend this evaluation to multiple languages.

M-IFEval currently supports **French**, **Japanese**, and **Spanish**, incorporating both general and language-specific instructions to provide a more comprehensive assessment of multilingual instruction adherence.

## Table of Contents
- [**🏆 Leader board**](#-leader-board)
- [**⚙️ How to run**](#️-how-to-run)  
  - [🌐 With Colab](#with-colab-)
  - [🖥️ Locally](#locally-desktop_computer)
    - [Setup Instructions](#setup-instructions)
    - [Evaluate YOUR model](#evaluate-your-model)
- [**Contributions 🤝**](#contributions-)
- [**📚 References**](#-reference)
- [**📜 License**](#-license)

## 🏆 Leader board 
The table below presents the average scores across all instructions for each language, sorted in ascending order by the mean scores across the languages supported by the M-IFEval benchmark (i.e., Spanish, French, and Japanese).

| Model Name            | English* | Spanish | French | Japanese | Average (ES/FR/JA) |
|:-----------------------|-----:|-----:|-----:|-----:|--------------:|
| o1                     | 85.9 | 92.7 | 91.3 | 75.7 |          86.6 |
| Opus                   | 87.3 | 90.5 | 87   | 75.7 |          84.4 |
| Sonnet                 | 88.1 | 87.6 | 88.1 | 77   |          84.2 |
| o1 Mini                | 83.9 | 92   | 88.4 | 69.5 |          83.3 |
| GPT4o                  | 88.6 | 89.8 | 87.8 | 70.4 |          82.7 |
| GPT4o Mini             | 86   | 85.4 | 85.5 | 65.9 |          78.9 |
| Qwen 2.5 32B I. 4-bit  | 86   | 82.5 | 81.7 | 65.9 |          76.7 |
| Qwen 2.5 14B I. 4-bit  | 84.2 | 83.2 | 82.6 | 57.5 |          74.4 |
| Haiku                  | 77.3 | 78.8 | 78.3 | 61.9 |          73   |
| Qwen 2.5 7B I. 4-bit   | 80.6 | 78.1 | 76.8 | 50.9 |          68.6 |
| Llama 3.1 8B I.        | 80.1 | 80.3 | 71.3 | 39.8 |          63.8 |
| Qwen 2.5 3B I. 4-bit   | 67.9 | 68.6 | 65.8 | 45.1 |          59.8 |
| Mistral 7B I.          | 59   | 62.8 | 61.7 | 29.2 |          51.2 |
| Aya 23 8B              | 50.6 | 59.9 | 57.4 | 35   |          50.7 |
| Qwen 2.5 1.5B I. 4-bit | 49.5 | 59.9 | 49.3 | 28.3 |          45.8 |
| DeepSeek LLM 7B Chat   | 48.9 | 48.2 | 45.8 | 25.2 |          39.7 |
| Qwen 2.5 0.5B I. 4-bit | 34.4 | 36.5 | 33.9 | 19.9 |          30.1 |

\(*) The values reported for English (EN) correspond to the evaluation on the original IFEval dataset.


## ⚙️ How to run

Benchmark input datasets live in `data/`. Generated responses and evaluation
outputs are centralized in `experiments/`; see `experiments/README.md` for the
current campaign map. The canonical Portuguese benchmark is
`data/pt_input_data.jsonl` and contains 535 prompts.

### With Colab 🌐

We provide a Jupyter Notebook designed to run seamlessly in Google Colab. This notebook guides you through:

- Generating responses from models supported by Hugging Face (HF), OpenAI, and Anthropic.
- Evaluating a given model based on its generated responses.
- Displaying basic visualizations of evaluation results.

Click the badge to open the notebook in Colab:
<a href="https://colab.research.google.com/github/lightblue-tech/M-IFEval/blob/main/colab_mifeval_run.ipynb">
    <img src="https://img.shields.io/badge/Colab-Notebook-gray?logo=googlecolab&color=%23F9AB00" height="20" style="vertical-align: -5px;"/>
</a> 


### Locally :desktop_computer:

If you prefer to run the evaluation directly on your machine, whether for computational reasons or other preferences, follow these steps:

#### Setup Instructions
<hr style="border: 0; height: 1px; box-shadow: 0 0 0 0.01px #ddd; margin-top: 0;"/>

##### Clone the Repository  

First, clone the repository:  

```bash
git clone -b main https://github.com/lightblue-tech/M-IFEval.git
```

Move into the cloned directory:
```bash
cd M-IFEval 
```

##### Install Dependencies  

Ensure all required Python packages are installed in your working environment:  

```bash
pip3 install --user -r requirements.txt
```

Download Spacy models:

```bash
python -m spacy download es_core_news_sm --quiet
python -m spacy download xx_sent_ud_sm --quiet
```

#### Evaluate Your Model  
<hr style="border: 0; height: 1px; box-shadow: 0 0 0 0.01px #ddd; margin-top: 0;"/>

##### Prepare Responses File  

To evaluate your model, you need to create a JSONL file containing the instruction prompts and their corresponding responses.  

You can either manually create a file with the following structure:  

```json
{"prompt": "Write a 300+ word summary ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
{"prompt": "I am planning a trip to ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
```

Or, you can use `get_responses.py` to automatically generate the JSONL file in the required format:

- **For Hugging Face, OpenAI, or Anthropic models**:  

  To add support for a new model, update the `SUPPORTED_MODELS` dictionary in `get_responses.py` by specifying the model name and its corresponding inference method:  

  ```python
  SUPPORTED_MODELS = {
      "openai_model_name_or_version": "openai",
      "anthropic_model_name_or_version": "anthropic",
      "hf_model_name_or_path": "vllm"
  }
  ```  

  Replace `"openai_model_name_or_version"`, `"anthropic_model_name_or_version"`, or `"hf_model_name_or_path"` with the actual model identifiers.  

  **Note:** The dictionary values (`"openai"`, `"anthropic"`, `"vllm"`) must remain unchanged to ensure the correct inference method is used.  

  Once the model is added, generate responses by running:  
  
    ```bash
    python3 get_responses.py --model_name {model_name}
    ```  

  Replace `{model_name}` with the exact model identifier you added.

> [!NOTE] 
> Before running the scripts, make sure to set your API keys and Hugging Face token as environment variables. For OpenAI or Anthropic, set the API key like this:  
>
>```bash
>export OPENAI_API_KEY="your_openai_api_key_here"
>export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
>```
>
>For Hugging Face, set your token with:  
>
>```bash
>export HUGGINGFACE_TOKEN="your_huggingface_token_here"
>huggingface-cli login --token $HUGGINGFACE_TOKEN
>```
>
>Replace the placeholders with your actual keys. This will >ensure the scripts can access the necessary services.

- **For *(yet)* unsupported providers  requiring API keys**:  

  Implement a subclass of `ResponseGenerator` in `get_responses.py` to handle API calls for your provider.  

##### Run the Evaluation  

Once you have the JSONL file ready, run the evaluation script:  

```bash
python3 -m evaluation_main \
  --input_data=./data/{lang}_input_data.jsonl \
  --input_response_data=./experiments/generated_responses/input_response_data_model_name.jsonl \
  --output_dir=./experiments/generated_evaluations/model_name/
```
- Replace `{lang}` with the language tag corresponding to the language you wish to evaluate (e.g., `en` for English, `fr` for French).
- Update `input_response_data` with the path to your model's response JSONL file.

This command will generate evaluation results in the specified output directory.


## Contributions 🤝

We welcome contributions to:
- Support the evaluation of other models
- Add models to the leaderboard
- Expand the benchmark to additional languages
- Improve evaluation quality (data and instructions) for currently supported languages

If you have any feedback, concerns, or inquiries about the benchmark, feel free to [open an issue](https://github.com/lightblue-tech/M-IFEval/issues/new).

## 📚 Reference 

If you use our work, please consider citing our preprint:

```
@article{Dussolle2025MIFEval,
  title={M-IFEval: Multilingual Instruction-Following Evaluation},
  author={Antoine Dussolle and Andrea Cardena Díaz and Shota Sato and Peter Devine},
  year={2025},
  journal={arXiv preprint},
  volume={arXiv:2502.04688},
  eprint={2502.04688},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2502.04688}
}
```

As well, as the original IFEval paper's preprint:
```
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```

## 📜 License 

This project is licensed under the Apache License, Version 2.0.
See the LICENSE file for [details](https://github.com/lightblue-tech/M-IFEval/blob/main/LICENSE.txt).

This project includes code from [Instruction Following Evaluation for Large Language Models](https://github.com/google-research/google-research/tree/master/instruction_following_eval) licensed under Apache 2.0.
