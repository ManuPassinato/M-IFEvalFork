
from config import load_section
CFG = load_section('data_gen', 'model_handler')

import os
import time
from typing import Dict, List, Optional
from openai import AsyncOpenAI, BadRequestError, OpenAI

VLLM_BASE_URL = os.environ.get(CFG['vllm_base_url_env'], CFG['vllm_base_url_default'])
VLLM_API_KEY  = os.environ.get(CFG['vllm_api_key_env'], CFG['vllm_api_key_default'])

RESP_TEMPERATURE = float(os.environ.get(CFG['resp_temperature_env'], str(CFG['resp_temperature_default'])))
RESP_TOP_P       = float(os.environ.get(CFG['resp_top_p_env'], str(CFG['resp_top_p_default'])))
RESP_MAX_TOKENS  = int(os.environ.get(CFG['resp_max_tokens_env'], CFG['resp_max_tokens_default']))

STOP_STRINGS      = CFG['stop_strings']
STOP_TOKEN_IDS    = CFG['stop_token_ids']
LOGITS_PROCESSORS: List[str] = CFG['logits_processors']

from functools import lru_cache

@lru_cache(maxsize=1)
def get_client():
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

def get_model_id() -> str:
    models = get_client().models.list()
    if not models.data:
        raise RuntimeError("Nenhum modelo disponível no endpoint vLLM.")
    print(models.data[0].id)
    return models.data[0].id


def chat_call(
    messages: List[Dict[str, str]],
    model_id: str,
    question_style: Optional[str] = None,
    extra_body_override: Optional[dict] = None
) -> str:
    final_messages = messages
    temperature = RESP_TEMPERATURE
    top_p       = RESP_TOP_P
    max_tokens  = RESP_MAX_TOKENS

    extra_body = {
        "chat_template_kwargs": {"enable_thinking": CFG['enable_thinking']},
        "stop": STOP_STRINGS,
        "stop_token_ids": STOP_TOKEN_IDS,
        "logits_processors": LOGITS_PROCESSORS,
    }

    if extra_body_override:
        # sobrescreve/mescla campos problemáticos quando necessário (fallback)
        for k, v in extra_body_override.items():
            extra_body[k] = v

    resp = get_client().chat.completions.create(
        model=model_id,
        messages=final_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""

def safe_chat_call(
    messages: List[Dict[str, str]],
    model_id: str,
    question_style: Optional[str] = None) -> Optional[str]:
    """Try a request plus its BadRequest fallback; retry only when configured."""
    for attempt in range(CFG['retry_attempts']):
        try:
            try:
                return chat_call(messages, model_id, question_style)
            except BadRequestError:
                eb2 = {"chat_template_kwargs": {"enable_thinking": CFG['fallback_enable_thinking']}}
                return chat_call(messages, model_id, question_style, extra_body_override=eb2)
        except Exception:
            if attempt + 1 < CFG['retry_attempts']:
                time.sleep(CFG['retry_wait_seconds'])
    return None
