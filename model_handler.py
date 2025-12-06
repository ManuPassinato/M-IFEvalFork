import os
from typing import Dict, List, Optional
from openai import AsyncOpenAI, BadRequestError, OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://10.100.0.111:8020/v1")
VLLM_API_KEY  = os.environ.get("VLLM_API_KEY", "no-key-needed")

RESP_TEMPERATURE = float(os.environ.get("RESP_TEMPERATURE", str(1)))
RESP_TOP_P       = float(os.environ.get("RESP_TOP_P", str(1)))
RESP_MAX_TOKENS  = int(os.environ.get("RESP_MAX_TOKENS", "8192"))

STOP_STRINGS      = ["<|im_end|>", "<|end_of_text|>"]
STOP_TOKEN_IDS    = None
LOGITS_PROCESSORS: List[str] = []

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

def get_model_id() -> str:
    models = client.models.list()
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
        "chat_template_kwargs": {"enable_thinking": False},
        "stop": STOP_STRINGS,
        "stop_token_ids": STOP_TOKEN_IDS,
        "logits_processors": LOGITS_PROCESSORS,
    }

    if extra_body_override:
        # sobrescreve/mescla campos problemáticos quando necessário (fallback)
        for k, v in extra_body_override.items():
            extra_body[k] = v

    resp = client.chat.completions.create(
        model=model_id,
        messages=final_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1))
def safe_chat_call(
    messages: List[Dict[str, str]],
    model_id: str,
    question_style: Optional[str] = None) -> Optional[str]:
    """
    Chama chat_call com 1 fallback específico para BadRequest 'Expected 2 output messages...'.
    Se falhar novamente ou surgir outro erro, retorna None (para descartar a conversa).
    """
    try:
        return chat_call(messages, model_id, question_style)
    except BadRequestError as e:
        msg = str(e)
        try:
            eb2 = {"chat_template_kwargs": {"enable_thinking": True}}
            return chat_call(messages, model_id, question_style, extra_body_override=eb2)
        except Exception:
            return None
    except Exception:
        return None