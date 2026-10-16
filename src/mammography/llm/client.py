from functools import lru_cache

from openai import OpenAI

from mammography.config import get_settings


@lru_cache
def get_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)
