# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenTeleEval Authors.

import time

import requests


class OpenAIJudge:
    """OpenAI-compatible API judge model.

    Args:
        base_url (str): Base URL of the OpenAI-compatible service,
            e.g. ``http://host:port/v1``.
        model (str): Model name served by the endpoint.
        api_key (str): API key; ignored by most local vLLM servers.
        retries (int): Number of retries on request failure.
    """

    def __init__(self,
                 base_url: str,
                 model: str,
                 api_key: str = 'None',
                 retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.retries = retries
        self.timeout = 15 * 60
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

    def chat(self, message, stream=False, detail=False):
        if isinstance(message, list):
            messages = message
        else:
            messages = [{'content': message, 'role': 'user'}]

        payload = {
            'model': self.model,
            'stream': False,
            'max_tokens': 4096,
            'temperature': 0,
            'messages': messages,
        }
        url = f'{self.base_url}/chat/completions'
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    url, headers=self.headers, json=payload,
                    timeout=self.timeout)
                response.raise_for_status()
                return response.json()['choices'][-1]['message']['content']
            except Exception as e:
                print(f'Attempt {attempt + 1} failed: {e}')
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        return 'LLM ERROR'

    def predict(self, input_text, **kwargs):
        return self.chat(message=input_text)

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


def maybe_build_openai_judge(judge_model_cfg):
    """Build an :class:`OpenAIJudge` from a config dict.

    Non-dict configs (e.g. an already-built judge instance) are returned
    as-is; ``None`` returns ``None``.
    """
    if judge_model_cfg is None or not isinstance(judge_model_cfg, dict):
        return judge_model_cfg
    return OpenAIJudge(**judge_model_cfg)
