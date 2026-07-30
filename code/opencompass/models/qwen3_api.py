# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenTeleEval Authors.

from opencompass.registry import MODELS

from .general_api import BaseGeneralApi


@MODELS.register_module()
class Qwen3API(BaseGeneralApi):
    THINKING_CONTROL_MODE = 'prompt_suffix'
    PARSE_REASONING = False
