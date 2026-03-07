#!/usr/bin/env python
import os
from pathlib import Path
import sys

import pytest

path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset

server = None


@pytest.fixture(autouse=True)
def create_server():
    global server
    model_file = os.getenv("LLAMA_QWEN3_REASONING_MODEL_FILE", "").strip()
    if not model_file:
        pytest.skip("LLAMA_QWEN3_REASONING_MODEL_FILE is not configured")

    server = ServerPreset.tinyllama2()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = model_file
    server.model_alias = os.getenv("LLAMA_QWEN3_REASONING_MODEL_ALIAS", "qwen3-reasoning")
    server.server_port = int(os.getenv("LLAMA_QWEN3_REASONING_PORT", "8082"))
    server.n_slots = 1
    server.n_ctx = int(os.getenv("LLAMA_QWEN3_REASONING_CTX", "16384"))
    server.n_predict = 512
    server.jinja = True
    server.chat_template_file = "../../../models/templates/Qwen-Qwen3-0.6B.jinja"
    server.reasoning_format = "deepseek"
    server.start(timeout_seconds=15 * 60)


@pytest.mark.slow
@pytest.mark.parametrize("stream", [False, True])
def test_qwen3_reasoning_coding_request_does_not_500(stream: bool):
    global server
    response = server.make_any_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias,
        "stream": stream,
        "max_tokens": 256,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "messages": [
            {
                "role": "user",
                "content": "Review this Java method and provide the smallest production-safe null handling fix: public String url(Config c){return c.getDb().getUrl();}",
            }
        ],
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
    })

    choice = response["choices"][0]
    message = choice["message"]
    assert message.get("tool_calls") in (None, []), message
    assert message.get("content") or message.get("reasoning_content"), message
