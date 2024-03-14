"""
@author nghiatc
@since 15/03/2024
"""


import contextlib
import random
import numpy as np
import torch
from gemma import config
from gemma import model as gemma_model
from config_gpt import GPTConfig
from logging import getLogger
from typing import Iterable
from sanic import Sanic
from sanic.response import json

logger = getLogger(__name__)


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


# Construct the model config.
gpt_config = GPTConfig()
model_config = config.get_model_config(gpt_config.variant)
model_config.dtype = "float32" if gpt_config.device == "cpu" else "float16"
model_config.quant = gpt_config.quant

# Seed random.
random.seed(gpt_config.seed)
np.random.seed(gpt_config.seed)
torch.manual_seed(gpt_config.seed)

# Create the model and load the weights.
device = torch.device(gpt_config.device)
with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma_model.GemmaForCausalLM(model_config)
    model.load_weights(gpt_config.ckpt)
    model = model.to(device).eval()
print("=======>>>>>>> Model loading done")

# Generate the response.
# result = model.generate(args.prompt, device, output_len=args.output_len)


# Config for Web Server Sanic
port = 24315
NDEBUG = True


def _add_cors_headers(response, methods: Iterable[str]) -> None:
    allow_methods = list(set(methods))
    if "OPTIONS" not in allow_methods:
        allow_methods.append("OPTIONS")
    headers = {
        "Access-Control-Allow-Methods": ",".join(allow_methods),
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Headers": (
            "origin, content-type, accept, "
            "authorization, x-xsrf-token, x-request-id"
        ),
    }
    response.headers.extend(headers)


# https://sanic.dev/en/guide/how-to/cors.html#server.py
def add_cors_headers(request, response):
    # if request.method != "OPTIONS":
    #     methods = [method for method in request.route.methods]
    #     _add_cors_headers(response, methods)
    methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    _add_cors_headers(response, methods)


# Web Server Sanic Version mới nhất hiện tại == 23.12.1
# Những phiên bản Sanic mới nhất yêu cầu Python versions 3.8 – 3.11
def start():
    try:
        # Start Web Sever Sanic
        app = Sanic("GemmaGPT")

        # Add Router
        app.add_route(chat_gemma_gpt, '/api/v1/chat', methods=["POST", "OPTIONS"], )

        # Route some file and client resources
        # https://sanic.dev/en/guide/how-to/static-redirects.html
        app.static('/files/', 'files')
        app.static('/', 'views')

        # Fill in CORS headers
        app.register_middleware(add_cors_headers, "response")
        # port = get_port()
        print(f'=====>>>>> GemmaGPT Sanic is running on port={port} with mode NDEBUG={NDEBUG}...')
        logger.info(f'=====>>>>> GemmaGPT Sanic is running on port={port} with mode NDEBUG={NDEBUG}...')
        app.run(port=port, debug=NDEBUG)
    except KeyboardInterrupt:
        print("xxxxxxxxxxxxxxxxxx WebServer Event KeyboardInterrupt")
    finally:
        print("~~~~~~~~~~~~~~~~ WebServer Exit")


def chat_gemma_gpt(req):
    try:
        data = req.json
        # print(data)
        prompt = data['prompt'] if 'prompt' in data else ""
        # print(prompt)
        if not prompt:
            return json({'err': -1, 'msg': 'Parameters invalid'})
        output_len = data['output_len'] if 'output_len' in data else 50
        # print(output_len)

        # Generate the response.
        result = model.generate(prompt, device, output_len=output_len)

        return json({
            'err': 0,
            'msg': 'success',
            'prompt': prompt,
            'output_len': output_len,
            'result': result
        })
    except Exception as e:
        logger.error(f'XXXXX GemmaGPT Handler chat_gemma_gpt has error={e}')
