"""
@author nghiatc
@since 18/03/2024
"""


import random
import numpy as np
import torch
from gemma import config
from gemma import model as gemma_model
from config_gpt import GPTConfig


# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]


class GemmaGPT:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    # Chưa khắc phục được lỗi GemmaGPT bị khởi tạo lại nhiều lần???
    # dẫn đến tốn RAM.
    def __init__(self):
        if not self._initialized:
            # Construct the model config.
            self.gpt_config = GPTConfig()
            self.model_config = config.get_model_config(self.gpt_config.variant)
            self.model_config.dtype = "float32" if self.gpt_config.device == "cpu" else "float16"
            self.model_config.quant = self.gpt_config.quant

            # Seed random.
            random.seed(self.gpt_config.seed)
            np.random.seed(self.gpt_config.seed)
            torch.manual_seed(self.gpt_config.seed)

            # Create the model and load the weights.
            self.device = torch.device(self.gpt_config.device)

            # Sets the default torch dtype to the given dtype.
            torch.set_default_dtype(self.model_config.get_dtype())
            self.model = gemma_model.GemmaForCausalLM(self.model_config)
            self.model.load_weights(self.gpt_config.ckpt)
            self.model = self.model.to(self.device).eval()
            print("=======>>>>>>> Model loading done")
            self._initialized = True

    def generate(self, prompt, output_len):
        return self.model.generate(prompt, self.device, output_len=output_len)


# Generate the response.
# result = model.generate(args.prompt, device, output_len=args.output_len)
