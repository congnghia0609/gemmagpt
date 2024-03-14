"""
@author nghiatc
@since 15/03/2024
"""


class GPTConfig:
    def __init__(self):
        # path to file checkpoint model GemmaGPT
        self.ckpt = "/home/nghiatc/lab/labGemma/pytorch-2b-it/gemma-2b-it.ckpt"
        self.variant = "2b"  # ["2b", "7b"]
        self.device = "cpu"  # ["cpu", "cuda"]
        self.seed = 123456
        self.quant = False  # True when run with the int8 quantized model

