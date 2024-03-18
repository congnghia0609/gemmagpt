"""
@author nghiatc
@since 07/03/2024
"""

import os.path
import sys
import webserver
import torch
import gemma_gpt

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

# python main.py
# make run
if __name__ == '__main__':
    try:
        sys.setrecursionlimit(10000)
        # init GemmaGPT
        gpt = gemma_gpt.GemmaGPT()
        # Start WebServer
        webserver.start()
    except KeyboardInterrupt:
        print("xxxxxxxxxxxxxxxxxx Event KeyboardInterrupt")
    finally:
        torch.set_default_dtype(torch.float)
        print("~~~~~~~~~~~~~~~~ Main Exit")
