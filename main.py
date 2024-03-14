"""
@author nghiatc
@since 07/03/2024
"""

import os.path
import sys
import webserver

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

# python main.py
# make run
if __name__ == '__main__':
    try:
        sys.setrecursionlimit(10000)
        # Start WebServer
        webserver.start()
    except KeyboardInterrupt:
        print("xxxxxxxxxxxxxxxxxx Event KeyboardInterrupt")
    finally:
        print("~~~~~~~~~~~~~~~~ Main Exit")
