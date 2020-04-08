import sys
import os

cur_dir = os.path.dirname(__file__)
sys.path.append(
    os.path.abspath(os.path.join(cur_dir, '../src'))
)

import torchChar