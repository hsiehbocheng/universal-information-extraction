import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import paddle
# from utils import convert_example, reader

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.trainer import (
    CompressionArguments,
    PdArgumentParser,
    Trainer,
    get_last_checkpoint,
)
from paddlenlp.transformers import UIE, AutoTokenizer, export_model
from paddlenlp.utils.log import logger
