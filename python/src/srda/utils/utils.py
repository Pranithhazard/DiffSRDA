import os
import random
from logging import getLogger

import numpy as np
import torch

logger = getLogger()

# How to reproduce the same results using pytorch.
# Ref.
# https://pytorch.org/docs/stable/notes/randomness.html


def set_seeds(seed: int = 42) -> None:
    """
    Do not forget to run `torch.use_deterministic_algorithms(True)`
    just after importing torch in your main program.
    """
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.error(e)


def seed_worker(worker_id: int):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class AverageMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
