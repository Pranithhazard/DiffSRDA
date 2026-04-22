import csv
import os
from logging import getLogger
from typing import Union

import numpy as np

logger = getLogger()


def save_list_as_csv(output_file_path: str, file: Union[list, np.array]):
    if not os.path.exists(output_file_path):
        with open(output_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Value"])  # ヘッダー行を書き込み
            writer.writerows([[item] for item in file])
        logger.info(f"File '{output_file_path}' saved.")
    else:
        logger.info(f"File '{output_file_path}' already exists. Not overwritten.")
