import glob
from logging import DEBUG, INFO, WARNING, FileHandler, StreamHandler, getLogger

from sklearn.model_selection import train_test_split
from src.srda.data.dataloader import make_dataloaders

logger = getLogger()


def make_dataloaders_dict(
    config_dataset: dict,
    root_dir: str,
    world_size: int = 1,
    rank: int = 0,
    return_samplers: bool = False,
):
    _files = sorted(
        p
        for p in glob.glob(
            f"{root_dir}{config_dataset['data_dir']}/seed*/*_hr_omega_*.npy", recursive=True
        )
    )

    _files, train_files = train_test_split(
        _files,
        test_size=config_dataset["train_data_len"],
        shuffle=True,
        random_state=42,
    )
    _files, valid_files = train_test_split(
        _files,
        test_size=config_dataset["valid_data_len"],
        shuffle=True,
        random_state=42,
    )

    _config = {
        "train_files": train_files,
        "valid_files": valid_files,
    }
    return make_dataloaders(
        **_config,
        **config_dataset,
        world_size=world_size,
        rank=rank,
        return_samplers=return_samplers,
    )
