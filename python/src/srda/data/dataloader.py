from logging import getLogger

from src.srda.data.dataset import (
    LatentDataset,
    SrdaByDdpmDataset,
    SrdaByDdpmDatasetScipy,
    SrdaByLdmDataset,
    SrdaDataset,
)
from src.srda.utils.utils import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


def make_dataloaders(
    dataset_name: str,
    train_files: list[str],
    valid_files: list[str],
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    world_size: int = 1,
    rank: int = 0,
    return_samplers: bool = False,
    seed: int = 42,
    **kwargs,
):
    dataset_initializer = None
    if dataset_name == "SrdaDataset":
        dataset_initializer = SrdaDataset
    elif dataset_name == "SrdaByDdpmDataset":
        dataset_initializer = SrdaByDdpmDataset
    elif dataset_name == "LatentDataset":
        dataset_initializer = LatentDataset
    elif dataset_name == "SrdaByLdmDataset":
        dataset_initializer = SrdaByLdmDataset
    elif dataset_name == "SrdaByDdpmDatasetScipy":
        dataset_initializer = SrdaByDdpmDatasetScipy
    else:
        raise NotImplementedError(f"{dataset_name} is not supported.")

    return _make_dataloaders(
        dataset_initilizer=dataset_initializer,
        train_files=train_files,
        valid_files=valid_files,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        world_size=world_size,
        rank=rank,
        return_samplers=return_samplers,
        seed=seed,
        **kwargs,
    )


def _make_dataloaders(
    dataset_initilizer,
    train_files: list[str],
    valid_files: list[str],
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    world_size: int,
    rank: int,
    return_samplers: bool,
    seed: int,
    **kwargs,
):
    _train = set(train_files)
    _valid = set(valid_files)

    assert _train.isdisjoint(_valid)

    dict_dataloaders = {}
    logger.info(f"Train batch size = {train_batch_size}")
    logger.info(f"Valid batch size = {valid_batch_size}")
    logger.info(f"DataLoader num_workers = {num_workers}, pin_memory = {pin_memory}")
    if num_workers > 0:
        logger.info(f"DataLoader prefetch_factor = {prefetch_factor}, persistent_workers = {persistent_workers}")

    sampler_dict = {"train": None, "valid": None}

    for kind, files in zip(["train", "valid"], [train_files, valid_files]):
        logger.info(f"Num files for {kind} = {len(files)}")

        dataset = dataset_initilizer(
            hr_file_paths=files,
            kind=kind,
            **kwargs,
        )

        is_train = kind == "train"
        sampler = None
        shuffle = is_train
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train,
                drop_last=is_train,
            )
            shuffle = False
        sampler_dict[kind] = sampler

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=train_batch_size if is_train else valid_batch_size,
            drop_last=True if is_train else False,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=get_torch_generator(seed),
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            loader_kwargs["persistent_workers"] = bool(persistent_workers)

        dict_dataloaders[kind] = DataLoader(**loader_kwargs)

        logger.info(
            f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
        )

    if return_samplers:
        return dict_dataloaders, sampler_dict
    return dict_dataloaders
