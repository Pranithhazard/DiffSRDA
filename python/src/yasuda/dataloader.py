import functools
import glob
import os
import typing
from logging import getLogger

from sklearn.model_selection import train_test_split
from src.yasuda.dataset import (
    Dataset2dGaussJetSameTimeStep,
    DatasetMakingObsInside,
    DatasetMakingObsInsideTimeseriesSplitted,
    DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    DatasetVorticityAssimilation,
    DatasetVorticitySnapshot,
    DatasetUnsupervisedLearningPrototype,
)
from src.yasuda.utils import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


def get_all_file_paths_in_dir(dir_path: str) -> typing.Tuple[str, str]:
    lr_paths = sorted(glob.glob(f"{dir_path}/LR_*.npy"))
    hr_paths = sorted(glob.glob(f"{dir_path}/HR_*.npy"))

    for l, h in zip(lr_paths, hr_paths):
        assert h.split("seed")[-1] == l.split("seed")[-1], "Seed is different."

    return lr_paths, hr_paths


def split_file_paths(
    paths: typing.List[str], train_valid_test_ratios: typing.List[float]
) -> tuple:

    assert len(train_valid_test_ratios) == 3  # train, valid, test, three ratios

    test_size = train_valid_test_ratios[-1]
    _paths, test_paths = train_test_split(paths, test_size=test_size, shuffle=False)

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_paths, valid_paths = train_test_split(
        _paths, test_size=valid_size, shuffle=False
    )

    assert set(train_paths).isdisjoint(set(valid_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(valid_paths).isdisjoint(set(test_paths))

    return train_paths, valid_paths, test_paths


def get_all_train_valid_test_file_paths(
    lr_paths: typing.List[str],
    hr_paths: typing.List[str],
    train_valid_test_ratios: typing.List[float],
) -> dict:

    dict_paths = {"train": {}, "valid": {}, "test": {}}

    for label, paths in zip(["lr", "hr"], [lr_paths, hr_paths]):

        train_paths, valid_paths, test_paths = split_file_paths(
            paths, train_valid_test_ratios
        )

        dict_paths["train"][label] = train_paths
        dict_paths["valid"][label] = valid_paths
        dict_paths["test"][label] = test_paths

        logger.info(
            f"{label} path lengh: train {len(train_paths)}, valid = {len(valid_paths)}, test = {len(test_paths)}"
        )
    return dict_paths


def _make_dataloaders_2d_gauss_jet(
    *,
    dict_all_paths: dict,
    batch_size: int,
    list_bias: typing.List[float] = [0, 0, 0],
    list_scale: typing.List[float] = [1, 1, 1],
    prob_observation: float = 0.1,
    missing_value: float = float("nan"),
    use_clamp: bool = False,
    clamp_min: float = None,
    clamp_max: float = None,
    start_time_index: int = 0,
    sampling_freq: int = 1,
    end_time_index_observation: int = None,  # None means full observation, 0 means no observation
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(f"Seed passed to make_dataloaders_2d_gauss_jet = {seed}")

    dict_dataloaders, dict_samplers = {}, {}

    for kind in ["train", "valid", "test"]:

        dataset = Dataset2dGaussJetSameTimeStep(
            lr_paths=dict_all_paths[kind]["lr"],
            hr_paths=dict_all_paths[kind]["hr"],
            list_bias=list_bias,
            list_scale=list_scale,
            prob_observation=prob_observation,
            missing_value=missing_value,
            use_clamp=use_clamp,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            start_time_index=start_time_index,
            sampling_freq=sampling_freq,
            end_time_index_observation=end_time_index_observation,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_2d_gauss_jet(
    root_dir: str, config: dict, world_size: int = None, rank: int = None
):
    lr_paths, hr_paths = get_all_file_paths_in_dir(
        f"{root_dir}/data/pytorch/DL_data/{config['data']['data_dir_name']}"
    )
    dict_all_paths = get_all_train_valid_test_file_paths(
        lr_paths, hr_paths, config["data"]["train_valid_test_ratios"]
    )

    return _make_dataloaders_2d_gauss_jet(
        dict_all_paths=dict_all_paths,
        batch_size=config["data"]["batch_size"],
        prob_observation=config["data"]["prob_observation"],
        missing_value=config["data"]["missing_value"],
        list_bias=config["data"]["bias"],
        list_scale=config["data"]["scale"],
        use_clamp=True,
        clamp_min=config["data"]["clamp_min"],
        clamp_max=config["data"]["clamp_max"],
        start_time_index=config["data"]["start_time_index"],
        sampling_freq=config["data"]["sampling_freq"],
        end_time_index_observation=config["data"]["end_time_index_observation"],
        seed=config["data"]["seed"],
        world_size=world_size,
        rank=rank,
    )


def make_test_dataloader_2d_gauss_jet(
    *,
    root_dir: str,
    config: dict,
    prob_observation: float,
    end_time_index_observation: int,
    use_clamp: bool = True,
):
    lr_paths, hr_paths = get_all_file_paths_in_dir(
        f"{root_dir}/data/pytorch/DL_data/{config['data']['data_dir_name']}"
    )
    dict_all_paths = get_all_train_valid_test_file_paths(
        lr_paths, hr_paths, config["data"]["train_valid_test_ratios"]
    )

    dataloaders, _ = _make_dataloaders_2d_gauss_jet(
        dict_all_paths=dict_all_paths,
        batch_size=config["data"]["batch_size"],
        prob_observation=prob_observation,
        missing_value=config["data"]["missing_value"],
        list_bias=config["data"]["bias"],
        list_scale=config["data"]["scale"],
        use_clamp=use_clamp,
        clamp_min=config["data"]["clamp_min"],
        clamp_max=config["data"]["clamp_max"],
        start_time_index=config["data"]["start_time_index"],
        sampling_freq=config["data"]["sampling_freq"],
        end_time_index_observation=end_time_index_observation,
        seed=config["data"]["seed"],
    )

    return dataloaders["test"]


def _make_dataloaders_vorticity_assimilation(
    *,
    dict_dir_paths: dict,
    batch_size: int,
    assimilation_period: int,
    n_snapshots: int,
    obs_noise_std: float,
    is_always_start_observed: bool = True,
    list_bias: typing.List[float] = [0.0],
    list_scale: typing.List[float] = [1.0],
    missing_value: float = float("nan"),
    use_clamp: bool = False,
    clamp_min: float = None,
    clamp_max: float = None,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
    initial_discarded_period: int,
    use_obs: bool,
    is_output_only_last: bool,
    input_sampling_interval: int = None,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(f"Seed passed to _make_dataloaders_vorticity_assimilation = {seed}")
    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}

    for kind in ["train", "valid", "test"]:

        dataset = DatasetVorticityAssimilation(
            data_dirs=dict_dir_paths[kind],
            assimilation_period=assimilation_period,
            n_snapshots=n_snapshots,
            obs_noise_std=obs_noise_std,
            biases=list_bias,
            scales=list_scale,
            is_always_start_observed=is_always_start_observed,
            seed=seed,
            missing_value=missing_value,
            use_clipping=use_clamp,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            initial_discarded_period=initial_discarded_period,
            use_obs=use_obs,
            is_output_only_last=is_output_only_last,
            input_sampling_interval=input_sampling_interval,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_vorticity_assimilation(
    root_dir: str,
    config: dict,
    world_size: int = None,
    rank: int = None,
    use_clamp: bool = True,
):
    cfd_data_dir = f'{root_dir}/data/pytorch/CFD/{config["data"]["data_dir_name"]}'

    data_dirs = sorted([p for p in glob.glob(f"{cfd_data_dir}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["data"]["train_valid_test_ratios"]
    )

    if "train_datasize" in config["data"]:
        train_dirs = train_dirs[: config["data"]["train_datasize"]]
        logger.info(f"Train datasize is changed to {len(train_dirs)}")

    if "valid_datasize" in config["data"]:
        valid_dirs = valid_dirs[: config["data"]["valid_datasize"]]
        logger.info(f"Valid datasize is changed to {len(valid_dirs)}")

    dict_data_dirs = {"train": train_dirs, "valid": valid_dirs, "test": test_dirs}

    return _make_dataloaders_vorticity_assimilation(
        dict_dir_paths=dict_data_dirs,
        batch_size=config["data"]["batch_size"],
        assimilation_period=config["data"]["assimilation_period"],
        n_snapshots=config["data"]["n_snapshots"],
        obs_noise_std=config["data"]["obs_noise_std"],
        is_always_start_observed=config["data"]["is_always_start_observed"],
        list_bias=config["data"]["bias"],
        list_scale=config["data"]["scale"],
        missing_value=config["data"]["missing_value"],
        use_clamp=use_clamp,
        clamp_min=config["data"]["clamp_min"],
        clamp_max=config["data"]["clamp_max"],
        seed=config["data"]["seed"],
        world_size=world_size,
        rank=rank,
        initial_discarded_period=config["data"].get("initial_discarded_period", 16),
        use_obs=config["data"].get("use_obs", True),
        is_output_only_last=config["data"].get("is_output_only_last", False),
        input_sampling_interval=config["data"].get("input_sampling_interval", None),
    )


def _make_dataloaders_vorticity_snapshot(
    *,
    batch_size: int,
    dict_dir_paths: dict,
    assimilation_period: int,
    obs_noise_std: float,
    obs_interval: int = 27,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    dict_dataloaders, dict_samplers = {}, {}

    for kind in ["train", "valid", "test"]:

        dataset = DatasetVorticitySnapshot(
            data_dirs=dict_dir_paths[kind],
            assimilation_period=assimilation_period,
            obs_noise_std=obs_noise_std,
            obs_interval=obs_interval,
            seed=seed,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_vorticity_snapshot(
    root_dir: str,
    config: dict,
    world_size: int = None,
    rank: int = None,
):
    cfd_data_dir = f'{root_dir}/data/pytorch/CFD/{config["data"]["cfd_name"]}'

    data_dirs = sorted([p for p in glob.glob(f"{cfd_data_dir}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["data"]["train_valid_test_ratios"]
    )
    dict_data_dirs = {"train": train_dirs, "valid": valid_dirs, "test": test_dirs}

    return _make_dataloaders_vorticity_snapshot(
        dict_dir_paths=dict_data_dirs,
        batch_size=config["data"]["batch_size"],
        assimilation_period=config["data"]["assimilation_period"],
        obs_noise_std=config["data"]["obs_noise_std"],
        obs_interval=config["data"]["obs_interval"],
        num_workers=2,
        world_size=world_size,
        rank=rank,
    )


def _make_dataloaders_vorticity_making_observation_inside(
    *,
    dict_dir_paths: dict,
    assimilation_period: int,
    lr_input_sampling_interval: int,
    hr_output_sampling_interval: int,
    observation_interval: int,
    observation_noise_percentage: float,
    vorticity_bias: float,
    vorticity_scale: float,
    use_observation: bool,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
    **kwargs,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}

    for kind in ["train", "valid", "test"]:

        dataset = DatasetMakingObsInside(
            data_dirs=dict_dir_paths[kind],
            assimilation_period=assimilation_period,
            lr_input_sampling_interval=lr_input_sampling_interval,
            hr_output_sampling_interval=hr_output_sampling_interval,
            observation_interval=observation_interval,
            observation_noise_percentage=observation_noise_percentage,
            vorticity_bias=vorticity_bias,
            vorticity_scale=vorticity_scale,
            use_observation=use_observation,
            seed=seed,
            use_ground_truth_clamping=False if kind == "test" else True,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_vorticity_making_observation_inside(
    root_dir: str,
    cfd_dir_name: str,
    config: dict,
    world_size: int = None,
    rank: int = None,
):
    cfd_dir_path = f"{root_dir}/data/pytorch/CFD/{cfd_dir_name}"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["data"]["train_valid_test_ratios"]
    )

    if "train_datasize" in config["data"]:
        train_dirs = train_dirs[: config["data"]["train_datasize"]]
        logger.info(f"Train datasize is changed to {len(train_dirs)}")

    if "valid_datasize" in config["data"]:
        valid_dirs = valid_dirs[: config["data"]["valid_datasize"]]
        logger.info(f"Valid datasize is changed to {len(valid_dirs)}")

    dict_data_dirs = {"train": train_dirs, "valid": valid_dirs, "test": test_dirs}

    return _make_dataloaders_vorticity_making_observation_inside(
        dict_dir_paths=dict_data_dirs,
        world_size=world_size,
        rank=rank,
        **config["data"],
    )


def _make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
    *,
    dict_dir_paths: dict,
    lr_kind_names: typing.List[str],
    lr_time_interval: int,
    obs_time_interval: int,
    obs_grid_interval: int,
    obs_noise_std: float,
    use_observation: bool,
    vorticity_bias: float,
    vorticity_scale: float,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
    **kwargs,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}

    for kind in ["train", "valid", "test"]:

        dataset = DatasetMakingObsInsideTimeseriesSplitted(
            data_dirs=dict_dir_paths[kind],
            lr_kind_names=lr_kind_names,
            lr_time_interval=lr_time_interval,
            obs_time_interval=obs_time_interval,
            obs_grid_interval=obs_grid_interval,
            obs_noise_std=obs_noise_std,
            use_observation=use_observation,
            vorticity_bias=vorticity_bias,
            vorticity_scale=vorticity_scale,
            use_ground_truth_clamping=False if kind == "test" else True,
            seed=seed,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def _make_dataloaders_vorticity_making_observation_inside_time_series_splitted_with_mixup(
    *,
    dict_dir_paths: dict,
    lr_kind_names: typing.List[str],
    lr_time_interval: int,
    obs_time_interval: int,
    obs_grid_interval: int,
    obs_noise_std: float,
    use_observation: bool,
    vorticity_bias: float,
    vorticity_scale: float,
    use_mixup: bool,
    use_mixup_init_time: bool,
    beta_dist_alpha: float,
    beta_dist_beta: float,
    batch_size: int,
    use_lr_forecast: bool = True,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
    train_valid_test_kinds: typing.List[str] = ["train", "valid", "test"],
    is_output_only_last: bool = False,
    is_last_obs_missing: bool = False,
    min_start_time_index: int = 12,
    max_start_time_index: int = 96,
    **kwargs,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}

    for kind in train_valid_test_kinds:

        dataset = DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling(
            data_dirs=dict_dir_paths[kind],
            lr_kind_names=lr_kind_names,
            lr_time_interval=lr_time_interval,
            obs_time_interval=obs_time_interval,
            obs_grid_interval=obs_grid_interval,
            obs_noise_std=obs_noise_std,
            use_observation=use_observation,
            vorticity_bias=vorticity_bias,
            vorticity_scale=vorticity_scale,
            use_ground_truth_clamping=False if kind == "test" else True,
            seed=seed,
            use_mixup=use_mixup,
            use_mixup_init_time=use_mixup_init_time,
            use_lr_forecast=use_lr_forecast,
            beta_dist_alpha=beta_dist_alpha,
            beta_dist_beta=beta_dist_beta,
            is_output_only_last=is_output_only_last,
            is_last_obs_missing=is_last_obs_missing,
            min_start_time_index=min_start_time_index,
            max_start_time_index=max_start_time_index,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
    root_dir: str,
    config: dict,
    *,
    train_valid_test_kinds: typing.List[str] = ["train", "valid", "test"],
    world_size: int = None,
    rank: int = None,
):
    cfd_dir_path = f"{root_dir}/data/ddpm/external/uhr_0050_simulations"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["data"]["train_valid_test_ratios"]
    )

    dict_data_dirs = {"train": train_dirs, "valid": valid_dirs, "test": test_dirs}

    if "use_mixup" in config["data"]:
        logger.info("Dataloader with mixup is used.")
        return _make_dataloaders_vorticity_making_observation_inside_time_series_splitted_with_mixup(
            dict_dir_paths=dict_data_dirs,
            world_size=world_size,
            rank=rank,
            train_valid_test_kinds=train_valid_test_kinds,
            **config["data"],
        )
    else:
        return (
            _make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
                dict_dir_paths=dict_data_dirs,
                world_size=world_size,
                rank=rank,
                **config["data"],
            )
        )


def _set_hr_file_paths(data_dirs: typing.List[str]):
    lst_hr_file_paths = [
        glob.glob(f"{dir_path}/*_hr_omega.npy") for dir_path in data_dirs
    ]
    hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

    extracted_paths = []
    for path in hr_file_paths:
        if "start12" in path:
            continue
        if "start08" in path:
            continue
        if "start04" in path:
            continue
        if "start00" in path:
            continue
        extracted_paths.append(path)

    return extracted_paths


def get_hr_file_paths(
    root_dir: str, cfd_dir_name: str, train_valid_test_ratios: typing.List[str]
):
    cfd_dir_path = f"{root_dir}/data/pytorch/CFD/{cfd_dir_name}"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, train_valid_test_ratios
    )

    return {
        "train": _set_hr_file_paths(train_dirs),
        "valid": _set_hr_file_paths(valid_dirs),
        "test": _set_hr_file_paths(test_dirs),
    }


def _make_dataloaders_unsupervised_prototype(
    dict_dir_paths: dict,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
    **kwargs,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}

    for kind, data_dirs in dict_dir_paths.items():

        dataset = DatasetUnsupervisedLearningPrototype(
            data_dirs=data_dirs,
            use_gt_clamp=False if kind == "test" else True,
            **kwargs,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )

    return dict_dataloaders, dict_samplers


def make_dataloaders_unsupervised_prototype(
    root_dir: str,
    config: dict,
    *,
    train_valid_test_kinds: typing.List[str] = ["train", "valid", "test"],
    world_size: int = None,
    rank: int = None,
):
    cfd_dir_path = f"{root_dir}/data/pytorch/CFD/{config['data']['data_dir_name']}"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["data"]["train_valid_test_ratios"]
    )

    dict_data_dirs = {}

    for kind in train_valid_test_kinds:
        if kind == "train":
            dict_data_dirs[kind] = train_dirs
        elif kind == "valid":
            dict_data_dirs[kind] = valid_dirs
        elif kind == "test":
            dict_data_dirs[kind] = test_dirs
        else:
            raise Exception(f"{kind} is not supported.")

    return _make_dataloaders_unsupervised_prototype(
        dict_dir_paths=dict_data_dirs,
        world_size=world_size,
        rank=rank,
        **config["data"],
    )