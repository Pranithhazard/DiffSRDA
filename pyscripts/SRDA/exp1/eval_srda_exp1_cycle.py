from __future__ import annotations

import argparse
import re
import sys
from logging import INFO, StreamHandler, getLogger
from pathlib import Path

import torch
import yaml

# Ensure repo root is on sys.path before importing project modules.
here = Path(__file__).resolve()
root = here
for p in [here.parent, *here.parents]:
    if (p / "pyproject.toml").exists():
        root = p
        break
if str(root) not in sys.path:
    sys.path.append(str(root))

from utils.path_setup import setup_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="srda_exp1")
    parser.add_argument("--test_name", type=str, default="base")
    parser.add_argument("--seeds_start", type=int, default=9995)
    parser.add_argument("--seeds_end", type=int, default=9999)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/SRDA/exp1/srda_exp1_ddim_sf4.yaml",
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--outputs_root", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--timestep_respacing", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument(
        "--obs_guidance_mode",
        type=str,
        default="off",
        choices=["off", "hard", "soft"],
    )
    parser.add_argument("--obs_guidance_gamma", type=float, default=1.0)
    parser.add_argument("--obs_guidance_sigma", type=float, default=None)
    parser.add_argument("--obs_guidance_every", type=int, default=1)
    parser.add_argument("--obs_guidance_blur_sigma_px", type=float, default=0.0)
    parser.add_argument("--obs_guidance_blur_sigma_px_final", type=float, default=None)
    parser.add_argument("--obs_guidance_blur_schedule_power", type=float, default=1.0)
    parser.add_argument("--obs_guidance_tighten_final_steps", type=int, default=0)
    parser.add_argument("--obs_guidance_recompute_eps", action="store_true")
    parser.add_argument("--cond_grid_interval", type=int, default=None)
    parser.add_argument("--guidance_grid_interval", type=int, default=None)
    parser.add_argument("--cond_obs_noise_sigma", type=float, default=None)
    parser.add_argument("--guidance_obs_noise_sigma", type=float, default=None)
    parser.add_argument("--obs_noise_seed", type=int, default=None)
    parser.add_argument(
        "--sensor_scenario",
        type=str,
        default="legacy",
        choices=["legacy", "regular_grid_fixed", "random_uniform_fixed"],
    )
    parser.add_argument("--sensor_seed", type=int, default=None)
    parser.add_argument("--sensor_obs_noise_sigma", type=float, default=None)
    parser.add_argument("--sensor_grid_interval", type=int, default=None)
    parser.add_argument("--sensor_num_sensors", type=int, default=None)
    parser.add_argument("--skip_srda_dm", action="store_true")
    parser.add_argument("--skip_enkf_hr", action="store_true")
    parser.add_argument("--skip_enkf_bicubic", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing_out_dir", type=str, default=None)
    parser.add_argument("--timing_warmup_cycles", type=int, default=1)
    parser.add_argument("--timing_skip_save_outputs", action="store_true")
    args = parser.parse_args()

    root = setup_paths()
    root_dir = str(root)

    from src.srda.utils.evaluate import evaluate_srda_dm_and_generate_obs
    from src.srda.utils.perform_enkf import perform_enkf_bicubic, perform_enkf_hr

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = getLogger()
    if not any(isinstance(h, StreamHandler) for h in logger.handlers):
        logger.addHandler(StreamHandler())
    logger.setLevel(INFO)

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path(root_dir) / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as cfg_file:
        srda_config = yaml.safe_load(cfg_file)

    if args.model_dir is not None:
        model_dir = str(args.model_dir)
        if model_dir.startswith(root_dir):
            model_dir = model_dir[len(root_dir) :]
        if not model_dir.startswith("/"):
            model_dir = "/" + model_dir
        srda_config.setdefault("path", {})["model"] = model_dir

    base_grid_interval = int(srda_config.get("datasets", {}).get("obs_grid_interval", 8))
    base_obs_noise_std = float(srda_config.get("datasets", {}).get("obs_noise_std", 0.0))
    default_obs_noise_seed = 771155

    if args.timestep_respacing is not None:
        srda_config["diffusion_model"]["beta_schedule"]["val"]["timestep_respacing"] = [
            args.timestep_respacing
        ]
    val_schedule = srda_config["diffusion_model"]["beta_schedule"]["val"]
    respacing_value = None
    if isinstance(val_schedule, dict):
        resp = val_schedule.get("timestep_respacing")
        if isinstance(resp, (list, tuple)) and len(resp) > 0:
            respacing_value = resp[0]
    if args.eta is not None:
        srda_config["diffusion_model"]["diffusion"]["eta"] = args.eta
    eta_value = srda_config["diffusion_model"]["diffusion"].get("eta")

    suffix_parts = []
    if respacing_value is not None:
        suffix_parts.append(f"tr{respacing_value}")
    if eta_value is not None:
        suffix_parts.append(f"eta{str(eta_value).replace('.', 'p')}")
    if args.batch_size != 30:
        suffix_parts.append(f"bs{args.batch_size}")
    if args.obs_guidance_mode != "off":
        suffix_parts.append(f"og{args.obs_guidance_mode}")
        if args.obs_guidance_mode == "soft" and args.obs_guidance_gamma != 1.0:
            suffix_parts.append(f"g{str(args.obs_guidance_gamma).replace('.', 'p')}")
        if args.obs_guidance_every != 1:
            suffix_parts.append(f"step{args.obs_guidance_every}")
        blur_start = float(args.obs_guidance_blur_sigma_px)
        blur_final = args.obs_guidance_blur_sigma_px_final
        if blur_final is not None:
            suffix_parts.append(
                "blur"
                + str(blur_start).replace(".", "p")
                + "to"
                + str(blur_final).replace(".", "p")
            )
            if args.obs_guidance_blur_schedule_power != 1.0:
                suffix_parts.append(
                    f"bpow{str(args.obs_guidance_blur_schedule_power).replace('.', 'p')}"
                )
        elif blur_start > 0:
            suffix_parts.append(f"blur{str(blur_start).replace('.', 'p')}")
        if args.obs_guidance_tighten_final_steps > 0:
            suffix_parts.append(f"tight{int(args.obs_guidance_tighten_final_steps)}")
        if args.obs_guidance_recompute_eps:
            suffix_parts.append("epsrec")

    sensor_scenario = str(args.sensor_scenario or "legacy").lower()
    scenario_requested = sensor_scenario != "legacy"
    sensor_shift_conflict = scenario_requested and any(
        v is not None
        for v in (
            args.cond_grid_interval,
            args.guidance_grid_interval,
            args.cond_obs_noise_sigma,
            args.guidance_obs_noise_sigma,
        )
    )
    if sensor_shift_conflict:
        raise ValueError(
            "sensor_scenario != 'legacy' cannot be combined with cond/guid sensor-shift knobs "
            "(cond_grid_interval/guidance_grid_interval/cond_obs_noise_sigma/guidance_obs_noise_sigma)."
        )

    if scenario_requested:
        sensor_seed = (
            int(args.sensor_seed)
            if args.sensor_seed is not None
            else (
                int(args.obs_noise_seed)
                if args.obs_noise_seed is not None
                else default_obs_noise_seed
            )
        )
        sensor_obs_noise_sigma = (
            float(args.sensor_obs_noise_sigma)
            if args.sensor_obs_noise_sigma is not None
            else float(base_obs_noise_std)
        )
        sensor_grid_interval = (
            int(args.sensor_grid_interval)
            if args.sensor_grid_interval is not None
            else int(base_grid_interval)
        )
        if sensor_obs_noise_sigma < 0:
            raise ValueError("sensor_obs_noise_sigma must be >= 0")

        if sensor_scenario == "regular_grid_fixed":
            if sensor_grid_interval <= 0:
                raise ValueError("sensor_grid_interval must be >= 1")
            suffix_parts.append(f"sensregfix_ogi{sensor_grid_interval:02d}")
        elif sensor_scenario == "random_uniform_fixed":
            if args.sensor_num_sensors is None or int(args.sensor_num_sensors) <= 0:
                raise ValueError("sensor_num_sensors must be set and >0 for random_uniform_fixed.")
            suffix_parts.append(f"sensrandfix_Ns{int(args.sensor_num_sensors)}")
        else:
            raise ValueError(f"Unknown sensor_scenario: {sensor_scenario}")

        suffix_parts.append(f"sn{str(sensor_obs_noise_sigma).replace('.', 'p')}")
        suffix_parts.append(f"sseed{sensor_seed}")

    sensor_shift_requested = any(
        v is not None
        for v in (
            args.cond_grid_interval,
            args.guidance_grid_interval,
            args.cond_obs_noise_sigma,
            args.guidance_obs_noise_sigma,
            args.obs_noise_seed,
        )
    )
    if sensor_shift_requested and not scenario_requested:
        cond_ogi = int(args.cond_grid_interval) if args.cond_grid_interval is not None else base_grid_interval
        guid_ogi = int(args.guidance_grid_interval) if args.guidance_grid_interval is not None else cond_ogi
        cond_sigma = float(args.cond_obs_noise_sigma) if args.cond_obs_noise_sigma is not None else base_obs_noise_std
        guid_sigma = float(args.guidance_obs_noise_sigma) if args.guidance_obs_noise_sigma is not None else cond_sigma
        obs_seed = int(args.obs_noise_seed) if args.obs_noise_seed is not None else default_obs_noise_seed

        if cond_ogi <= 0 or guid_ogi <= 0:
            raise ValueError("cond_grid_interval and guidance_grid_interval must be >= 1")
        if cond_sigma < 0 or guid_sigma < 0:
            raise ValueError("cond_obs_noise_sigma and guidance_obs_noise_sigma must be >= 0")

        if cond_ogi == guid_ogi:
            suffix_parts.append(f"ogi{cond_ogi:02d}")
        else:
            suffix_parts.append(f"condogi{cond_ogi:02d}")
            suffix_parts.append(f"guidogi{guid_ogi:02d}")
        if cond_sigma == guid_sigma:
            suffix_parts.append(f"n{str(cond_sigma).replace('.', 'p')}")
        else:
            suffix_parts.append(f"condn{str(cond_sigma).replace('.', 'p')}")
            suffix_parts.append(f"guidn{str(guid_sigma).replace('.', 'p')}")
        suffix_parts.append(f"obsseed{obs_seed}")

    if args.model_tag:
        tag = re.sub(r"[^0-9A-Za-z_-]+", "", str(args.model_tag))
        if tag:
            suffix_parts.append(tag)
    run_suffix = "_".join(suffix_parts) if suffix_parts else None

    outputs_root = Path(args.outputs_root) if args.outputs_root else Path(root_dir) / "work" / "SRDA" / "exp1"
    if not outputs_root.is_absolute():
        outputs_root = Path(root_dir) / outputs_root
    outputs_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using SRDA config for evaluation: {config_path}")
    if respacing_value is not None:
        logger.info(f"  Inference respacing: {respacing_value}")
    if eta_value is not None:
        logger.info(f"  Inference eta: {eta_value}")
    logger.info(f"Output root: {outputs_root} (suffix={run_suffix or 'none'})")

    if not args.skip_srda_dm:
        logger.info(
            f"Running diffusion SRDA evaluation for seeds {args.seeds_start}..{args.seeds_end}, batch_size={args.batch_size}"
        )
        evaluate_srda_dm_and_generate_obs(
            i_seed_start=args.seeds_start,
            i_seed_end=args.seeds_end,
            root_dir=root_dir,
            experiment_name=args.experiment_name,
            test_name=args.test_name,
            device=device,
            batch_size=args.batch_size,
            config=srda_config,
            processed_base_dir=str(outputs_root),
            run_suffix=run_suffix,
            obs_guidance_mode=args.obs_guidance_mode,
            obs_guidance_gamma=args.obs_guidance_gamma,
            obs_guidance_sigma=args.obs_guidance_sigma,
            obs_guidance_every=args.obs_guidance_every,
            obs_guidance_blur_sigma_px=args.obs_guidance_blur_sigma_px,
            obs_guidance_recompute_eps=args.obs_guidance_recompute_eps,
            obs_guidance_tighten_final_steps=args.obs_guidance_tighten_final_steps,
            obs_guidance_blur_sigma_px_final=args.obs_guidance_blur_sigma_px_final,
            obs_guidance_blur_schedule_power=args.obs_guidance_blur_schedule_power,
            cond_grid_interval=args.cond_grid_interval,
            guidance_grid_interval=args.guidance_grid_interval,
            cond_obs_noise_sigma=args.cond_obs_noise_sigma,
            guidance_obs_noise_sigma=args.guidance_obs_noise_sigma,
            obs_noise_seed=args.obs_noise_seed,
            sensor_scenario=args.sensor_scenario,
            sensor_seed=args.sensor_seed,
            sensor_obs_noise_sigma=args.sensor_obs_noise_sigma,
            sensor_grid_interval=args.sensor_grid_interval,
            sensor_num_sensors=args.sensor_num_sensors,
            timing=args.timing,
            timing_out_dir=args.timing_out_dir,
            timing_warmup_cycles=args.timing_warmup_cycles,
            timing_skip_save_outputs=args.timing_skip_save_outputs,
        )

    if not args.skip_enkf_hr:
        logger.info("Running EnKF baseline (HR) for the same seeds...")
        perform_enkf_hr(
            i_seed_start=args.seeds_start,
            i_seed_end=args.seeds_end,
            root_dir=root_dir,
            experiment_name=args.experiment_name,
            test_name=args.test_name,
            device=device,
            processed_base_dir=str(outputs_root),
            run_suffix=run_suffix,
            timing=args.timing,
            timing_out_dir=args.timing_out_dir,
            timing_warmup_cycles=args.timing_warmup_cycles,
            timing_skip_save_outputs=args.timing_skip_save_outputs,
            timing_run_suffix="enkf_hr",
        )

    if not args.skip_enkf_bicubic:
        logger.info("Running EnKF baseline (SR / bicubic) for the same seeds...")
        perform_enkf_bicubic(
            i_seed_start=args.seeds_start,
            i_seed_end=args.seeds_end,
            root_dir=root_dir,
            experiment_name=args.experiment_name,
            test_name=args.test_name,
            device=device,
            processed_base_dir=str(outputs_root),
            run_suffix=run_suffix,
            timing=args.timing,
            timing_out_dir=args.timing_out_dir,
            timing_warmup_cycles=args.timing_warmup_cycles,
            timing_skip_save_outputs=args.timing_skip_save_outputs,
            timing_run_suffix="enkf_bicubic",
        )

    logger.info("SRDA Exp1 evaluation cycle finished.")


if __name__ == "__main__":
    main()
