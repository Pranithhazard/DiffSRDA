from __future__ import annotations

import argparse
from pathlib import Path
from logging import getLogger, StreamHandler, INFO

import torch
import yaml

import sys

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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="srda_exp2",
        help="Name of SRDA experiment (maps to python/configs/srda/<experiment_name>/...).",
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default="base",
        help="Name of SRDA test config (file <test_name>.yaml in the experiment folder).",
    )
    parser.add_argument(
        "--seeds_start",
        type=int,
        default=9995,
        help="First UHR seed index (inclusive).",
    )
    parser.add_argument(
        "--seeds_end",
        type=int,
        default=9999,
        help="Last UHR seed index (inclusive).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Batch size used during SRDA diffusion inference.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08.yaml",
        help="Path to the SRDA config to load (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=None,
        help="Directory where processed SRDA outputs should be stored (defaults to work/SRDA/exp2).",
    )
    parser.add_argument(
        "--timestep_respacing",
        type=int,
        default=None,
        help="Override diffusion inference timestep respacing (e.g., 20 or 1000).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Override diffusion eta for inference (e.g., 0.0 for DDIM, 1.0 for DDPM).",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable timing instrumentation (writes per-cycle/per-seed timing CSVs).",
    )
    parser.add_argument(
        "--timing_out_dir",
        type=str,
        default=None,
        help="Optional directory to store timing outputs (defaults under outputs_root/timing).",
    )
    parser.add_argument(
        "--timing_warmup_cycles",
        type=int,
        default=1,
        help="Number of initial assimilation cycles to exclude from timing averages.",
    )
    parser.add_argument(
        "--timing_skip_save_outputs",
        action="store_true",
        help="When --timing is enabled, skip writing large .npz/.csv outputs (timing-only runs).",
    )
    args = parser.parse_args()

    root = setup_paths()
    root_dir = str(root)

    # Import SRDA utilities now that root/python is on sys.path.
    from src.srda.utils.evaluate import evaluate_srda_dm_and_generate_obs

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
        eta_label = str(eta_value).replace(".", "p")
        suffix_parts.append(f"eta{eta_label}")
    if args.batch_size != 30:
        suffix_parts.append(f"bs{args.batch_size}")
    run_suffix = "_".join(suffix_parts) if suffix_parts else None

    outputs_root = (
        Path(args.outputs_root)
        if args.outputs_root
        else Path(root_dir) / "work" / "SRDA" / "exp2"
    )
    if not outputs_root.is_absolute():
        outputs_root = Path(root_dir) / outputs_root
    outputs_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using SRDA config for evaluation: {config_path}")
    if respacing_value is not None:
        logger.info(f"  Inference respacing: {respacing_value}")
    if eta_value is not None:
        logger.info(f"  Inference eta: {eta_value}")
    logger.info(f"Output root: {outputs_root} (suffix={run_suffix or 'none'})")

    logger.info(
        f"Running diffusion SRDA evaluation for seeds "
        f"{args.seeds_start}..{args.seeds_end}, batch_size={args.batch_size}"
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
        timing=args.timing,
        timing_out_dir=args.timing_out_dir,
        timing_warmup_cycles=args.timing_warmup_cycles,
        timing_skip_save_outputs=args.timing_skip_save_outputs,
    )

    logger.info(
        "SRDA Exp2 diffusion evaluation finished. "
        "Reuse EnKF outputs from Exp1 for comparisons."
    )
    logger.info("SRDA Exp2 evaluation cycle finished.")

if __name__ == "__main__":
    main()
