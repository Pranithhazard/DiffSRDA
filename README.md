# DiffSRDA

This is the public code release for **DiffSRDA**, a diffusion-model method for spatiotemporal super-resolution data assimilation.

The repo includes the code used for the paper:

- `exp1`: the main pixel-space DiffSRDA method
- `exp2`: the latent DiffSRDA / VQ-VAE appendix ablation
- baselines: EnKF-HR and EnKF-SR
- exp1 observation-consistency guidance
- regular-grid and fixed random sensor layouts

This repo does **not** include data, checkpoints, generated outputs, notebooks, or manuscript files.

## Quick Start

Use Python 3.10.

Install the main environment with Poetry:

```bash
poetry install --only main
export PYTHONPATH="$PWD/python"
```

If `torch` does not install cleanly through Poetry on your machine, install the right PyTorch wheel for your platform first, then rerun `poetry install`.

## Repo Map

```text
configs/SRDA/exp1/          main exp1 config
configs/SRDA/exp2/          main exp2 config
pyscripts/SRDA/exp1/        exp1 train and eval entrypoints
pyscripts/SRDA/exp2/        exp2 train and eval entrypoints
shellscripts/SRDA/          simple launch scripts
python/src/srda/            main DiffSRDA code
python/src/yasuda/          CFD / EnKF support code used by the pipelines
python/configs/srda/        mirrored baseline and latent-model configs
data/README.md              where external data should go
saved_models/README.md      where external checkpoints should go
```

## What You Need

You need to provide:

- external data under `data/`
- trained checkpoints under `saved_models/`

Expected data roots below are the paths used by the authors in the manuscript experiments. They can be changed to match your own local setup, as long as the config files and loaders are updated consistently:

```text
data/ddpm/external/hr_5000_simulations
data/ddpm/external/uhr_0050_simulations
data/ddpm/external/hr_5000_simulations_non_overlap
```

The checkpoint paths below are the default locations used by the authors. They can be changed to match your own setup by updating the corresponding config files or script arguments:

```text
saved_models/SRDA/exp1/weight_diffusion.pth
saved_models/SRDA/exp2/weight_diffusion.pth
saved_models/SRDA/latent_vqvae_ogi08/weight_latent.pth
```

Exp2 also needs the VQ-VAE checkpoint above.

## Main Commands

Train exp1:

```bash
PYTHONPATH=python poetry run python pyscripts/SRDA/exp1/train_srda_exp1.py \
  --config_path configs/SRDA/exp1/srda_exp1_ddim_sf4.yaml
```

Evaluate exp1:

```bash
PYTHONPATH=python poetry run python pyscripts/SRDA/exp1/eval_srda_exp1_cycle.py \
  --config_path configs/SRDA/exp1/srda_exp1_ddim_sf4.yaml
```

Evaluate exp1 with guidance:

```bash
PYTHONPATH=python poetry run python pyscripts/SRDA/exp1/eval_srda_exp1_cycle.py \
  --config_path configs/SRDA/exp1/srda_exp1_ddim_sf4.yaml \
  --obs_guidance_mode soft \
  --cond_grid_interval 8 \
  --guidance_grid_interval 4
```

Evaluate exp1 with a fixed random sensor layout:

```bash
PYTHONPATH=python poetry run python pyscripts/SRDA/exp1/eval_srda_exp1_cycle.py \
  --config_path configs/SRDA/exp1/srda_exp1_ddim_sf4.yaml \
  --sensor_scenario random_uniform_fixed \
  --sensor_seed 771155 \
  --sensor_num_sensors 128
```

Train exp2:

```bash
PYTHONPATH=python poetry run python pyscripts/SRDA/exp2/train_srda_exp2.py \
  --config_path configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08.yaml
```

Evaluate exp2:

```bash
PYTHONPATH=python poetry run python pyscripts/SRDA/exp2/eval_srda_exp2_cycle.py \
  --config_path configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08.yaml
```

Exp2 is a latent appendix ablation. The public exp2 path does not use observation guidance.

## Smoke Checks

These small runs are meant to check that training starts, loss moves, and files can be written before you launch larger jobs.

```bash
bash shellscripts/SRDA/exp1/smoke_train_exp1.sh 0 1
bash shellscripts/SRDA/exp2/smoke_train_exp2.sh 0 1
```

For multi-GPU runs, pass visible GPU ids and the matching process count, for example:

```bash
bash shellscripts/SRDA/exp1/smoke_train_exp1.sh 0,1 2
```

## Notes

- The main public configs are in `configs/SRDA/exp1/` and `configs/SRDA/exp2/`.
- Some baseline and latent-model defaults are also mirrored under `python/configs/srda/`.
- If you change data paths, model paths, or grid settings, keep the matching config files consistent.
- EnKF-HR may create a local covariance cache at `data/srda/processed/enkf_hr/sys_noise_covs.pickle`. Treat it as generated local data.

## Sanity Checks

Useful quick checks after install:

```bash
poetry check

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python poetry run python -c "from pathlib import Path; import ast; [ast.parse(p.read_text(), filename=str(p)) for p in Path('.').rglob('*.py') if '.venv' not in p.parts and '.cache' not in p.parts and '.git' not in p.parts and 'noncommit' not in p.parts]; print('ast parse ok')"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python poetry run python -c "import src.srda; import src.srda.data.dataset; import src.srda.model.ddim_sr_modules.diffusion; import src.srda.utils.sensor_scenarios; import src.yasuda.cfd_model.enkf.sr_enkf; print('import smoke ok')"

bash -n shellscripts/SRDA/exp1/deploy_train_exp1.sh
bash -n shellscripts/SRDA/exp1/smoke_train_exp1.sh
bash -n shellscripts/SRDA/exp1/deploy_eval_exp1.sh
bash -n shellscripts/SRDA/exp1/deploy_eval_exp1_guidance_ablation.sh
bash -n shellscripts/SRDA/exp2/deploy_train_exp2.sh
bash -n shellscripts/SRDA/exp2/smoke_train_exp2.sh
bash -n shellscripts/SRDA/exp2/deploy_eval_exp2.sh
```

## Citation

If you use this code, please cite the manuscript.
