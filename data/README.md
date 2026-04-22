# External Data

No dataset files are versioned in this repository.

The default configs expect external data under:

```text
data/ddpm/external/hr_5000_simulations
data/ddpm/external/uhr_0050_simulations
data/ddpm/external/hr_5000_simulations_non_overlap
```

The leading slash used in YAML config values such as `/data/ddpm/...` is treated by the scripts as repository-root-relative because the code concatenates it with the repo root.

Training dataloaders look for HR files matching `seed*/*_hr_omega_*.npy`. The paired LR files are inferred by replacing `hr_omega` with `lr_omega_no-noise` in the same path. Evaluation and EnKF routines also expect UHR seed folders such as `data/ddpm/external/uhr_0050_simulations/seed00000`.

EnKF-HR may create a local covariance cache at `data/srda/processed/enkf_hr/sys_noise_covs.pickle`. Treat it as generated data.

Populate these directories locally, or update the `datasets.data_dir` fields in the YAML configs for your own similarly structured data. Do not commit dataset contents, covariance caches, or generated outputs here.
