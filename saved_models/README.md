# External Checkpoints

No pretrained weights or checkpoint binaries are versioned in this repository.

Default checkpoint locations used by the public configs are:

```text
saved_models/SRDA/exp1/weight_diffusion.pth
saved_models/SRDA/exp2/weight_diffusion.pth
saved_models/SRDA/latent_vqvae_ogi08/weight_latent.pth
```

Exp2 is latent DiffSRDA and requires the VQ-VAE checkpoint at `saved_models/SRDA/latent_vqvae_ogi08/weight_latent.pth` before its train/eval entrypoints can load the latent model.

Place checkpoints locally if you want to run evaluation with existing weights. Do not commit model binaries here.
