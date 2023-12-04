# Instructions for the outputs (models weights, logs, etc.)

## Description of the outputs and how to obtain them

Our outputs consist of benchmark results all available in [this W&B project](https://wandb.ai/claire-labo/pytoych-benchmark).

Furthermore, for one experiment, we make our trained models available.
You can download them with the following commands:

```bash
# FROM the PROJECT_ROOT do
mkdir outputs/saved_models # Or make it a symlink, as you prefer.
wget 'https://www.dropbox.com/scl/fi/6nstmaeu2kmoksachkgte/saved_models.tar.gz?rlkey=ph7hiwf28g70md1xiijj4jn6e' -O outputs/saved_models/saved_models.tar.gz
tar -xzf outputs/saved_models/saved_models.tar.gz -C outputs/saved_models
```
