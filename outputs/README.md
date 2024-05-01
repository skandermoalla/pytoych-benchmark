# Instructions for the outputs (models weights, logs, etc.)

## Description of the outputs and how to obtain them

Our outputs consist of benchmark results all available in [this W&B project](https://wandb.ai/claire-labo/pytoych-benchmark).

Furthermore, for one experiment, we make our trained models available.
Download them [here](https://www.icloud.com/iclouddrive/075gJe3IqQ5QtRgxlGAEk5xbQ#saved_models).
Then move them to the machine where you are running the code and extract them with the following commands:

```bash
# FROM the PROJECT_ROOT do
mkdir outputs/saved_models # Or make it a symlink, as you prefer.
tar -xzf saved_models.tar.gz -C outputs/saved_models
```
