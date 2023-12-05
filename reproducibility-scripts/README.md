# Reproducibility scripts

This directory contains scripts to reproduce the results of the paper.

## Benchmark

The benchmark (figure in the README) can be run by running the WandB sweep `runtime-benchmark.yaml`.
Details to create it and run are provided in the file.
You can also find the live results at https://wandb.ai/claire-labo/pytoych-benchmark.

The hardware specs (CPU and GPU) are automatically picked up, and if
you want to tag your machine, you can add an override config `src/configs/override/mnist.yaml` with the following content:

```yaml
# @package _global_
# The above line should appear in the override configs so that they sit at the root of the config tree.

device: mps
hardware_tag: MacBook Air
```

## Experiments

We performed one experiment to show how you can save model weights, share them, and reproduce the results with them.

Run them with

```bash
# From the PROJECT ROOT
zsh reproducibility-scripts/script_name.sh
```

Read the instructions in each script for their setups (download models, etc).

* `train-model.sh` describes how to train a model and export it.
* `evaluate-saved-model.sh` describes how to load the model and evaluate it.
