"""Fake script to show how to share models and load them for some experiment."""

import logging
from pathlib import Path

import hydra
import omegaconf
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms

from pytoych_benchmark import utils
from pytoych_benchmark.mnist import test
from pytoych_benchmark.models.mnist import Net

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="mnist_evaluate")
def main(config: DictConfig) -> None:
    # Resolve device.
    with omegaconf.open_dict(config):
        if (config.device.startswith("cuda") and not torch.cuda.is_available()) or (
            config.device == "mps" and not torch.backends.mps.is_available()
        ):
            config.device = "cpu"

    # Record the config with wandb.
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        project=config.wandb.project,
        tags=config.wandb.tags,
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )

    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config, resolve=True)}")

    # Update this function whenever you have a library that needs to be seeded.
    utils.seeding.seed_everything(config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        f"{config.data_dir}/mnist", train=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.test.batch_size,
        pin_memory=True,
        num_workers=config.test.n_workers,
    )

    model = Net().to(config.device)
    model.load_state_dict(
        torch.load(
            f"{config.outputs_dir}/{config.load_model_dir}/mnist_cnn.pt",
            map_location=config.device,
        )
    )
    test(config, model, test_loader, 0)


if __name__ == "__main__":
    main()
