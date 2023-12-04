"""Based on the example in https://github.com/pytorch/examples/blob/main/mnist/main.py"""

import logging
import os
import sys
import time
from pathlib import Path

import cpuinfo
import hydra
import omegaconf
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from pytoych_benchmark import utils
from pytoych_benchmark.models.mnist import Net

# Hydra sets up the logger automatically.
# https://hydra.cc/docs/tutorials/basic/running_your_app/logging/
logger = logging.getLogger(__name__)

# Resolvers can be used in the config files.
# https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
# They are useful when you want to make the default values of some config variables
# result from direct computation of other config variables.
# Only put variables meant to be edited by the user (as opposed to read-only variables described below)
# and avoid making them too complicated, the point is not to write code in the config file.

# Useful to evaluate expressions in the config file.
OmegaConf.register_new_resolver("eval", eval, use_cache=True)
# Generate a random seed and record it in the config of the experiment.
OmegaConf.register_new_resolver(
    "generate_random_seed", utils.seeding.generate_random_seed, use_cache=True
)


def train(config, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config.device), target.to(config.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % config.train.log_interval == 0:
            loss = loss.item()
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "samples": batch_idx * len(data)
                    + len(train_loader.dataset) * (epoch - 1),
                    "loss_minibatch": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )


def test(config, model, test_loader, epoch):
    model.eval()
    test_loss = torch.tensor(0.0, device=config.device)
    correct = torch.tensor(0.0, device=config.device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum")
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss = (test_loss / len(test_loader.dataset)).item()
    correct = correct.item()

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    wandb.log(
        {
            "epoch": epoch,
            "test_loss": test_loss,
            "accuracy": 100.0 * correct / len(test_loader.dataset),
        }
    )


@hydra.main(version_base=None, config_path="configs", config_name="mnist")
def main(config: DictConfig) -> None:
    # Here you can make some computations with the config to add new keys, correct some values, etc.
    # E.g., read-only variables that can be useful when navigating the experiments on wandb (filtering, sorting, etc.).
    # Save the new config (as a file to record it) and pass it to wandb to record it with your experiment.

    # Resolve device.
    with omegaconf.open_dict(config):
        # Fix the devices (it's resolved to config.device by default).
        config.device_suggested = config.device_suggested
        if (
            config.device_suggested.startswith("cuda") and not torch.cuda.is_available()
        ) or (
            config.device_suggested == "mps" and not torch.backends.mps.is_available()
        ):
            config.device = "cpu"
        else:
            config.device = config.device_suggested

    # Record the hardware specs
    # For CPU
    cpu_specs = cpuinfo.get_cpu_info()
    cpu_specs = f'{cpu_specs["brand_raw"]}, {cpu_specs["count"]} cpus'
    # For training device
    if config.device.startswith("cuda"):
        device_specs = torch.cuda.get_device_name()
    elif config.device == "mps":
        gpu_model = (
            os.popen('system_profiler SPDisplaysDataType | grep "Model"')
            .read()
            .split()[-1]
        )
        n_gpu_cores = (
            os.popen('system_profiler SPDisplaysDataType | grep "Cores"')
            .read()
            .split()[-1]
        )
        device_specs = f"{gpu_model}, {n_gpu_cores} GPU cores"
    else:
        device_specs = cpu_specs

    with omegaconf.open_dict(config):
        config.cpu_specs = cpu_specs
        config.device_specs = device_specs

    # Conda or non-conda environment (to compare CUDA in conda vs. CUDA in NGC).
    executable = sys.executable
    is_conda = (
        "mamba" in executable or "conda" in executable
    )  # Assuming the prefix was unchanged.
    with omegaconf.open_dict(config):
        config.is_conda = is_conda

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

    # Record total runtime.
    total_runtime = time.time()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        f"{config.data_dir}/mnist", train=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        f"{config.data_dir}/mnist", train=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        pin_memory=True,
        num_workers=config.test.n_workers,
    )

    model = Net().to(config.device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.train.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=config.train.lr_decay_per_epoch)
    test(config, model, test_loader, 0)
    for epoch in range(1, config.train.n_epochs + 1):
        train(config, model, train_loader, optimizer, epoch)
        test(config, model, test_loader, epoch)
        scheduler.step()

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # Record total runtime.
    total_runtime = time.time() - total_runtime
    wandb.log({"total_runtime": total_runtime})


if __name__ == "__main__":
    main()
