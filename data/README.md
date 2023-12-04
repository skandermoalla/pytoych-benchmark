# Instructions for the data

## Description of the data

The data consists of the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

## Instructions to obtain the data

The dataset is 64 MB.
The following script will download it in the `data/mnist` folder.
You can make `data/mnist` a symbolic link to a folder in another location if needed.

```bash
# From the the PROJECT ROOT directory (i.e. the root folder of the repository)
# (Make sure the environment is activated for the conda installation))
python data/download_mnist.py
```
