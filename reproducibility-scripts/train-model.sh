exec python -m pytoych_benchmark.mnist \
  hydra.run.dir=outputs/saved_models/mnist

# To get an archive of the model, run:
# From the root of the repository:
# tar -czf outputs/saved_models/saved_models.tar.gz -C outputs/saved_models .
