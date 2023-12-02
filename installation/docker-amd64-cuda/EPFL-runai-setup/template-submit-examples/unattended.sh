runai submit \
  --name example-unattended \
  --image registry.rcp.epfl.ch/claire/moalla/pytoych-benchmark:run-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/pytoych-benchmark/run \
  -- python -m pytoych_benchmark.some_experiment some_arg=2

# or -- zsh pytoych_benchmark/reproducibility-scripts/some_experiment.sh

# To separate the dev state of the project from frozen checkouts to be used in unattended jobs you can observe that
# we're pointing to the .../run instance of the repository on the PVC.
# That would be a copy of the pytoych-benchmark repo frozen in a commit at a working state to be used in unattended jobs.
# Otherwise while developing we would change the code that would be picked by newly scheduled jobs.

# Useful commands.
# runai describe job example-unattended
# runai logs example-unattended
