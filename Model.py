from __future__ import print_function
# from helpers import datapreparation as datp
from helpers import dataset as ds
import torch

path = "datasets/training/piano_roll_fs1/"

batches = ds.pianoroll_dataset_batch(path, binarize=False)
print(batches.data[0])

