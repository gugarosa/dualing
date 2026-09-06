# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

from dualing.models import ContrastiveSiamese
from dualing.models.base import CNN

cnn = CNN(n_blocks=3, init_kernel=5, n_output=128)

model = ContrastiveSiamese(
    cnn,
    margin=1.0,
    distance_metric="L2",
    name="contrastive_siamese",
)
