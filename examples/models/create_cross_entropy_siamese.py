# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

from dualing.models import CrossEntropySiamese
from dualing.models.base import MLP

mlp = MLP(n_hidden=(512, 256, 128))

model = CrossEntropySiamese(
    mlp,
    distance_metric="concat",
    name="cross_entropy_siamese",
)
