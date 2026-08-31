from dualing.models import CrossEntropySiamese
from dualing.models.base import MLP

mlp = MLP(n_hidden=(512, 256, 128))

model = CrossEntropySiamese(
    mlp,
    distance_metric="concat",
    name="cross_entropy_siamese",
)
