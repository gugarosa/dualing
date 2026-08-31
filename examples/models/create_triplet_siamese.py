from dualing.models import TripletSiamese
from dualing.models.base import CNN

cnn = CNN(n_blocks=3, init_kernel=5, n_output=128, activation="linear")

model = TripletSiamese(
    cnn,
    loss="hard",
    margin=0.5,
    soft=False,
    distance_metric="L2",
    name="triplet_siamese",
)
