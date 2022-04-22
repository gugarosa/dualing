from dualing.models import ContrastiveSiamese
from dualing.models.base import CNN

# Creates the base architecture
cnn = CNN(n_blocks=3, init_kernel=5, n_output=128)

# Creates the contrastive siamese network
s = ContrastiveSiamese(
    cnn, margin=1.0, distance_metric="L2", name="contrastive_siamese"
)
