from dualing.models import CrossEntropySiamese
from dualing.models.base import MLP

# Creates the base architecture
mlp = MLP(n_hidden=(512, 256, 128))

# Creates the cross-entropy siamese network
s = CrossEntropySiamese(mlp, distance_metric='concat', name='cross_entropy_siamese')
