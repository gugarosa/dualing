from dualing.models import TripletSiamese
from dualing.models.base import CNN

# Creates the base architecture
cnn = CNN(n_blocks=3, init_kernel=5, n_output=128, activation='linear')

# Creates the triplet siamese network
s = TripletSiamese(cnn, loss='hard', margin=0.5, soft=False, distance_metric='L2', name='triplet_siamese')
