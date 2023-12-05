import numpy as np
position_field = np.load('data/input_data/data_coordinates.npy')
toroidal_field = np.load('data/input_data/B_field_Toroidal.npy')
solenoid_field = np.load('data/input_data/B_field_Solenoid.npy')

#3d quiver plot of position and solenoid field

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')

#sample 50% of the data
sample = np.random.choice(position_field.shape[0], int(position_field.shape[0]/100), replace=False)
ax.quiver(position_field[sample,0], position_field[sample,1], position_field[sample,2], toroidal_field[sample,0], toroidal_field[sample,1], toroidal_field[sample,2], length=0.7, normalize=True)

plt.show()