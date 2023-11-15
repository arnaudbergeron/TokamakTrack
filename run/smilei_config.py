# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------

import math as m
import numpy as np
import pandas as pd

TkeV = 10.						# electron & ion temperature in keV
T   = TkeV/511.   				# electron & ion temperature in me c^2
n0  = 1.
Lde = m.sqrt(T)					# Debye length in units of c/\omega_{pe}
dx  = 0.5 				# cell length (same in x & y)
dy  = dx
dz  = dx
dt  = (0.95 * dx/m.sqrt(3.))/2		# timestep (0.95 x CFL)

Lx    = 32
Ly    = 32
Lz    = 10
Tsim  = m.pi*2/2

x_shift = 15
y_shift = 15
z_shift = 5

b_multiplier = 1
solenoid_factor = 6
momentum_multiplier = 500

position_field = np.load('data/input_data/data_coordinates.npy')
toroidal_field = np.load('data/input_data/B_field_Toroidal.npy')
solenoid_field = np.load('data/input_data/B_field_Solenoid.npy')

position_field[:, 0] = position_field[:, 0]
position_field[:, 1] = position_field[:, 1]
position_field[:, 2] = position_field[:, 2]


list_data = position_field.tolist()
rounded_index = []
for i in list_data:
	rounded_index.append(tuple([round(j,3) for j in i]))
	
index_multi = pd.MultiIndex.from_tuples(rounded_index, names=["x", "y","z"])
df = pd.DataFrame(index = index_multi)
df['B_tor_x'] = toroidal_field[:,0]
df['B_tor_y'] = toroidal_field[:,1]
df['B_tor_z'] = toroidal_field[:,2]

df['B_sol_x'] = solenoid_field[:,0]
df['B_sol_y'] = solenoid_field[:,1]
df['B_sol_z'] = solenoid_field[:,2]

def create_initial_positions(r, discretized_pos):
	# Put inital particles in a discretized circle of radius r
    # with a constant density n0
    # and a constant tangential momentum p_t = 0.1

    # Number of particles in the circle
    N = int(200)
    
    # Create a list of N positions
    positions =  np.zeros((4, N))
    #make a list of 1,2,3,4,5,...
    theta = np.linspace(0, 2*np.pi, N)

    positions[0,:] = r*np.cos(theta) + x_shift
    positions[1,:] = r*np.sin(theta) + y_shift
    positions[2,:] = z_shift
    positions[3,:] = 0.25

    # find index of the closest positions in the discretized field
    discretized_initial_pos = []
    for i in range(N):
        ind = np.argmin(np.sqrt((discretized_pos[:,0] - positions[0,i])**2 + \
                     (discretized_pos[:,1] - positions[1,i])**2 + \
                     (discretized_pos[:,2] - positions[2,i])**2))
        discretized_initial_pos.append(discretized_pos[ind,:].tolist() + [1.])
    #remove duplicates
    discretized_initial_pos = list(set(tuple(i) for i in discretized_initial_pos))
    discretized_initial_pos = np.array(discretized_initial_pos)
    discretized_initial_pos[:,3] = 1/(discretized_initial_pos.shape[0])

    new_theta = np.arctan2(discretized_initial_pos[:,1]-x_shift, discretized_initial_pos[:,0]-y_shift)
    tengential_momentum = np.zeros((3, new_theta.shape[0]))
    tengential_momentum[0,:] = -np.sin(new_theta)
    tengential_momentum[1,:] = np.cos(new_theta)
    
    discretized_initial_pos = discretized_initial_pos.T

    tengential_momentum = tengential_momentum*momentum_multiplier
    return discretized_initial_pos, tengential_momentum

	
def time_evolution(b_sol, t):
	return b_sol * solenoid_factor # * (t)

def get_bx(x,y,z,t):
	#change for variable time
    if x<0 or y<0 or z<0 or z >= Lz or x >= Lx or y >= Ly:
        return 0
    x, y, z = round(x,3), round(y,3), round(z,3)
    b_tor = df.loc[(x,y,z)]['B_tor_x']
    b_sol = df.loc[(x,y,z)]['B_sol_x']
    return (b_tor + time_evolution(b_sol, t))*b_multiplier


def get_by(x,y,z,t):
    if x<0 or y<0 or z<0 or z >= Lz or x >= Lx or y >= Ly:
        return 0
    x, y, z = round(x,3), round(y,3), round(z,3)
    b_tor = df.loc[(x,y,z)]['B_tor_y']
    b_sol = df.loc[(x,y,z)]['B_sol_y']
    
    return (b_tor + time_evolution(b_sol, t))*b_multiplier


def get_bz(x,y,z,t):
    if x<0 or y<0 or z<0 or z >= Lz or x >= Lx or y >= Ly:
        return 0
    x, y, z = round(x,3), round(y,3), round(z,3)
    b_tor = df.loc[(x,y,z)]['B_tor_z']
    b_sol = df.loc[(x,y,z)]['B_sol_z']
    
    return (b_tor + time_evolution(b_sol, t))*b_multiplier
	
init_pos, init_mom = create_initial_positions(10, position_field)

Main(
    geometry = "3Dcartesian",
    
    interpolation_order = 2,
    
    timestep = dt,
    simulation_time = Tsim,
    
    cell_length  = [dx,dy,dz],
    grid_length = [Lx,Ly,Lz],
    
    number_of_patches = [8,8,2],
    
    EM_boundary_conditions = [ ["periodic"],["periodic"],["periodic"] ],
    
    print_every = 1,

)

LoadBalancing(
    every = 20,
    cell_load = 1.,
    frozen_particle_load = 0.1
)


Species(
    name = "proton",
    position_initialization = init_pos,
    momentum_initialization = init_mom,
    # particles_per_cell = 8, 
    c_part_max = 1.0,
    # mass = 1836.0,
    mass = 1,
    charge = 1.0,
    # charge_density = 1,
    # mean_velocity = [0., 0.0, 0.0],
    # temperature = [T],
    pusher = "boris",
    boundary_conditions = [
    	["periodic", "periodic"],
    	["periodic", "periodic"],
    	["periodic", "periodic"],
    ],
)

Species(
    name = "electron",
    position_initialization = init_pos,
    momentum_initialization = init_mom,
    # particles_per_cell = 8, 
    c_part_max = 1.0,
    mass = 1.0,
    charge = -1.0,
    # charge_density = 1,
    # mean_velocity = [0., 0.0, 0.0],
    # temperature = [T],
    pusher = "boris",
    boundary_conditions = [
    	["periodic", "periodic"],
    	["periodic", "periodic"],
    	["periodic", "periodic"],
    ],
)

Checkpoints(
    dump_step = 0,
    dump_minutes = 0.0,
    exit_after_dump = False,
)

# DiagFields(
#     every = 4
# )

# DiagScalar(every = 1)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 1,
    time_average = 1,
    species = ["proton"],
    axes = [["x", 0, 30, 2*Lx/dx],
            ["y", 0, 30, 2*Ly/dy],
            ["z", 4, 6, 2]],
)

PrescribedField(
    field = "Bx_m",
    profile = get_bx
)

PrescribedField(
    field = "By_m",
    profile = get_by
)

PrescribedField(
    field = "Bz_m",
    profile = get_bz
)
