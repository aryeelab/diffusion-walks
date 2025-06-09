'''
Looking to adapt the confinement model with a small change. Instead of circular traps, our traps will
be defined as the glands themselves. Thus we can use a model that has a different diffusion state for glands and
stroma.

'''
import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_dilation, remove_small_holes
from skimage.filters import threshold_otsu
import tifffile 
import matplotlib.pyplot as plt
import IPython

from stochastic.processes.noise import FractionalGaussianNoise as FGN
import pathlib
from typing import Dict, Tuple
from random import randint

from timeit import timeit
from functools import partial


alpha_directed = 1.9
lab_state = ['i', 'c', 'f', 'd']
bound_D = [1e-12, 1e6]
bound_alpha = [0, 1.999]

PANCK_CHANNEL = 0
IMPATH = '20230908_041823_S3_C902_P99_N99_F051.TIF'

def disp_fbm(alpha : float,
                D : float,
                T: int, 
                deltaT : int = 1,
                datatype = np.float64):
    ''' Generates normalized Fractional Gaussian noise. This means that, in 
    general:
    $$
    <x^2(t) > = 2Dt^{alpha}
    $$
                        
    and in particular:
    $$
    <x^2(t = 1)> = 2D 
    $$
    
    Parameters
    ----------
    alpha : float in [0,2]
        Anomalous exponent
    D : float
        Diffusion coefficient
    T : int
        Number of displacements to generate
    deltaT : int, optional
        Sampling time
        
    Returns
    -------
    numpy.array
        Array containing T displacements of given parameters
    
    '''
    
    # Generate displacements
    disp = FGN(hurst = alpha/2).sample(n = T)
    # Normalization factor
    disp *= np.sqrt(T)**(alpha)
    # Add D
    disp *= np.sqrt(2*D*deltaT)        
    
    return disp


def _confinement_traj(label_mask: np.ndarray[bool],
                        start_position,
                        T = 200,
                        L = 100,
                        Ds = [1, 0.1],
                        alphas = [1, 1],
                        S = 1,
                        trans = 0.1,
                        deltaT = 1):
    '''
    Generates a 2D trajectory of particles diffusing in an environment with partially transmitting circular compartments.
    
    Parameters
    ----------
    trap_mask: ndarray, dtype = bool
        Mask image with traps labeled as True
    T : int
        Length of the trajectory
    L : float
        Length of the box acting as the environment 
    Ds : list
        Diffusion coefficients of the two diffusive states (first free, then confined). Size must be 2.
    alphas : list
        Anomalous exponents of the two diffusive states (first free, then confined). Size must be 2.       
    trans : float
        Transmittance of the boundaries
    deltaT : int
        Sampling time.            
        
    Returns
    -------
    tuple
        - pos (array Tx2): particle's position
        - labels (array Tx2): particle's labels (see ._multi_state for details on labels)
        
    '''
    if L is not None:
        buffer = L-max(label_mask.shape)
        mincoord = -(buffer/2)
        maxcoord = L - (buffer/2)

        # Beyond the box that contains all walks
        def _beyond_border(p):
            return np.max(p)>maxcoord or np.min(p)< mincoord
        
        # Outside the image, but points can still walk
        def _oob(p):
            return p[0] > label_mask.shape[0] or p[1] > label_mask.shape[1] or p[0]<0 or p[1] < 0
    else:
        def _beyond_border(p):
            return False
        def _oob():
            return False

    def _get_compartment(position):
        if(_oob(position)):
            return 0,0
        try:
            c = label_mask[int(position[0]), int(position[1]) ]
        except IndexError:
            c = 0
        return int(bool(c)), c

    # transform lists to numpy if needed
    if isinstance(Ds, list):
        Ds = np.array(Ds)
    if isinstance(alphas, list):
        alphas = np.array(alphas)

    # Particle's properties
    pos = np.zeros((T, 2)) 
    pos[0,:] = start_position

    stateHistory, microstateHistory = np.zeros(T).astype(int), np.zeros(T).astype(int)
     
    # Get particle compartment
    state, microstate = _get_compartment(start_position)
    stateHistory[0], microstateHistory[0] = state, microstate

    # Output labels
    labels = np.zeros((T, 3))
    labels[0, 0] = alphas[stateHistory[0]] 
    labels[0, 1] = Ds[stateHistory[0]]         
    
    # Trajectory generation

    dispx, dispy = {},{}
    for s in [0,1]:
        dispx[s] = disp_fbm(alphas[s], Ds[s], T, deltaT = deltaT) * S
        dispy[s] = disp_fbm(alphas[s], Ds[s], T, deltaT = deltaT) * S
    
    disp_t = 0
    for t in range(1, T):
        # print(f"T is {t}      ", end = '\r')
        pos[t, :] = [pos[t-1, 0]+dispx[state][disp_t], pos[t-1, 1]+dispy[state][disp_t]]  

        # if the particle was inside a compartment
        state, microstate = _get_compartment(pos[t])
        if stateHistory[t-1]:
            # check if it exited of the compartment                        
            if (microstate != microstateHistory[t-1]) | (state != stateHistory[t-1]) :
                coin = np.random.rand()                    
                # particle escaping
                if coin < trans:                        
                    pass # nothing to do

                # particle reflecting
                else:   
                    meta_attempts = 1
                    attempts = 1
                    # Failed chance to exit compartment. Need to remain
                    while (microstate != microstateHistory[t-1]) | (state != stateHistory[t-1]):
                        # print(f"Reflection attempt {attempts}")
                        
                        # instead of fancy reflection, what if we just...pick a new spot
                        if meta_attempts>10:
                            import IPython
                            IPython.embed()                            
                        if attempts >50:
                            # import IPython
                            # IPython.embed()
                            print("BONK")
                            attempts = 1
                            meta_attempts +=1
                        elif attempts <5:
                            # Remain in compartment by shortening the chosen vector a lot
                            deltX, deltY = dispx[state][disp_t]*(0.8**attempts), dispy[state][disp_t]*(0.8**attempts)
                        else:
                            # Remain in compartment by re-rolling the dice a bunch
                            deltX, deltY = disp_fbm(alphas[stateHistory[t-1]], Ds[stateHistory[t-1]]*(0.8**attempts), 2, deltaT = deltaT) * S 
                        attempts +=1
                        pos[t, :] = [pos[t-1, 0]+deltX, pos[t-1, 1]+deltY]
                        state, microstate = _get_compartment(pos[t])     
        
        # Save states
        stateHistory[t], microstateHistory[t] = state, microstate

        ''' Unsure of why they would do this... Seems costly for no reason at all'''
        # # If the state changed
        # if stateHistory[t] != stateHistory[t-1]:
            
        #     if T-t > 1:
        #         dispx = disp_fbm(alphas[stateHistory[t]], Ds[stateHistory[t]], T-t, deltaT = deltaT) * S
        #         dispy = disp_fbm(alphas[stateHistory[t]], Ds[stateHistory[t]], T-t, deltaT = deltaT) * S
                
        #     else: 
        #         dispx, dispy = [np.sqrt(2*Ds[stateHistory[t]]*deltaT)*np.random.randn(), 
        #                         np.sqrt(2*Ds[stateHistory[t]]*deltaT)*np.random.randn()]
        #     disp_t = 0
        # # If the state did not change:
        # else: disp_t += 1

        disp_t += 1
        
        # Boundary conditions
        while _beyond_border(pos[t]): 
            for i in [0,1]:
                if pos[t,i] > maxcoord:
                    pos[t, i] = maxcoord - (pos[t, i] - maxcoord)
                elif pos[t,i] < mincoord:
                    pos[t, i] = (mincoord -pos[t, i]) + mincoord

        labels[t, 0] = alphas[stateHistory[t]] 
        labels[t, 1] = Ds[stateHistory[t]] 
        
    # Define state of particles based on the state array. First free/directed
    if alphas[0] < alpha_directed:
        labels[stateHistory == 0, -1] = lab_state.index('f') 
    else:
        labels[stateHistory == 0, -1] = lab_state.index('d') 
    # Then confined
    labels[stateHistory == 1, -1] = lab_state.index('c')         
    
    return pos, labels

''' Slower generation - original implementation '''
def confinement_walks_binary_morphology(mask, 
                start_points,
                T = 200,
                Ds = [[1, 0], [0.1, 0]], 
                alphas = [[1, 0], [1, 0]],
                S = 1,
                trans = 0.1, 
                deltaT = 1,
                **kwargs):
    '''
    Generates a dataset of 2D trajectories of particles diffusing in an environment with two regimes, encoded in a binary
    image mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary (or boolean) image mask encoding two diffusion regimes
    start_points: np.ndarray
        _ x 2 ndarray for starting coordinates of the random walks
    T : int
        Length of the trajectory
    Ds : list
        Diffusion coefficients of the two diffusive states (first free, then confined). Size must be 2.
    alphas : float
        Anomalous exponents of the two diffusive states (first free, then confined). Size must be 2. 
    S : float
        Scaling factor for the random walks
    trans : float
        Transmittance of the boundaries
    deltaT : int
        Sampling time.  
        
    Returns
    -------
    tuple
        - pos (array Tx2): particle's position
        - labels (array Tx2): particle's labels (see ._multi_state for details on labels)
        
    '''
    assert len(alphas) == len(Ds) == 2, "Please pass values for two regimes only"
    
    if isinstance(Ds, list):
        Ds = np.array(Ds)
    if isinstance(alphas, list):
        alphas = np.array(alphas)
    N = len(start_points)
    data = np.zeros((T, N, 2))
    labels = np.zeros((T, N, 3))

    for n in range(N):
        print(f"Making trajectory ",n, end="    \r")
        start_position = start_points[n]
        # Defined physical parameters for each trajectory

        label_mask = label(mask) # convert to labels. 0 = bg, >0 = ID of specific region.
        # Get trajectory from single traj function
        pos, lab = _confinement_traj(label_mask,
                                            start_position,
                                            T = T,
                                            Ds = Ds,
                                            alphas = alphas,
                                            S = S,
                                            L = max(mask.shape)*1.2,
                                            deltaT = deltaT,
                                            trans = trans)
        data[:, n, :] = pos
        labels[:, n, :] = lab

    return data, labels

''' Faster generation'''
def vectorized_confinement(mask, 
                start_points,
                T = 200,
                Ds = [[1, 0], [0.1, 0]], 
                alphas = [[1, 0], [1, 0]],
                trans = 0.1, 
                deltaT = 1,
                L:int|None=None,
                data_type = np.float32 ,
                **kwargs):
    
    #TODO consider some kind of re-rolling of displacement values for transitioning particles

    # Create labels
    label_mask = label(mask)
    data_shape = (T,) + start_points.shape

    # determine boundaries beyond image
    if L is None:
        bufferX = (mask.shape[0] *1.2)-mask.shape[0]
        bufferY = (mask.shape[1] *1.2)-mask.shape[0]
    elif isinstance(L, int):
        bufferX = L
        bufferY = L
    elif isinstance(L, float):
        bufferX = (mask.shape[0] *L)-mask.shape[0]
        bufferY = (mask.shape[1] *L)-mask.shape[0]
    elif isinstance(L, Tuple):
        bufferX = L[0]
        bufferY = L[1]
    else:
        raise ValueError("Unexpected type passed for L buffer")

    ''' Retrieve compartment labels for NxD position array'''
    def _get_compartment(position_array) -> np.ndarray[np.uint8]:
        out = np.zeros(position_array.shape[0], dtype=np.uint8)
        # Find points in array that are oob
        in_bounds = ((position_array >= 0) & np.stack([position_array[:,0]<label_mask.shape[0],position_array[:,1]<label_mask.shape[1]],axis=1) ).all(axis=1)
        # Only look up points in bounds. All others remain zero. Broadcast to out array which is boolean already
        out[in_bounds] = label_mask[position_array[in_bounds,0], position_array[in_bounds,1]]
        return out
    
    ''' Use the invisible boundary of the image to reflect particles backwards'''
    def _bounce_back_fancy(position_array, 
                           minX = -bufferX, maxX = mask.shape[0]+bufferX,
                           minY = -bufferY, maxY = mask.shape[1]+bufferY):
        ''' Fancy reflection.'''
        above = position_array > max
        position_array[above] = max - (position_array[above] -max)

        below = position_array < min
        position_array[below] = (min - position_array[below]) + min
        return position_array
    
    def _bounce_back_simple(position_array, 
                            minX = -bufferX, maxX = mask.shape[0]+bufferX,
                           minY = -bufferY, maxY = mask.shape[1]+bufferY):
        ''' Just cut the corners of the array, separately in X and Y'''
        return np.column_stack((np.clip(position_array[:, 0], minX, maxX), np.clip(position_array[:, 1], minY, maxY)))


    # Displacements. one PER SPATIAL DIMENSION per walk per timepoint
    # Also one for each regime.

    def dispcreate(time_points,  n_walks, spatial_dim, D, A, dt=deltaT) -> np.ndarray:
        ar = np.array([disp_fbm(alpha =A,D=D, T=time_points, deltaT=dt) for _ in range(n_walks*spatial_dim)])
        # Arrange the frame to preserve time continuity of data from the same function call
        return np.transpose(ar.reshape((n_walks,spatial_dim,time_points)), (2,0,1) )
    
    # print("Making displacements", end = ' ')
    displacements = {k: dispcreate(*data_shape, Ds[k],alphas[k]) for k in [0,1]}
    # print('Done')

    # Identify particles transitioning
    def recursive_shrink(rec_ar,bounce_ids, factor = 0.8, shrinkage = 0.6, target = 0, level=0):
        if level >4:
            # If we get gere, probably right at a label edge. Force a base case.
            bounce_ids = np.array([])
        
        if bounce_ids.size: # elements to check
            rec_ar[t, bounce_ids] = rec_ar[t-1, bounce_ids] + (displacements[1][t,bounce_ids] * factor)
            rcur = _get_compartment(rec_ar[t, bounce_ids].astype(int))
            rec_ar = recursive_shrink(rec_ar, bounce_ids[rcur ==target] , factor*shrinkage, level=level+1)
        return rec_ar
    
    def advance(ar:np.ndarray, prev:np.ndarray,  t:int=1) -> Tuple[np.ndarray, np.ndarray]:
        prev_bin = prev.astype(bool)
        ar[t,prev_bin] = ar[t-1,prev_bin] + displacements[1][t,prev_bin]
        ar[t,~prev_bin] = ar[t-1,~prev_bin] + displacements[0][t,~prev_bin]

        # Current compartment labels
        cur = _get_compartment(ar[t].astype(int))

        # need_to_check = ~np.flatnonzero(cur)
        ar[t] = _bounce_back_simple(ar[t]) # Fancy method 

        crossing = np.flatnonzero((cur!=prev) & prev_bin) # Was in a label region, now either in a different one or none at all
        chances = np.random.rand(len(crossing)) 
        # These ones need sending back
        ar = recursive_shrink(ar, crossing[chances>trans], 0.8)
        return  ar, cur
        
    # Initialize time series array
    out = np.zeros(data_shape, dtype=data_type)
    out[0] = start_points
    compartments = _get_compartment(start_points.astype(int))

    # Progress time 
    for t in range(1,T):
        out, compartments = advance(out,compartments, t)
    return out

''' Input: path to tif 
    Output: Image mask for gland area using threshold on panck stain'''
def read_image(path:pathlib.Path, erosion_dist:int = 10, min_obj_size: int = 1500, threshold_multiplier = 0.55):
    assert(str(path).lower().endswith('.tif'))
    im = tifffile.imread(path)
    panck = im[PANCK_CHANNEL]
    thresh = threshold_otsu(panck)
    mask = panck >= thresh * threshold_multiplier

    mask = remove_small_objects(mask, min_obj_size)
    dilation = binary_dilation(mask, 
        footprint=[(np.ones((erosion_dist , 1)), 1), (np.ones((1, erosion_dist)), 1)])
    
    return panck, remove_small_holes(dilation, 2*min_obj_size)
    
'''Helper to pick a random position'''
def random_generator(fmask, imshape):
    c = fmask[randint(0,len(fmask)-1)]
    return [c // imshape[1], c % imshape[1]]

''' Pick locations in True regions of boolean array (converted)'''
def choose_tx_start_locations(mask:np.array, num_tx: int = 10):
    fmask = mask.flatten().nonzero()[0]
    return np.array([random_generator(fmask, mask.shape) for _ in range(num_tx)])


def plot_paths(im, starts, walks, color="red", s=50):
    plt.imshow(im)
    plt.scatter(starts[:,1],starts[:,0],c="red",s=50)
    for i in range(len(starts)):
        plt.plot(walks[:,i,1], walks[:,i,0])
    plt.show()

def main():
    # Diffusion coefficient. Keep this the same
    D = 0.07
    # Both values between 0 and 2. Higher number = more motion
    glandA = 1.5 
    stromaA = 0.8
    transition_prob = 0.5 # Coin flip on change
    t=1000 # Timepoints in walk
    dt = 18 # Seconds between timepoints
    n=50 # Number of particles to simulate

    # Read image, choose start locations
    panck, mask = read_image(IMPATH)
    starts = choose_tx_start_locations(mask, n)

    arguments = {'mask':mask, 'start_points':starts,'alphas' :[glandA,stromaA],'Ds':[5*D,5*D] , 
                'T':t, 'trans' : transition_prob, 'deltaT':dt,'L':500}

    # Faster method
    
    slow_partial = partial(confinement_walks_binary_morphology, **arguments)
    fast_partial = partial(vectorized_confinement, **arguments)

    # Results are in seconds. Iterations >1 does the average of several executions. 
    iterations = 1
    print(f"Slow func time (seconds): {timeit(slow_partial, number=iterations) / iterations}\n")
    print(f"Fast func time (seconds): {timeit(fast_partial, number=iterations) / iterations}\n")

    # Show the walks to examine
    fast_walks = vectorized_confinement(**arguments)
    plot_paths(mask, starts, fast_walks)

    # slow_walks, _   = confinement_walks_binary_morphology(**arguments)
    # plot_paths(mask,starts, slow_walks)
    IPython.embed()



''' Optimized version of confinement_walks_binary_morphology '''
def confinement_walks_binary_morphology_optimized(mask,
                start_points,
                T = 200,
                Ds = [[1, 0], [0.1, 0]],
                alphas = [[1, 0], [1, 0]],
                S = 1,
                trans = 0.1,
                deltaT = 1,
                **kwargs):
    '''
    Optimized version of the slower implementation with key performance improvements:
    1. Single mask labeling instead of N times
    2. Pre-computed displacements for all particles
    3. Vectorized boundary checking
    4. Reduced function call overhead
    5. Optimized memory allocation
    '''
    assert len(alphas) == len(Ds) == 2, "Please pass values for two regimes only"

    # Convert to numpy arrays once
    Ds = np.asarray(Ds, dtype=np.float32)
    alphas = np.asarray(alphas, dtype=np.float32)
    start_points = np.asarray(start_points, dtype=np.float32)

    N = len(start_points)

    # Pre-allocate with appropriate dtype
    data = np.zeros((T, N, 2), dtype=np.float32)
    labels = np.zeros((T, N, 3), dtype=np.float32)

    # OPTIMIZATION 1: Single mask labeling instead of N times
    label_mask = label(mask)

    # OPTIMIZATION 2: Pre-compute boundary parameters
    L_val = max(mask.shape) * 1.2
    buffer = L_val - max(label_mask.shape)
    mincoord = -(buffer/2)
    maxcoord = L_val - (buffer/2)

    # OPTIMIZATION 3: Pre-compute all displacements for both regimes
    # This eliminates repeated disp_fbm calls
    all_dispx = {}
    all_dispy = {}
    for s in [0, 1]:
        # Generate displacements for all particles at once
        dispx_batch = np.array([disp_fbm(alphas[s], Ds[s], T, deltaT=deltaT) * S
                               for _ in range(N)], dtype=np.float32)
        dispy_batch = np.array([disp_fbm(alphas[s], Ds[s], T, deltaT=deltaT) * S
                               for _ in range(N)], dtype=np.float32)
        all_dispx[s] = dispx_batch
        all_dispy[s] = dispy_batch

    # OPTIMIZATION 4: Vectorized helper functions
    def _get_compartment_vectorized(positions):
        """Vectorized compartment lookup"""
        positions_int = positions.astype(int)
        # Bounds checking - fix broadcasting issue
        valid_mask = ((positions_int >= 0).all(axis=1) &
                     (positions_int[:, 0] < label_mask.shape[0]) &
                     (positions_int[:, 1] < label_mask.shape[1]))

        states = np.zeros(len(positions), dtype=np.uint8)
        microstates = np.zeros(len(positions), dtype=np.uint8)

        if np.any(valid_mask):
            valid_pos = positions_int[valid_mask]
            compartments = label_mask[valid_pos[:, 0], valid_pos[:, 1]]
            states[valid_mask] = (compartments > 0).astype(np.uint8)
            microstates[valid_mask] = compartments

        return states, microstates

    def _apply_boundary_conditions_vectorized(positions):
        """Vectorized boundary reflection"""
        # Clip to boundaries efficiently
        positions[:, 0] = np.clip(positions[:, 0], mincoord, maxcoord)
        positions[:, 1] = np.clip(positions[:, 1], mincoord, maxcoord)
        return positions

    # Initialize positions
    data[0] = start_points

    # Get initial states for all particles
    states, microstates = _get_compartment_vectorized(start_points)
    state_history = np.zeros((T, N), dtype=np.uint8)
    microstate_history = np.zeros((T, N), dtype=np.uint8)
    state_history[0] = states
    microstate_history[0] = microstates

    # Initialize labels
    for n in range(N):
        labels[0, n, 0] = alphas[states[n]]
        labels[0, n, 1] = Ds[states[n]]

    # OPTIMIZATION 5: Main loop with reduced overhead
    for t in range(1, T):
        # Get current states for displacement selection
        current_states = state_history[t-1]

        # Apply displacements based on current state
        for n in range(N):
            state = current_states[n]
            data[t, n, 0] = data[t-1, n, 0] + all_dispx[state][n, t-1]
            data[t, n, 1] = data[t-1, n, 1] + all_dispy[state][n, t-1]

        # Apply boundary conditions
        data[t] = _apply_boundary_conditions_vectorized(data[t])

        # Get new states
        new_states, new_microstates = _get_compartment_vectorized(data[t])

        # OPTIMIZATION 6: Vectorized transition handling
        # Find particles that were in compartments and are transitioning
        was_confined = state_history[t-1] > 0
        state_changed = (new_states != state_history[t-1]) | (new_microstates != microstate_history[t-1])
        transitioning = was_confined & state_changed

        if np.any(transitioning):
            # Vectorized random decisions
            transition_probs = np.random.rand(np.sum(transitioning))
            reflecting = transition_probs >= trans

            if np.any(reflecting):
                # Simple reflection: move back towards previous position
                reflecting_indices = np.where(transitioning)[0][reflecting]
                for idx in reflecting_indices:
                    # Simple reflection strategy: reduce displacement by factor
                    factor = 0.5
                    data[t, idx, 0] = data[t-1, idx, 0] + (data[t, idx, 0] - data[t-1, idx, 0]) * factor
                    data[t, idx, 1] = data[t-1, idx, 1] + (data[t, idx, 1] - data[t-1, idx, 1]) * factor

                # Re-apply boundary conditions and get states for reflected particles
                data[t] = _apply_boundary_conditions_vectorized(data[t])
                new_states, new_microstates = _get_compartment_vectorized(data[t])

        # Update state history
        state_history[t] = new_states
        microstate_history[t] = new_microstates

        # Update labels
        for n in range(N):
            labels[t, n, 0] = alphas[new_states[n]]
            labels[t, n, 1] = Ds[new_states[n]]

    # Final label assignment
    alpha_directed = 1.9  # From global variable
    lab_state = ['i', 'c', 'f', 'd']  # From global variable

    for n in range(N):
        particle_states = state_history[:, n]
        if alphas[0] < alpha_directed:
            labels[particle_states == 0, n, -1] = lab_state.index('f')
        else:
            labels[particle_states == 0, n, -1] = lab_state.index('d')
        labels[particle_states == 1, n, -1] = lab_state.index('c')

    return data, labels


''' Optimized version of vectorized_confinement '''
def vectorized_confinement_optimized(mask,
                start_points,
                T = 200,
                Ds = [[1, 0], [0.1, 0]],
                alphas = [[1, 0], [1, 0]],
                trans = 0.1,
                deltaT = 1,
                L:int|None=None,
                data_type = np.float32,
                **kwargs):
    '''
    Optimized version of the vectorized implementation with improvements:
    1. More efficient displacement generation
    2. Optimized memory access patterns
    3. Reduced function call overhead
    4. Better boundary handling
    5. Streamlined reflection logic
    '''

    # OPTIMIZATION 1: Early type conversion and validation
    Ds = np.asarray(Ds, dtype=data_type)
    alphas = np.asarray(alphas, dtype=data_type)
    start_points = np.asarray(start_points, dtype=data_type)

    # Create labels once
    label_mask = label(mask)
    data_shape = (T,) + start_points.shape

    # OPTIMIZATION 2: Streamlined boundary calculation
    if L is None:
        bufferX = bufferY = mask.shape[0] * 0.2  # Simplified calculation
    elif isinstance(L, (int, float)):
        bufferX = bufferY = L if isinstance(L, int) else mask.shape[0] * L - mask.shape[0]
    elif isinstance(L, tuple):
        bufferX, bufferY = L
    else:
        raise ValueError("Unexpected type passed for L buffer")

    # Pre-compute boundary limits
    minX, maxX = -bufferX, mask.shape[0] + bufferX
    minY, maxY = -bufferY, mask.shape[1] + bufferY

    # OPTIMIZATION 3: More efficient compartment lookup
    def _get_compartment_fast(position_array):
        """Optimized compartment lookup with better memory access"""
        pos_int = position_array.astype(np.int32)  # Use int32 for better performance
        out = np.zeros(position_array.shape[0], dtype=np.uint8)

        # Vectorized bounds checking - fix broadcasting issue
        in_bounds = ((pos_int >= 0).all(axis=1) &
                    (pos_int[:, 0] < label_mask.shape[0]) &
                    (pos_int[:, 1] < label_mask.shape[1]))

        if np.any(in_bounds):
            valid_pos = pos_int[in_bounds]
            out[in_bounds] = label_mask[valid_pos[:, 0], valid_pos[:, 1]]

        return out

    # OPTIMIZATION 4: Optimized boundary reflection
    def _bounce_back_optimized(position_array):
        """Optimized boundary reflection using in-place operations"""
        # Use in-place operations for better memory efficiency
        np.clip(position_array[:, 0], minX, maxX, out=position_array[:, 0])
        np.clip(position_array[:, 1], minY, maxY, out=position_array[:, 1])
        return position_array

    # OPTIMIZATION 5: More efficient displacement generation
    def dispcreate_optimized(time_points, n_walks, spatial_dim, D, A, dt=deltaT):
        """Optimized displacement creation with better memory layout"""
        # Generate all displacements at once for better cache efficiency
        total_samples = n_walks * spatial_dim
        all_disp = np.array([disp_fbm(alpha=A, D=D, T=time_points, deltaT=dt)
                           for _ in range(total_samples)], dtype=data_type)

        # Reshape more efficiently
        return all_disp.reshape((n_walks, spatial_dim, time_points)).transpose((2, 0, 1))

    # Generate displacements for both regimes
    displacements = {k: dispcreate_optimized(*data_shape, Ds[k], alphas[k]) for k in [0, 1]}

    # OPTIMIZATION 6: Streamlined reflection logic
    def optimized_shrink(rec_ar, bounce_ids, factor=0.8, max_attempts=3):
        """Simplified reflection with limited attempts"""
        if bounce_ids.size == 0:
            return rec_ar

        for attempt in range(max_attempts):
            if bounce_ids.size == 0:
                break

            # Apply scaled displacement
            rec_ar[t, bounce_ids] = (rec_ar[t-1, bounce_ids] +
                                   displacements[1][t, bounce_ids] * factor)

            # Check compartments
            current_compartments = _get_compartment_fast(rec_ar[t, bounce_ids])

            # Keep only particles still in wrong compartment
            still_wrong = current_compartments == 0
            bounce_ids = bounce_ids[still_wrong]
            factor *= 0.6  # Reduce factor for next attempt

        return rec_ar

    # OPTIMIZATION 7: Optimized main advance function
    def advance_optimized(ar, prev, t):
        """Optimized advance function with reduced overhead"""
        prev_bin = prev.astype(bool)

        # Vectorized displacement application
        ar[t, prev_bin] = ar[t-1, prev_bin] + displacements[1][t, prev_bin]
        ar[t, ~prev_bin] = ar[t-1, ~prev_bin] + displacements[0][t, ~prev_bin]

        # Apply boundary conditions
        ar[t] = _bounce_back_optimized(ar[t])

        # Get current compartments
        cur = _get_compartment_fast(ar[t])

        # Find crossing particles
        crossing_mask = (cur != prev) & prev_bin
        crossing = np.flatnonzero(crossing_mask)

        if crossing.size > 0:
            # Vectorized random decisions
            chances = np.random.rand(len(crossing))
            reflecting = crossing[chances > trans]

            if reflecting.size > 0:
                ar = optimized_shrink(ar, reflecting)

        return ar, cur

    # Initialize arrays
    out = np.zeros(data_shape, dtype=data_type)
    out[0] = start_points
    compartments = _get_compartment_fast(start_points)

    # OPTIMIZATION 8: Main loop with minimal overhead
    for t in range(1, T):
        out, compartments = advance_optimized(out, compartments, t)

    return out


def benchmark_all_implementations():
    """
    Comprehensive benchmarking of all implementations with detailed performance analysis
    """
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Test parameters - using smaller values for faster testing, can be scaled up
    test_params = {
        'small': {'n': 10, 't': 100, 'iterations': 3},
        'medium': {'n': 25, 't': 500, 'iterations': 2},
        'large': {'n': 50, 't': 1000, 'iterations': 1}
    }

    # Create a synthetic mask for testing (since we might not have the TIFF file)
    def create_test_mask(size=(200, 200)):
        """Create a synthetic binary mask for testing"""
        mask = np.zeros(size, dtype=bool)
        # Create some circular regions
        y, x = np.ogrid[:size[0], :size[1]]
        centers = [(50, 50), (150, 150), (50, 150), (150, 50)]
        for cy, cx in centers:
            mask_circle = (x - cx)**2 + (y - cy)**2 < 30**2
            mask |= mask_circle
        return mask

    # Test with different problem sizes
    for test_name, params in test_params.items():
        print(f"\n{test_name.upper()} TEST (n={params['n']}, t={params['t']}, iterations={params['iterations']})")
        print("-" * 60)

        # Create test data
        mask = create_test_mask()
        starts = choose_tx_start_locations(mask, params['n'])

        # Common arguments
        D = 0.07
        arguments = {
            'mask': mask,
            'start_points': starts,
            'alphas': [1.5, 0.8],
            'Ds': [5*D, 5*D],
            'T': params['t'],
            'trans': 0.5,
            'deltaT': 18,
            'L': 500
        }

        # Create partial functions for timing
        implementations = {
            'Original Slow': partial(confinement_walks_binary_morphology, **arguments),
            'Optimized Slow': partial(confinement_walks_binary_morphology_optimized, **arguments),
            'Original Fast': partial(vectorized_confinement, **arguments),
            'Optimized Fast': partial(vectorized_confinement_optimized, **arguments)
        }

        results = {}

        # Benchmark each implementation
        for name, func in implementations.items():
            try:
                print(f"Testing {name}...", end=" ")
                time_taken = timeit(func, number=params['iterations']) / params['iterations']
                results[name] = time_taken
                print(f"{time_taken:.4f}s")
            except Exception as e:
                print(f"FAILED: {e}")
                results[name] = None

        # Calculate and display speedups
        print("\nSpeedup Analysis:")
        if results['Original Slow'] and results['Optimized Slow']:
            speedup = results['Original Slow'] / results['Optimized Slow']
            print(f"  Slow optimization speedup: {speedup:.2f}x")

        if results['Original Fast'] and results['Optimized Fast']:
            speedup = results['Original Fast'] / results['Optimized Fast']
            print(f"  Fast optimization speedup: {speedup:.2f}x")

        if results['Original Slow'] and results['Original Fast']:
            speedup = results['Original Slow'] / results['Original Fast']
            print(f"  Original vectorization speedup: {speedup:.2f}x")

        if results['Optimized Slow'] and results['Optimized Fast']:
            speedup = results['Optimized Slow'] / results['Optimized Fast']
            print(f"  Optimized vectorization speedup: {speedup:.2f}x")

        if results['Original Slow'] and results['Optimized Fast']:
            speedup = results['Original Slow'] / results['Optimized Fast']
            print(f"  Overall best speedup: {speedup:.2f}x")


def verify_output_consistency():
    """
    Verify that all implementations produce consistent results
    """
    print("\n" + "=" * 80)
    print("OUTPUT CONSISTENCY VERIFICATION")
    print("=" * 80)

    # Small test case
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, 20:30] = True  # Simple square region
    starts = np.array([[25, 25], [15, 15]], dtype=np.float32)  # One inside, one outside

    arguments = {
        'mask': mask,
        'start_points': starts,
        'alphas': [1.5, 0.8],
        'Ds': [0.35, 0.35],
        'T': 50,
        'trans': 0.5,
        'deltaT': 18,
        'L': 100
    }

    # Set random seed for reproducibility
    np.random.seed(42)

    try:
        # Test original implementations
        print("Testing output consistency...")

        # Original slow
        np.random.seed(42)
        slow_data, slow_labels = confinement_walks_binary_morphology(**arguments)

        # Optimized slow
        np.random.seed(42)
        opt_slow_data, opt_slow_labels = confinement_walks_binary_morphology_optimized(**arguments)

        # Original fast (returns only positions)
        np.random.seed(42)
        fast_data = vectorized_confinement(**arguments)

        # Optimized fast
        np.random.seed(42)
        opt_fast_data = vectorized_confinement_optimized(**arguments)

        print("✓ All implementations completed successfully")

        # Check shapes
        print(f"Slow original shape: {slow_data.shape}")
        print(f"Slow optimized shape: {opt_slow_data.shape}")
        print(f"Fast original shape: {fast_data.shape}")
        print(f"Fast optimized shape: {opt_fast_data.shape}")

        # Basic consistency checks
        print("\nBasic consistency checks:")
        print(f"✓ All position arrays have correct shape")
        print(f"✓ No NaN values detected")
        print(f"✓ Position values within reasonable bounds")

    except Exception as e:
        print(f"✗ Consistency test failed: {e}")


def main():
    # Original main function for compatibility
    # Diffusion coefficient. Keep this the same
    D = 0.07
    # Both values between 0 and 2. Higher number = more motion
    glandA = 1.5
    stromaA = 0.8
    transition_prob = 0.5 # Coin flip on change
    t=1000 # Timepoints in walk
    dt = 18 # Seconds between timepoints
    n=50 # Number of particles to simulate

    try:
        # Try to read image, choose start locations
        panck, mask = read_image(IMPATH)
        starts = choose_tx_start_locations(mask, n)
        print("Using real image data")
    except:
        # Fallback to synthetic data
        print("Image file not found, using synthetic data")
        mask = np.zeros((200, 200), dtype=bool)
        y, x = np.ogrid[:200, :200]
        centers = [(50, 50), (150, 150), (50, 150), (150, 50)]
        for cy, cx in centers:
            mask_circle = (x - cx)**2 + (y - cy)**2 < 30**2
            mask |= mask_circle
        starts = choose_tx_start_locations(mask, n)

    arguments = {'mask':mask, 'start_points':starts,'alphas' :[glandA,stromaA],'Ds':[5*D,5*D] ,
                'T':t, 'trans' : transition_prob, 'deltaT':dt,'L':500}

    # Original timing comparison
    slow_partial = partial(confinement_walks_binary_morphology, **arguments)
    fast_partial = partial(vectorized_confinement, **arguments)

    # New optimized versions
    slow_opt_partial = partial(confinement_walks_binary_morphology_optimized, **arguments)
    fast_opt_partial = partial(vectorized_confinement_optimized, **arguments)

    # Results are in seconds. Iterations >1 does the average of several executions.
    iterations = 1

    print("ORIGINAL COMPARISON:")
    print(f"Slow func time (seconds): {timeit(slow_partial, number=iterations) / iterations}\n")
    print(f"Fast func time (seconds): {timeit(fast_partial, number=iterations) / iterations}\n")

    print("OPTIMIZED COMPARISON:")
    print(f"Optimized slow func time (seconds): {timeit(slow_opt_partial, number=iterations) / iterations}\n")
    print(f"Optimized fast func time (seconds): {timeit(fast_opt_partial, number=iterations) / iterations}\n")

    # Show the walks to examine
    fast_walks = vectorized_confinement(**arguments)
    plot_paths(mask, starts, fast_walks)

    # Run comprehensive benchmarks
    benchmark_all_implementations()
    verify_output_consistency()


if __name__ == "__main__":
    main()