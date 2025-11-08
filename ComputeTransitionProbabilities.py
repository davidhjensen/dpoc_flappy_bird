"""ComputeTransitionProbabilities.py

Template to compute the transition probability matrix.

Dynamic Programming and Optimal Control
Fall 2025
Programming Exercise

Contact: Antonio Terpin aterpin@ethz.ch
Authors: Marius Baumann, Antonio Terpin

--
ETH Zurich
Institute for Dynamic Systems and Control
--
"""

import numpy as np
from Const import Const
from utils import *

def compute_transition_probabilities(C:Const) -> np.array:
    """Computes the transition probability matrix P.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L), where:
            - K is the size of the state space;
            - L is the size of the input space.
            - P[i,j,l] corresponds to the probability of transitioning
              from the state i to the state j when input l is applied.
    """
    P = np.zeros((C.K, C.K, C.L))

    for i, state in enumerate(C.state_space):
        for j, next_state in enumerate(C.state_space):
            for u, input in enumerate(C.input_space):
                
                # current state variables
                y = state[0]                        # height
                v = state[1]                        # velocity
                d = (0, *state[2:C.M+2])            # distance to obstacle i (indexing MATCHES i: d1 is at d[1])
                h = (0, *state[C.M+2:])             # height of gap i (indexing MATCHES i: h1 is at h[1])

                # next state variables
                y_n = next_state[0]                 # height
                v_n = next_state[1]                 # velocity
                d_n = (0, *next_state[2:C.M+2])     # distance to obstacle i (indexing MATCHES i: d1 is at d_n[1])
                h_n = (0, *next_state[C.M+2:])      # height of gap i (indexing MATCHES i: h1 is at h_n[1])

                # a few helpful variables that aren't in the state space
                # s - free distance 
                #   - difference between the width minus 1 (for the bird) and
                #     the sum of all distances minus 1 (for the shift of obstacles left)
                s = (C.X - 1) - (sum(d) - 1)    

                # whether or not a new obstacle can spawn
                can_spawn = s >= C.D_min    

                # m_min - smallest index with no assigned obstacle at time k
                try:
                    m_min = d.index(0, 2)           # look for 0s starting at d2 since d1 doesn't matter for this
                except:
                    m_min = C.M                     # if no zero is found, m_min = M


            # WORK THORUGH SHORTCUTS IN ORDER (y, v, d1, d2, etc) with collision first
            # NOTE: P is initialized to zeros, so continuing on the loop means the entry is "set" to zero

                # Collision - set probability to 0
                if is_collision(C, y, d[1], h[1]):
                    P[i,j,u] = 0
                    continue

                # y - vertical position
                #   - y(k+1) must be sum of current position (y) and velocity (v)
                #   - clipped into feasable range
                if y_n != min(max(y + v, 0), C.Y - 1):
                    P[i,j,u] = 0
                    continue

                # v - vertical velocity
                #   - v(k+1) must be in the range of current velocity plus input \pm deviation
                #   - clipped into feasable range
                v_min = (v + input - C.g) - C.V_dev
                v_max = (v + input - C.g) + C.V_dev
                v_potential = range(v_min, v_max + 1)
                if v_n not in v_potential:
                    P[i,j,u] = 0
                    continue

                # FOR d1(k) == 0:
                if d[1] == 0:

                    # d1 - distance to first obstacle
                    #    - d1(k+1) must be shifted d2(k)
                    if d_n[1] != d[2] - 1:
                        P[i,j,u] = 0
                        continue 

                    # di - distance between obstacles for i(k) = 3, 4, ... , m_min - 1, for m_min: dm_min(k) == 0
                    #    - indicies all must decrease by 1 from time k to k+1 for i(k) = 3, 4, ... , m_min - 1
                    #      For example, d4(k+1)should be d5(k)
                    if d_n[2:m_min-1] != d[3:m_min]:
                        P[i,m_min,u] = 0
                        continue

                    # di - distance between obstacles for i(k) = m_min, for m_min: dm_min(k) == 0
                    #    - d(m_min-1)(k+1) must either be in {0, s} if obstacle can spawn, or 0 if obstacle cannot
                    if (can_spawn and d_n[m_min-1] not in (0, s)) or (not can_spawn and d_n[m_min-1] != 0):
                        P[i,j,u] = 0
                        continue

                    # di - distance between obstacles for i(k) = m_min + 1, m_min + 2, ... , M, for m_min: dm_min(k) == 0
                    #    - they all must be zero
                    if sum(d_n[m_min:]) != 0:
                        P[i,j,u] = 0
                        continue

                    # hi - height of gap for each obstacle for i(k) = 2, 3, ... , m_min - 1, for m_min: dm_min(k) == 0
                    #    - indicies all must decrease by 1 from time k to k+1 for i(k) = 2, 3, ... , m_min - 1
                    #      i.e. h4(k+1) should be h5(k)
                    if h_n[1:m_min-1] != h[2:m_min]:
                        P[i,j,u] = 0
                        continue

                    # hi - height of gap for each obstacle for i(k) = m_min, for m_min: dm_min(k) == 0
                    #    - the height must be default if no object can spawn
                    if not can_spawn and h_n[m_min-1] != C.S_h[0]:
                        P[i,j,u] = 0
                        continue

                    # hi - height of gap for each obstacle for i(k) = m_min + 1, m_min + 2, ... , M, for m_min: dm_min(k) == 0
                    #    - the height must be the default height
                    tup_default = tuple((C.M - m_min)*[C.S_h[0]])   # a tuple of default heights with length M - m_min = d_n[m_min:]
                    if h_n[m_min:] != tup_default:
                        P[i,j,u] = 0
                        continue
                
                # FOR d1(k) != 0:
                else:

                    # d1 - distance to first obstacle
                    #    - d1(k+1) must be shifted d1(k)
                    if d_n[1] != d[1] - 1:
                        P[i,j,u] = 0
                        continue 

                    # di - distance between obstacles for i(k) = 2, 3, ... , m_min - 1, for m_min: dm_min(k) == 0
                    #    - distances must all be the same
                    if d_n[2:m_min] != d[2:m_min]:
                        P[i,j,u] = 0
                        continue

                    # di - distance between obstacles for i(k) = m_min, for m_min: dm_min(k) == 0
                    #    - dm_min(k+1) must either be in {0, s} if obstacle can spawn, or 0 if obstacle cannot
                    if m_min != C.M and (
                        (can_spawn and d_n[m_min] not in (0, s)) or
                        (not can_spawn and d_n[m_min] != 0)):
                        P[i,j,u] = 0
                        continue

                    # di - distance between obstacles for i(k) = m_min + 1, m_min + 2, ... , M, for m_min: dm_min(k) == 0
                    #    - they all must be zero
                    if m_min != C.M and sum(d_n[m_min+1:]) != 0:
                        P[i,j,u] = 0
                        continue

                    # hi - height of gap for each obstacle for i(k) = 1, 2, ... , m_min - 1, for m_min: dm_min(k) == 0
                    #    - the heights must all stay the same
                    if h_n[1:m_min] != h[1:m_min]:
                        P[i,j,u] = 0
                        continue

                    # hi - height of gap for each obstacle for i(k) = m_min, for m_min: dm_min(k) == 0
                    #    - the height must be default if no object can spawn
                    if not can_spawn and h_n[m_min] != C.S_h[0]:
                        P[i,j,u] = 0
                        continue

                    # hi - height of gap for each obstacle for i(k) = m_min + 1, m_min + 2, ... , M, for m_min: dm_min(k) == 0
                    #    - the height must be the default height
                    tup_default = tuple((C.M - m_min)*[C.S_h[0]])   # a tuple of default heights with length M - m_min = d_n[m_min:]
                    if m_min != C.M and h_n[m_min+1:] != tup_default:
                        P[i,j,u] = 0
                        continue
    
    return P