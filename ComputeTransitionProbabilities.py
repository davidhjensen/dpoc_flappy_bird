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

    # probability tensor to return
    P = np.zeros((C.K, C.K, C.L))

    # constants
    M = C.M
    d_min = C.D_min
    default_height = C.S_h[0]
    n_heights = len(C.S_h)

    # calculate possible velocity transitions
    v_next_cache = {}
    for v in C.S_v:
        for u in C.input_space:
            v_base = v + u - C.g
            if u == C.U_strong:
                v_pot = np.clip([v_base + v_dev for v_dev in C.W_v], -C.V_max, C.V_max)
                unique, counts = np.unique(v_pot, return_counts=True)
                probs = counts / len(C.W_v)             # store values and corrisponding probabilities
            else:
                unique = [np.clip(v_base, -C.V_max, C.V_max)]
                probs = [1.0]
            v_next_cache[(v,u)] = list(zip(unique, probs))

    N_dense = compute_total_possible_states(C)
    unfiltered_to_filtered = -np.ones(N_dense, dtype=int)

    for sparse_i, s in enumerate(C.state_space):
        dense_i = my_state_to_index(s, C)
        unfiltered_to_filtered[dense_i] = sparse_i

    # convert statespace to numpy array
    np_ss = np.array(C.state_space)

    # narrow down to only non-collision states
    non_col_mask = ~((np_ss[:, 2] == 0) & ~(np.abs(np_ss[:, 0] - np_ss[:, M + 2]) <= (C.G - 1) // 2))
    non_col_idx = np.nonzero(non_col_mask)[0]
    non_col_states = np_ss[non_col_mask, :]
    
    # calculate deterministic values
    y_vec = non_col_states[:, 0]
    v_vec = non_col_states[:, 1]
    d_vec = non_col_states[:, 2:M+2]
    h_vec = non_col_states[:, M+2:]
    
    # next y value
    yn_vec = np.clip(y_vec + v_vec, 0, C.Y - 1)

    # free space
    free_space_vec = (C.X - 1) - (d_vec.sum(axis=1) - 1)

    # true where spawn can happen
    spawn_mask = free_space_vec >= d_min

    # true when passing
    is_passing_mask = d_vec[:, 0] == 0

    # probability of spawning
    pspawn_vec = np.clip((free_space_vec - (d_min - 1)) / (C.X - d_min), 0, 1)

    # m where spawn occurs or -1 if no spawn occurs
    d_zeros = (d_vec[:, 1:] == 0)                   # skip first column (d0)
    d_first_zero_idx = np.argmax(d_zeros, axis=1)
    d_had_zero = d_zeros.any(axis=1)
    m_min_vec = np.where(d_had_zero, d_first_zero_idx + 1, M)
    space_spawn_mask = d_had_zero | is_passing_mask

    # intermediate next distances and heights
    dn_pass_vec = np.concatenate([d_vec[:,1:2] - 1, d_vec[:, 2:], np.zeros((d_vec.shape[0], 1))], axis=1)
    dn_drift_vec = np.concatenate([d_vec[:,0:1] - 1, d_vec[:, 1:]], axis=1)
    dn_int_vec = np.where(d_vec[:, 0:1] == 0, dn_pass_vec, dn_drift_vec)
    hn_pass_vec = np.concatenate([h_vec[:, 1:], np.full(h_vec[:,0:1].shape, default_height)], axis=1)
    hn_int_vec = np.where((d_vec[:, 0:1] == 0), hn_pass_vec, h_vec)
    
    # for l, u in enumerate(C.input_space):

    #     v_pot_vec = v_next_cache  # list of (v_n, p_flap_obs)

    #     for w_spawn_obs in (0, 1):

    #         p_spawn_obs_vec = pspawn_vec if w_spawn_obs else (1.0 - pspawn_vec)
    #         p_spawn_obs_vec[~spawn_mask & w_spawn_obs] = 0

    #         if w_spawn_obs:
    #             height_choices = C.S_h
    #             p_height_obs = 1.0 / len(height_choices)
    #         else:
    #             height_choices = [default_height]
    #             p_height_obs = 1.0

    #         for this_h in height_choices:

    #             # build dn_vec and hn_vec here (already fixed with rows/cols)
    #             # dn_vec shape (N, M)
    #             # hn_vec shape (N, M)
    #             dn_vec = dn_int_vec.copy()
    #             hn_vec = hn_int_vec.copy()
                
    #             # build spawning mask
    #             is_spawning_mask = spawn_mask & w_spawn_obs

    #             for v_n, p_flap_obs in v_pot_vec:

    #                 # spawn location if it occurs, 0 otherwise
    #                 m_min_idx_vec = m_min_vec - is_passing_mask
    #                 #dn_vec[space_spawn_mask, m_min_idx_vec[space_spawn_mask]] = (free_space_vec*is_spawning_mask)[space_spawn_mask]
    #                 rows = np.where(space_spawn_mask)[0]
    #                 cols = m_min_idx_vec[rows]
    #                 dn_vec[rows, cols] = (free_space_vec * is_spawning_mask)[rows]

    #                 # spawn height if it occurs, default otherwise
    #                 # hn_vec[space_spawn_mask, m_min_vec - is_passing_mask] = this_h*is_spawning_mask + default_height*~is_spawning_mask
    #                 rows = np.where(space_spawn_mask)[0]
    #                 cols = m_min_vec[rows] - is_passing_mask[rows]
    #                 hn_vec[rows, cols] = (this_h * is_spawning_mask[rows] + default_height * ~is_spawning_mask[rows])

    #                 v_next_vec = np.full_like(yn_vec, v_n)

    #                 next_states = np.column_stack([
    #                     yn_vec,
    #                     v_next_vec,
    #                     dn_vec,
    #                     hn_vec
    #                 ])

    #                 j_vec = np.array([
    #                     C.state_to_index(tuple(s))
    #                     for s in next_states
    #                 ])

    #                 prob_vec = p_flap_obs * p_spawn_obs_vec * p_height_obs

    #                 P[non_col_idx, j_vec, l] += prob_vec


    # for every current state, input pair, build next states and calculate probability
    for i, state in enumerate(non_col_states):
        v = state[1]
        d = list(state[2:M+2])

        # free space
        s = free_space_vec[i]

        # spawn check
        can_spawn = spawn_mask[i]

        # set m_min to index or M if no 0 found
        m_min = m_min_vec[i]

        # next position (deterministic)
        y_next = yn_vec[i]

        # next distances and heights for passing case (intermediate version, before spawning)
        d_n_int = dn_int_vec[i]
        h_n_int = hn_int_vec[i]
        
        for l, input in enumerate(C.input_space):

            # next velocity (random - set of potential velocities)
            v_pot = v_next_cache[(v, input)] 

            for v_n, p_flap_obs in v_pot:

                # update distances/heights with spawn
                for w_spawn_obs in (0, 1):
                    if not can_spawn and w_spawn_obs:
                        continue
                    
                    # calculate probability the observed spawn behavior happened
                    p_spawn_obs = pspawn_vec[i] if w_spawn_obs else (1.0 - pspawn_vec[i])

                    if p_spawn_obs == 0:
                        continue
                
                    # heights of spawned obstacle
                    if w_spawn_obs:
                        p_height_obs = 1.0 / n_heights
                        height_pot = C.S_h
                    else:
                        p_height_obs = 1.0
                        height_pot = [default_height]
                    
                    for this_h in height_pot:

                        # spawn in shift case
                        if space_spawn_mask[i]:
                            if can_spawn and w_spawn_obs:
                                d_n_int[m_min - (d[0] == 0)] = s
                                h_n_int[m_min - (d[0] == 0)] = this_h
                            else:
                                d_n_int[m_min - (d[0] == 0)] = 0
                                h_n_int[m_min - (d[0] == 0)] = default_height

                        # build next state 
                        next_state = (y_next, v_n, *d_n_int, *h_n_int)
                        unfiltered_idx = my_state_to_index(next_state, C)
                        real_idx = unfiltered_to_filtered[unfiltered_idx]
                        j = C.state_to_index(next_state)
                        if real_idx == -1:
                            print("\n\n\nHELP\n\n\n")
                        # if j != real_idx:
                        #     print(j, next_state, real_idx, C.state_space[real_idx])
                            
                        
                        # update probability
                        prob = p_flap_obs * p_spawn_obs * p_height_obs
                        P[non_col_idx[i], j, l] = prob
    
    return P


def compute_total_possible_states(C):
    """
    Returns the total number of possible states before filtering.
    """
    n_y = C.Y
    n_v = 2 * C.V_max + 1
    n_d1 = C.X
    n_d_rest = C.X - C.D_min
    n_h = len(C.S_h)
    
    return n_y * n_v * n_d1 * (n_d_rest ** (C.M - 1)) * (n_h ** C.M)

def my_state_to_index(x, C):
    y, v = x[0], x[1]
    D = list(x[2:2+C.M])
    H = list(x[2+C.M:])

    Nv = 2*C.V_max + 1
    Nd1 = C.X - 1
    Nd_rest = C.X - C.D_min
    Nh = len(C.S_h)

    idx = y
    idx = idx * Nv + (v + C.V_max)
    idx = idx * Nd1 + D[0]   # first distance uses full range

    for d in D[1:]:
        if d == 0:
            idx = idx * Nd_rest
        else:
            idx = idx * Nd_rest + (d - C.D_min + 1)
 
    for h in H:
        idx = idx * Nh + C.S_h.index(h)

    return int(idx)