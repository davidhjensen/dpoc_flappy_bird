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

    # calculate spawn probabilities for all s
    pspawn = np.zeros(C.X + 1)
    for s in range(C.X + 1):
        if s <= C.D_min - 1:
            pspawn[s] = 0.0
        elif s >= C.X:
            pspawn[s] = 1.0
        else:
            pspawn[s] = (s - (C.D_min - 1)) / (C.X - C.D_min)

    # save spaces and variables
    state_space = C.state_space
    input_space = C.input_space
    M = C.M
    default_height = C.S_h[0]
    n_heights = len(C.S_h)

    # for every current state, input pair, build next states and calculate probability
    for i, state in enumerate(C.state_space):
        y, v = state[0], state[1]
        d = list(state[2:M+2])
        h = list(state[M+2:])

        # skip rows for collision states
        if is_collision(C, y, d[0], h[0]):
            continue

        # free space
        s = (C.X - 1) - (sum(d) - 1)

        # spawn check
        can_spawn = s >= C.D_min

        # set m_min to index or M if no 0 found
        m_min = next((k for k in range(1,M) if d[k] == 0), M)

        for l, input in enumerate(C.input_space):
            
            # next position (deterministic)
            y_next = min(max(y + v, 0), C.Y - 1)

            # next velocity (random - set of potential velocities)
            v_pot = v_next_cache[(v, input)] 

            for v_n, p_flap_obs in v_pot:

                # next distances and heights for passing case (intermediate version, before spawning)
                if d[0] == 0:
                    d_n_int = [d[1] - 1] + d[2:] + [0]
                    h_n_int = h[1:] + [default_height]
                
                # next distances and heights for drifting case (intermediate version, before spawning)
                else:
                    d_n_int = [d[0] - 1] + d[1:]
                    h_n_int = list(h)

                # update distances/heights with spawn
                for w_spawn_obs in (0, 1):
                    if not can_spawn and w_spawn_obs:
                        continue
                    
                    # calculate probability the observed spawn behavior happened
                    p_spawn_obs = pspawn[s] if w_spawn_obs else (1.0 - pspawn[s])

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

                        # copy lists beceause we want to leave intermediate versions unchanged
                        d_n = d_n_int.copy()
                        h_n = h_n_int.copy()

                        # spawn in shift case
                        if d[0] == 0:
                            if can_spawn and w_spawn_obs:
                                d_n[m_min - 1] = s
                                h_n[m_min - 1] = this_h
                            else:
                                d_n[m_min - 1] = 0
                                h_n[m_min - 1] = default_height

                        # spawn in dift case
                        else:
                            if m_min != M:
                                if can_spawn and w_spawn_obs:
                                    d_n[m_min] = s
                                    h_n[m_min] = this_h
                                else:
                                    d_n[m_min] = 0
                                    h_n[m_min] = default_height

                        # build next state 
                        next_state = (y_next, v_n, *d_n, *h_n)
                        j = C.state_to_index(next_state)
                        if j is None:
                            continue
                        
                        # update probability
                        P[i, j, l] = p_flap_obs * p_spawn_obs * p_height_obs

    return P