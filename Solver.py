"""Solver.py

Template to solve the stochastic shortest path problem.

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
from ComputeExpectedStageCosts import compute_expected_stage_cost
from ComputeTransitionProbabilities import compute_transition_probabilities
from scipy.sparse import csr_matrix


def solution(C: Const) -> tuple[np.array, np.array]:
    """Computes the optimal cost and the optimal control policy. 
    
    You can solve the SSP by any method:
    - Value Iteration
    - Policy Iteration
    - Linear Programming
    - A combination of the above
    - Others?

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP, 
            of shape (C.K,), where C.K is the number of states.
        np.array: The optimal control policy for the stochastic SPP, 
            of shape (C.K,), where each entry is in {0,...,C.L-1}.
    """    
    # You're free to use the functions below, implemented in the previous
    # tasks, or come up with something else.
    # If you use them, you need to add the corresponding imports 
    # at the top of this file.
    
    
    # TODO: implement Value Iteration, Policy Iteration, Linear Programming 
    # or a combination of these
    P = compute_transition_probabilities(C)
    Q = compute_expected_stage_cost(C)
    J_opt = np.zeros(C.K)
    u_opt = np.zeros(C.K, dtype=int)

    K, _, L = P.shape
    P_sparse = [csr_matrix(P[:, :, l]) for l in range(L)]
    alpha = 1.0        
    epsilon = 1e-5
    max_iter = 10000

    for it in range(max_iter):
        J_new = np.full(K, 0)
        for l in range(L):
            expected_cost = Q[:, l] + alpha * (P_sparse[l].dot(J_opt))
            J_new = np.minimum(J_new, expected_cost)

        delta = np.max(np.abs(J_new - J_opt))
        if delta < epsilon:
            print(f"breaking due to epsilon at {it}")
            break
        J_opt = J_new

    best_cost = np.full(K, np.inf)
    best_action = np.zeros(K, dtype=int)
    for l in range(L):
        cost_l = Q[:, l] + alpha * (P_sparse[l].dot(J_opt))
        mask = cost_l < best_cost
        best_cost[mask] = cost_l[mask]
        best_action[mask] = l

    u_opt = np.array([C.input_space[a] for a in best_action])


    return J_opt, u_opt
