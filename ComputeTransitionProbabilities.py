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
    
    return P