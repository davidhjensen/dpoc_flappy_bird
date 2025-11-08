"""utils.py

Python script containg utility functions. Modify if needed,
but be careful as these functions are used, e.g., in simulation.py.

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

from Const import Const

def spawn_probability(C: Const, s: int) -> float:
    """Distance-dependent spawn probability p_spawn(s).
    
    Args:
        C (Const): The constants describing the problem instance.
        s (int): Free distance, as defined in the assignment.

    Returns:
        float: The probability of the observed spawn occuring.
    """
    return max(min((s - C.D_min + 1) / (C.X - C.D_min), 1.0), 0.0)

def flap_probability(C: Const, u: int, w_flap_obs: int) -> float:
    """Input-dependent flap distubance probability p_flap(u).

    Args:
        C (const): The constants describing the problem instance.
        u (int): The input flap strength.
        w_flap_obs (int): The observed disturbance.
    
    Returns:
        float: The probability of the observed disturbance occuring.
    """
    if u == C.U_strong:
        return 1 / (2*C.V_dev + 1)
    elif u in (C.U_weak, C.U_no_flap) and w_flap_obs == 0:
        return 1
    elif u in (C.U_weak, C.U_no_flap) and w_flap_obs != 0:
        return 0
    else:
        print(f"THIS IS BAD\nu: {u}\nw_flap_obs: {w_flap_obs}")
        return 0
    
def is_in_gap(C: Const, y: int, h1: int) -> bool:
    """Returns true if bird in gap.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is in the gap, False otherwise.
    """
    half = (C.G - 1) // 2
    return abs(y - h1) <= half

def is_passing(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is currently passing the gap without colliding.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is passing the gap, False otherwise.
    """
    return (d1 == 0) and is_in_gap(C, y, h1)

def is_collision(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is colliding with obstacle.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is colliding with obstacle, False otherwise.
    """
    return (d1 == 0) and not is_in_gap(C, y, h1)