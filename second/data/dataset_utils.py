import math


def wrap_to_pi(theta):
    # (-inf, inf) -> [-pi, pi]
    return math.atan2(math.sin(theta), math.cos(theta))
