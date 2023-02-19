import numpy as np
from numba import jit, njit, prange

def init_boids(boids: np.ndarray, asp: float, vrange: tuple = (0., 1.)):
    """
    Initialize boids as set of coordinates (x, y) and velocity projections (v_x, v_y).

    :param boids: (np.ndarray) array to store coordinates and velocities
    :param asp: (float) right border of available x coordinates
    :param vrange: (tuple) range of available velocities
    :return: None
    """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


@njit
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    Calculate the vector of motion direction

    :param boids: (np.ndarray) boids array
    :param dt: time step
    :return: (np.ndarray) array with rows like [x_stepback, y_stepback, x_curr, y_curr]
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))

@njit
def numfrend_norm(arr: np.array):
    """
    Numba-friendly norm calculation.
    :param arr: (np.ndarray) array with vectors
    :return: (float) norm (over 1st axis)
    """
    return np.sqrt(np.sum(arr.copy() ** 2, axis=1))

@njit
def numfrend_mean(arr: np.array, axis=0):
    """
    Numba-friendly mean calculation.
    :param arr: (np.ndarray) array
    :param axis: (int) axis to calculate mean
    :return: (float) mean value over axis
    """
    return np.sum(arr, axis=axis) / arr.shape[axis]

@njit
def numfrend_median_ax0(arr: np.array):
    """
    Numba-friendly median calculation.
    :param arr: (np.ndarray) array
    :return: (arr.dtype) median value
    """
    arr_sorted = arr.copy()
    arr_sorted[:, 0] = np.sort(arr[:, 0])
    arr_sorted[:, 1] = np.sort(arr[:, 1])

    if len(arr_sorted) % 2 == 1:
        return arr_sorted[len(arr_sorted) // 2]
    else:
        return (arr_sorted[len(arr_sorted) // 2] + arr_sorted[len(arr_sorted) // 2 - 1]) / 2

@njit
def clip_mag(arr: np.ndarray, lims=(0., 1.)):
    """
    Clip values of array which are out of limits to upper/lower bound.

    :param arr: (np.ndarray) array to clip
    :param lims: (tuple) lower and upper bounds
    :return: None
    """
    v = numfrend_norm(arr)
    mask = v > 0
    v_clip = np.clip(v, *lims)
    arr[mask] *= (v_clip[mask] / v[mask]).reshape(-1, 1)


@njit
def propagate(boids: np.ndarray, dt: float, vrange: tuple):
    """
    Change locations and velocities of boids in a timestep.

    :param boids: (np.ndarray) boids array
    :param dt: (float) timestep
    :param vrange: (tuple) velocity range
    :return: None
    """
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], lims=vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit
def periodic_walls(boids: np.ndarray, asp: float):
    """
    Implements the feature of periodic walls.

    :param boids: (np.ndarray) boids array
    :param asp: (float) upper limit of x coordinate
    :return: None
    """
    boids[:, 0:2] %= np.array([asp, 1.])


@njit
def wall_avoidance(boids, asp):
    """
    Implements wall avoidance features (changes v_x, v_y to run away from walls).

    :param boids: (np.ndarray) boids array
    :param asp: (float) upper limit of x coordinate
    :return: None
    """
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])

    ax = 1. / left**2 - 1. / right**2
    ay = 1. / bottom**2 - 1. / top**2
    boids[:, 4:6] += np.column_stack((ax, ay))


@njit
def walls(boids: np.ndarray, asp: float):
    """
    Find wall avoidance vector.

    :param boids: (np.ndarray) boids array
    :param asp: (float) upper limit of x coordinate
    :return: (np.ndarray) horizontal and vertical ranges in terms of walls
    """
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]

    a_left = 1 / (np.abs(x) + c)**2
    a_right = -1 / (np.abs(x - asp) + c)**2

    a_bottom = 1 / (np.abs(y) + c)**2
    a_top = -1 / (np.abs(y - 1.) + c)**2

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit(cache=True)
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """
    Calculate cohesion vector.

    :param boids: (np.ndarray) boids array
    :param idx: (int) boid index
    :param neigh_mask: (np.ndarray) boolean mask of "neighbours" (boids inside perception circle)
    :param perception: (float) perception distance
    :return: (np.ndarray) cohesion vector
    """

    center = numfrend_median_ax0(boids[neigh_mask, :2])
    # center = numfrend_mean(boids[neigh_mask, :2])
    a = (center - boids[idx, :2]) * perception
    return a


@njit(cache=True)
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """
    Calculate separation vector.

    :param boids: (np.ndarray) boids array
    :param idx: (int) boid index
    :param neigh_mask: (np.ndarray) boolean mask of "neighbours" (boids inside perception circle)
    :return: (np.ndarray) separation vector
    """

    d = numfrend_median_ax0(boids[neigh_mask, :2] - boids[idx, :2])
    # d = numfrend_mean(boids[neigh_mask, :2] - boids[idx, :2])
    return -d / ((d[0]**2 + d[1]**2) + 1)


@njit(cache=True)
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:

    """
    Calculate alignment vector.

    :param boids: (np.ndarray) boids array
    :param idx: (int) boid index
    :param neigh_mask: (np.ndarray) boolean mask of "neighbours" (boids inside perception circle)
    :param vrange: (tuple) velocity range
    :return: (np.ndarray) alignment vector
    """

    v_mean = numfrend_median_ax0(boids[neigh_mask, 2:4])
    # v_mean = numfrend_mean(boids[neigh_mask, 2:4])
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit(cache=True, parallel=False)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple):
    """
    Recalculate accelerations of each boid.

    :param boids: (np.ndarray) boids array
    :param perception: (float) perception distance
    :param coeffs: (np.ndarray) coefficients for each motion tendencies
    :param asp: (float) upper limit of x coordinate
    :param vrange: (tuple) velocity range
    :return: None
    """

    # -----------------------distance calculation------------------------- #
    p = boids[:, :2]
    res = np.empty((boids.shape[0], boids.shape[0], 2))

    for i in range(boids.shape[0]):
        for j in range(boids.shape[0]):
            res[i, j, 0] = p[i, 0] - p[j, 0]
            res[i, j, 1] = p[i, 1] - p[j, 1]

    dist = (res[:, :, 0]**2 + res[:, :, 1]**2)**0.5
    # -------------------------------------------------------------------- #

    N = boids.shape[0]

    for i in prange(N):
        dist[i, i] = perception + 1

    mask = dist < perception
    wal = walls(boids, asp)

    for i in range(N):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i])

        noize = np.random.uniform(0, 1, size=2)

        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * sep + coeffs[3] * wal[i] * \
            coeffs[4] * noize

        boids[i, 4:6] = a
