""" Module contains functions for computations relating to half spaces.
"""
import numpy as np
from scipy import optimize
from scipy.spatial import HalfspaceIntersection


def point_inside_half_space_intersection(
    n: np.ndarray, x0: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    """
    Find the points that lie in the intersection of half spaces (3D)

    Parameters
    ----------
    n : np.ndarray, size 3 x num_planes
        This is the normal vectors of the half planes. The normal
        vectors is assumed to point out of the half spaces.
    x0 : np.ndarray, size 3 x num_planes
        Point on the boundary of the half-spaces. Half space i is given
        by all points satisfying (x - x0[:,i])*n[:,i]<=0
    pts : np.ndarray, size 3 x num_points
        The points to be tested if they are in the intersection of all
        half-spaces or not.

    Returns
    -------
    out : np.ndarray, size num_points
        A logical array with length equal number of pts.
        out[i] is True if pts[:,i] is in all half-spaces

    Examples
    --------
    >>> import numpy as np
    >>> n = np.array([[0, 1], [1, 0], [0, 0]])
    >>> x0 = np.array([[0, -1], [0, 0], [0, 0]])
    >>> pts = np.array([[-1 ,-1 ,4], [2, -2, -2], [0, 0, 0]])
    >>> half_space_int(n,x0,pts)
    array([False,  True, False], dtype=bool)

    """
    assert n.shape[0] == 3, " only 3D supported"
    assert x0.shape[0] == 3, " only 3D supported"
    assert pts.shape[0] == 3, " only 3D supported"
    assert (
        n.shape[1] == x0.shape[1]
    ), "there must be the same number of normal vectors as points"

    n_pts = pts.shape[1]
    in_hull = np.zeros(n_pts)
    x0 = np.repeat(x0[:, :, np.newaxis], n_pts, axis=2)
    n = np.repeat(n[:, :, np.newaxis], n_pts, axis=2)
    for i in range(x0.shape[1]):
        in_hull += np.sum((pts - x0[:, i, :]) * n[:, i, :], axis=0) <= 0

    return in_hull == x0.shape[1]


def half_space_interior_point(
    n: np.ndarray, x0: np.ndarray, pts: np.ndarray, recompute: bool = True
) -> np.ndarray:
    """
    Find an interior point for the halfspaces.

    We use linear programming to find one interior point for the half spaces.
    Assume, num_planes halfspaces defined by

        aj*x1+bj*x2+cj*x3+dj<=0, j=1..num_planes.

    Perform the following linear program:

        max(x5) aj*x1+bj*x2+cj*x3+dj*x4+x5<=0, j=1..num_planes

    Then, if [x1,x2,x3,x4,x5] is an optimal solution with x4>0 and x5>0 we get:

        aj*(x1/x4)+bj*(x2/x4)+cj*(x3/x4)+dj<=(-x5/x4) j=1..num_planes

    and
         (-x5/x4)<0,

    and conclude that the point [x1/x4,x2/x4,x3/x4] is in the interior of all
    the halfspaces. Since x5 is optimal, this point is "way in" the interior
    (good for precision errors).
    http://www.qhull.org/html/qhalf.htm#notes

    Parameters
    ----------
    n : np.ndarray, size 3 x num_planes
        This is the normal vectors of the half planes. The normal vectors
        are assumed to be coherently oriented for all the half spaces
        (inward or outward).
    x0 : np.ndarray, size 3 x num_planes
        Point on the boundary of the half-spaces. Half space i is given
        by all points satisfying (x - x0[:, i]) * n[:, i] <= 0
    pts : np.ndarray, size 3 x num_points
        Points used to bound the search space for interior point. The optimum
        solution will be sought within (pts.min(axis=1), pts.max(axis=1)).
    recompute: bool
        If the algorithm fails try again with flipped normals.

    Returns
    -------
    out: np.ndarray, size num_points
        Interior point of the halfspaces.

    """
    import scipy.optimize as opt

    dim = (1, n.shape[1])
    c = np.array([0, 0, 0, 0, -1])
    A_ub: np.ndarray = np.vstack((n, [np.sum(-n * x0, axis=0)], np.ones(dim))).T
    b_ub = np.zeros(dim).T
    b_min, b_max = np.amin(pts, axis=1), np.amax(pts, axis=1)
    bounds = (
        (b_min[0], b_max[0]),
        (b_min[1], b_max[1]),
        (b_min[2], b_max[2]),
        (0, None),
        (0, None),
    )
    res = opt.linprog(c, A_ub, b_ub, bounds=bounds)

    if recompute and (not res.success or np.all(np.isclose(res.x[3:], 0))):
        return half_space_interior_point(-n, x0, pts, False)

    if res.success and not np.all(np.isclose(res.x[3:], 0)):
        return np.array(res.x[:3]) / res.x[3]
    else:
        raise ValueError("Half space intersection empty")


def vertexes_of_convex_domain(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the vertexes of a convex domain specified as an intersection of half spaces.

    The function assumes the domain is defined by inequalities on the form

        A * x + b <= 0

    For more information, see scipy.spatial functions HalfspaceIntersection.

    The function has been tested for 2d and 3d domains.

    Error messages from HalfspaceIntersection are quite possibly from qhull.
    These will often reflect errors in the way the matrix A and b are set up
    (e.g. sign errors that imply that the inequalities do not form a closed domain).

    Parameters
    ----------
    A : np.ndarray, size num_planes x num_dim
        Matrix of normal vectors (in rows) for the half planes. Should be oriented
        so that A * x + b < 0
    b : np.ndarray, size num_planes
        Constants used to define inequalities of the half spaces. Should be scaled
        so that A * x + b < 0.

    """
    b = b.reshape((-1, 1))

    # First, find an interior point of the half space. For this we could have used
    # the function half_space_interior_point, but that function is heavily geared
    # towards 3d domains, so we prefer the simpler option below.

    # Find the point that minimizes the distance from all half planes; this should
    # be a point in the middle (somehow defined) of the domain.
    fun = lambda x: np.linalg.norm(A.dot(x.reshape((-1, 1))) + b)
    # Use scipy optimization to find an interior point to the half space.
    interior_point = optimize.minimize(fun, np.zeros(A.shape[1])).x

    # Set up constraints on the format that scipy.spatial HalfspaceIntersection
    # expects
    constraints = np.hstack((A, b))

    # Get hold of domain (this will call qhull)
    domain = HalfspaceIntersection(constraints, interior_point)

    # Return intersections in the expected format (thus the transpose)
    return domain.intersections.T
