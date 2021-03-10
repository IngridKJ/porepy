# -*- coding: utf-8 -*-
"""
@author: fumagalli, alessio
"""

import logging
from typing import Dict

import numpy as np
import scipy.sparse as sps

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["numerics", "discretization"]


class RT0(pp.numerics.vem.dual_elliptic.DualElliptic):
    @pp.time_logger(sections=module_sections)
    def __init__(self, keyword: str) -> None:
        super(RT0, self).__init__(keyword, "RT0")
        # variable name to store the structure that map a cell to the opposite nodes
        # of the local faces
        self.cell_face_to_opposite_node = "rt0_class_cell_face_to_opposite_node"

    @pp.time_logger(sections=module_sections)
    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """Discretize a second order elliptic equation using using a RT0-P0 method.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d and 1d grids.

        parameter_dictionary contains the entries:
            second_order_tensor: (pp.SecondOrderTensor) Permeability defined
                cell-wise. This is the effective permeability, including any
                aperture scalings etc.

        matrix_dictionary will be updated with the following entries:
            mass: sps.csc_matrix (g.num_faces, g.num_faces)
                The mass matrix.
            div: sps.csc_matrix (g.num_cells, g.num_faces)
                The divergence matrix.

        Optional parameter:
        --------------------
        is_tangential: Whether the lower-dimensional permeability tensor has been
            rotated to the fracture plane. Defaults to False and stored in the data
            dictionary.
        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # Get dictionary for discretization matrix storage
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        # If a 0-d grid is given then we return an identity matrix
        if g.dim == 0:
            mass = sps.dia_matrix(([1], 0), (g.num_faces, g.num_faces))
            matrix_dictionary[self.mass_matrix_key] = mass
            matrix_dictionary[self.div_matrix_key] = sps.csr_matrix(
                (g.num_faces, g.num_cells)
            )
            matrix_dictionary[self.vector_proj_key] = sps.csr_matrix((3, g.num_cells))
            return

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability
        k = parameter_dictionary["second_order_tensor"]

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
            g, deviation_from_plane_tol
        )

        node_coords = node_coords[: g.dim, :]

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if g.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix A entries
        size_A = np.power(g.dim + 1, 2) * g.num_cells
        rows_A = np.empty(size_A, dtype=int)
        cols_A = np.empty(size_A, dtype=int)
        data_A = np.empty(size_A)
        idx_A = 0

        # Allocate the data to store matrix P entries
        size_P = 3 * (g.dim + 1) * g.num_cells
        rows_P = np.empty(size_P, dtype=int)
        cols_P = np.empty(size_P, dtype=int)
        data_P = np.empty(size_P)
        idx_P = 0
        idx_row_P = 0

        size_HB = g.dim * (g.dim + 1)
        HB = np.zeros((size_HB, size_HB))
        for it in np.arange(0, size_HB, g.dim):
            HB += np.diagflat(np.ones(size_HB - it), it)
        HB += HB.T
        HB /= g.dim * g.dim * (g.dim + 1) * (g.dim + 2)

        # define the function to compute the inverse of the permeability matrix
        if g.dim == 1:
            inv_matrix = self._inv_matrix_1d
        elif g.dim == 2:
            inv_matrix = self._inv_matrix_2d
        elif g.dim == 3:
            inv_matrix = self._inv_matrix_3d

        # compute the oppisite node per face
        self._compute_cell_face_to_opposite_node(g, data)
        cell_face_to_opposite_node = data[self.cell_face_to_opposite_node]

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            # get the opposite node id for each face
            node = cell_face_to_opposite_node[c, :]
            coord_loc = node_coords[:, node]

            # Compute the H_div-mass local matrix
            A = RT0.massHdiv(
                inv_matrix(k.values[0 : g.dim, 0 : g.dim, c]),
                g.cell_volumes[c],
                coord_loc,
                sign[loc],
                g.dim,
                HB,
            )

            # Compute the flux reconstruction matrix
            P = RT0.faces_to_cell(
                c_centers[:, c],
                coord_loc,
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                dim,
                R,
            )

            # Save values for Hdiv-mass local matrix in the global structure
            cols = np.concatenate(faces_loc.size * [[faces_loc]])
            loc_idx = slice(idx_A, idx_A + A.size)
            rows_A[loc_idx] = cols.T.ravel()
            cols_A[loc_idx] = cols.ravel()
            data_A[loc_idx] = A.ravel()
            idx_A += A.size

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx_P, idx_P + P.size)
            cols_P[loc_idx] = np.concatenate(3 * [[faces_loc]]).ravel()
            rows_P[loc_idx] = np.repeat(np.arange(3), faces_loc.size) + idx_row_P
            data_P[loc_idx] = P.ravel()
            idx_P += P.size
            idx_row_P += 3

        # Construct the global matrices
        mass = sps.coo_matrix((data_A, (rows_A, cols_A)))
        div = -g.cell_faces.T
        proj = sps.coo_matrix((data_P, (rows_P, cols_P)))

        matrix_dictionary[self.mass_matrix_key] = mass
        matrix_dictionary[self.div_matrix_key] = div
        matrix_dictionary[self.vector_proj_key] = proj

    @staticmethod
    def project_flux_matrix(g, data):
        """Construct the matrix that will project the velocity computed with a rt0
        discretization to obtain a piecewise constant vector field, one triplet for each cell.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d and 1d grids.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.

        Return
        ------
        mass :

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        if g.dim == 0:
            return np.zeros(3).reshape((3, 1))

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
            g, deviation_from_plane_tol
        )

        nodes, _, _ = sps.find(g.face_nodes)

        P0u = np.zeros((3, g.num_cells))

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 3 * (g.dim + 1) * g.num_cells
        nd = np.arange(g.num_cells * 3)
        I = np.vstack((nd, nd))
        if g.dim > 1:
            for i in range(1, g.dim):
                I = np.vstack((I, nd))
        J = np.empty(size, dtype=np.int)
        dataIJ = np.empty(size)
        idx = 0

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]
            A = np.zeros((3, faces_loc.size))

            # find the opposite node id for each face
            node = RT0.opposite_side_node(g.face_nodes, nodes, faces_loc)

            # extract the coordinates
            center = np.tile(c_centers[:, c], (node.size, 1)).T
            delta_c = center - node_coords[:, node]
            delta_f = f_centers[:, faces_loc] - node_coords[:, node]
            normals = f_normals[:, faces_loc]

            Pi = delta_c / np.einsum("ij,ij->j", delta_f, normals)
            A[dim, :] = delta_c / np.einsum("ij,ij->j", delta_f, normals)

            # extract the velocity for the current cell
            # P0u[dim, c] = np.dot(Pi, u[faces_loc])
            # P0u[:, c] = np.dot(R.T, P0u[:, c])

            # Save values for RT0 local matrix in the global structure
            cols = np.tile(faces_loc, (3, 1))
            loc_idx = slice(idx, idx + cols.size)
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = np.dot(R.T, A).ravel()
            idx += cols.size

        # Construct the global matrices
        mass = sps.coo_matrix((dataIJ, (I.ravel("f"), J)))

        return mass

    @staticmethod
    def project_flux(g, u, data):
        """Project the velocity computed with a rt0 solver to obtain a
        piecewise constant vector field, one triplet for each cell.
        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]
            deviation_from_plane_tol: The geometrical tolerance, used in the check to
                rotate 2d and 1d grids.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        u : array (g.num_faces) Velocity at each face.

        Return
        ------
        P0u : ndarray (3, g.num_faces) Velocity at each cell.

        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        if g.dim == 0:
            return np.zeros(3).reshape((3, 1))

        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        deviation_from_plane_tol = data.get("deviation_from_plane_tol", 1e-5)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
            g, deviation_from_plane_tol
        )

        nodes, _, _ = sps.find(g.face_nodes)

        P0u = np.zeros((3, g.num_cells))

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            # find the opposite node id for each face
            node = RT0.opposite_side_node(g.face_nodes, nodes, faces_loc)

            # extract the coordinates
            center = np.tile(c_centers[:, c], (node.size, 1)).T
            delta_c = center - node_coords[:, node]
            delta_f = f_centers[:, faces_loc] - node_coords[:, node]
            normals = f_normals[:, faces_loc]

            Pi = delta_c / np.einsum("ij,ij->j", delta_f, normals)

            # extract the velocity for the current cell
            P0u[dim, c] = np.dot(Pi, u[faces_loc])
            P0u[:, c] = np.dot(R.T, P0u[:, c])

        return P0u

    @staticmethod
    @pp.time_logger(sections=module_sections)
    def massHdiv(
        inv_K: np.ndarray,
        c_volume: float,
        coord: np.ndarray,
        sign: np.ndarray,
        dim: int,
        HB: np.ndarray,
    ) -> np.ndarray:
        """Compute the local mass Hdiv matrix using the mixed vem approach.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_volume : scalar
            Cell volume.
        sign : array (num_faces_of_cell)
            +1 or -1 if the normal is inward or outward to the cell.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """
        ind = np.eye(dim + 1)
        # expand the inv_K tensor
        inv_K_exp = (
            ind[:, np.newaxis, :, np.newaxis]
            * inv_K[np.newaxis, :, np.newaxis, :]
            / c_volume
        )
        inv_K_exp.shape = (ind.shape[0] * inv_K.shape[0], ind.shape[1] * inv_K.shape[1])

        N = coord.flatten("F").reshape((-1, 1)) * np.ones(
            (1, dim + 1)
        ) - np.concatenate((dim + 1) * [coord])
        C = np.diag(sign)

        return np.dot(C.T, np.dot(N.T, np.dot(HB, np.dot(inv_K_exp, np.dot(N, C)))))

    @staticmethod
    @pp.time_logger(sections=module_sections)
    def faces_to_cell(
        pt: np.ndarray,
        coord: np.ndarray,
        f_centers: np.ndarray,
        f_normals: np.ndarray,
        dim: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """Construct a local matrix that evaluate a RT0 solution in a give point (cell center).

        Parameters
        ----------
        pt: the point where to evaluate the field, usually cell center
        coord: the vertices of the simplex
        f_centers: the centre of the faces ordered following coord
        f_normals: the normal of the faces ordered as f_centers
        dim: the spatial dimension

        """
        pt_reshaped = np.repeat(pt, coord.shape[1]).reshape((-1, coord.shape[1]))
        c_delta = pt_reshaped - coord
        f_delta = f_centers - coord

        # the resulting vector has always three componenets
        P = np.zeros((3, coord.shape[1]))
        P[dim, :] = c_delta / np.einsum("ij,ij->j", f_delta, f_normals)
        return np.dot(R.T, P)

    @pp.time_logger(sections=module_sections)
    def _compute_cell_face_to_opposite_node(
        self, g: pp.Grid, data: Dict, recompute: bool = False
    ) -> None:
        """Compute a map that given a face return the node on the opposite side,
        typical request of a Raviart-Thomas approximation.
        This function is mainly for internal use and, if the geometry is fixed during
        the simulation, it will be called once.

        The map constructed is that for each cell, return the id of the node
        their opposite side of the local faces.

        Parameters:
        ----------
        g: grid
        data: data associated to the grid where the map will be stored
        recompute: (optional) recompute the map even if already computed. Default False
        """

        # if already computed avoid to do it again
        if not (data.get(self.cell_face_to_opposite_node, None) is None or recompute):
            return

        faces, cells, _ = sps.find(g.cell_faces)
        faces = faces[np.argsort(cells)]

        # initialize the map
        cell_face_to_opposite_node = np.empty((g.num_cells, g.dim + 1), dtype=int)

        nodes, _, _ = sps.find(g.face_nodes)
        indptr = g.face_nodes.indptr

        # loop on all the cells to construct the map
        for c in np.arange(g.num_cells):
            # For the current cell retrieve its faces
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = faces[loc]

            # get the local nodes, face based
            face_nodes = np.array([nodes[indptr[f] : indptr[f + 1]] for f in faces_loc])
            nodes_loc = np.unique(face_nodes.flatten())

            # get the opposite node for each face
            opposite_node = np.array(
                [np.setdiff1d(nodes_loc, f, assume_unique=True) for f in face_nodes]
            )

            # find the opposite node id for each face
            cell_face_to_opposite_node[c, :] = opposite_node.flatten()

        data[self.cell_face_to_opposite_node] = cell_face_to_opposite_node
