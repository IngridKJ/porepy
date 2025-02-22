"""
The module contains the base class for all elliptic discretization methods, to be used
in both mono and mixed-dimensional problems. One instance for the scalar and one for
the vector equation.

The classes may be interpreted as a "contract" which elliptic discretization methods
must honor. This is particularly useful to ensure uniformity when coupling
discretizations between dimensions by interface laws.
"""
from abc import abstractmethod
from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization


class EllipticDiscretization(Discretization):
    """This is the parent class of all discretizations for second order elliptic
    problems. The class cannot be used itself, but should rather be seen as a
    declaration of which methods are assumed implemented for all specific
    discretization schemes.

    Subclasses are intended used both on single grids, and as components in a
    mixed-dimensional, or more generally multiple grid problem. In the latter case,
    this class will provide the discretization on individual grids; a full
    discretization will also need discretizations of the edge problems (coupling between
    grids) and a way to assemble the grids.

    The class methods should together take care of both the discretization
    (defined as construction of operators used when writing down the continuous
    form of the equations, e.g. divergence, transmissibility matrix etc), and
    assembly of (whole or part of) the system matrix. For problems on multiple
    grids, an assembly will only be a partial (considering a single grid)- the
    full system matrix for multiple grids is handled somewhere else.

    Known realizations of the elliptic discretization are:
        Tpfa: Finite volume method using a wwo-point flux approximation.
        Mpfa: Finite volume method using a multi-point flux approximation.
        RT0: Mixed finite element method, using the lowest order Raviart-Thomas
            elements.
        MVEM: Mixed formulation of the lowest order virtual element method.

    Attributes:
        keyword (str): This is used to identify operators and parameters that
            the discretization will work on.

    """

    def __init__(self, keyword: str) -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    @abstractmethod
    def ndof(self, sd: pp.Grid) -> int:
        """Abstract method. Return the number of degrees of freedom associated to the
        method.

        Args:
            sd (grid): Computational grid.

        Returns:
            int: the number of degrees of freedom.

        """
        pass

    def extract_pressure(self, sd: pp.Grid, solution_array: np.ndarray, data: dict):
        """Abstract method. Extract the pressure part of a solution.

        The implementation will depend what the primary variables of the specific
        implementation are.

        Args:
            sd (pp.Grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            data (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.

        Returns:
            np.array (g.num_cells): Pressure solution vector. Will be identical
                to solution_array.

        """
        raise NotImplementedError("Method not implemented")

    def extract_flux(self, sd: pp.Grid, solution_array: np.ndarray, data: dict):
        """Abstract method. Extract the pressure part of a solution.

        The implementation will depend what are the primary variables of the specific
        implementation.

        TODO: We should incrude the boundary condition as well?

        Args:
            sd (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem. Will
                correspond to the pressure solution.
            data (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (g.num_faces): Flux vector.

        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def assemble_int_bound_flux(
        self,
        sd: pp.Grid,
        sd_data: dict,
        intf: pp.MortarGrid,
        intf_data: dict,
        cc: np.ndarray,
        matrix: np.ndarray,
        rhs: np.ndarray,
        self_ind: int,
        use_secondary_proj: bool = False,
    ) -> None:
        """Abstract method. Assemble the contribution from an internal
        boundary, manifested as a flux boundary condition.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
            sd_data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            intf_data (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                secondary side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """
        pass

    @abstractmethod
    def assemble_int_bound_source(
        self,
        sd: pp.Grid,
        sd_data: dict,
        intf: pp.MortarGrid,
        intf_data: dict,
        cc: np.ndarray,
        matrix: np.ndarray,
        rhs: np.ndarray,
        self_ind: int,
    ) -> None:
        """Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_intf (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                secondary side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        pass

    @abstractmethod
    def assemble_int_bound_pressure_trace(
        self,
        sd: pp.Grid,
        sd_data: dict,
        intf: pp.MortarGrid,
        intf_data: dict,
        cc: Optional[np.ndarray],
        matrix: np.ndarray,
        rhs: np.ndarray,
        self_ind: int,
        use_secondary_proj: bool = False,
        assemble_matrix=True,
        assemble_rhs=True,
    ) -> None:
        """Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
            sd_data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            intf_data (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                secondary side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """
        pass

    @abstractmethod
    def assemble_int_bound_pressure_trace_rhs(
        self, sd, sd_data, intf, intf_data, cc, rhs, self_ind, use_secondary_proj=False
    ):
        """Assemble the rhs contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        For details, see self.assemble_int_bound_pressure_trace()

        Args:
            sd (Grid): Grid which the condition should be imposed on.
            sd_data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            intf_data (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """

    @abstractmethod
    def assemble_int_bound_pressure_trace_between_interfaces(
        self,
        sd: pp.Grid,
        sd_data: dict,
        proj_primary: sps.spmatrix,
        proj_secondary: sps.spmatrix,
        cc: np.ndarray,
        matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
            sd_data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
           proj_primary (sparse matrix): Pressure projection from the higher-dim
                grid to the primary mortar grid.
           proj_secondary (sparse matrix): Flux projection from the secondary mortar
                grid to the main grid.
           cc (block matrix, 3x3): Block matrix of size 3 x 3, whwere each block
                represents coupling between variables on this interface. Index 0, 1 and
                2 represent the primary grid, the primary and secondary interface,
                respectively.
           matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
           rhs (block_array 3x1): Block matrix of size 3 x 1, representing the right
                hand side of this coupling. Index 0, 1 and 2 represent the primary grid,
                the primary and secondary interface, respectively.

        """
        pass

    @abstractmethod
    def assemble_int_bound_pressure_cell(
        self,
        sd: pp.Grid,
        sd_data: dict,
        intf: pp.MortarGrid,
        intf_data: dict,
        cc: np.ndarray,
        matrix: np.ndarray,
        rhs: np.ndarray,
        self_ind: int,
    ) -> None:
        """Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Args:
            sd (Grid): Grid which the condition should be imposed on.
            sd_data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            intf_data (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        pass

    @abstractmethod
    def enforce_neumann_int_bound(
        self,
        sd: pp.Grid,
        intf: pp.MortarGrid,
        intf_data: dict,
        matrix: np.ndarray,
        self_ind: int,
    ) -> None:
        """Enforce Neumann boundary conditions on a given system matrix.

        Methods based on a mixed variational form will need this function to
        implement essential boundary conditions.

        The discretization matrix should be modified in place.

        Args:
            sd (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.

        """
        pass
