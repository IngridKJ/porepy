# import numpy as np
# import scipy.sparse as sps
# import porepy as pp

# Nx = Ny = 6
# phys_dims = [1,1]
# g = pp.CartGrid([Nx, Ny], phys_dims)
# g.compute_geometry()

# pp.plot_grid(g, info = 'cfno',figsize=(15, 12))

# gb,_ = pp.grid_buckets_2d.two_intersecting([Nx,Ny],x_endpoints=[0.2,0.8],y_endpoints=[0.2,0.8],simplex=False)
# g = pp.CartGrid([Nx, Ny], [1, 1])
# g.compute_geometry()

# # pp.plot_grid(g, info = 'cfno',alpha=0, figsize=(15, 12))

# import numpy as np
# import scipy.sparse as sps

from typing import Dict, List

import numpy as np
import porepy as pp
import scipy.sparse as sps

node_var: str = "pressure"
edge_var: str = "interface flux"
param_key: str = "flow"

Nx = Ny = 10  

#Cartesian grid, one horizontal fracture
# gb,_ = pp.grid_buckets_2d.single_horizontal(
#       mesh_args=[Nx,Ny],x_endpoints = [0.2,0.8],simplex=False
# )
 
# Cartesian grid, two intersecting fractures
gb,_ = pp.grid_buckets_2d.two_intersecting(
        mesh_args=[Nx,Ny],x_endpoints=[0.2,0.8],y_endpoints=[0.2,0.8],simplex=False
)

# Simplex gris, one horizontal fracture
# gb,_ = pp.grids.standard_grids.grid_buckets_2d.single_horizontal(
#       mesh_args={'mesh_size_frac':0.15,}, simplex=True
# )

# Simplex grid, two intersecting fractures
# gb, domain = pp.grids.standard_grids.grid_buckets_2d.two_intersecting(
#       mesh_args={'mesh_size_frac':0.1,}, x_endpoints = [0.35,0.9], y_endpoints = [0.2,0.8], simplex=True
# )

# pp.plot_grid(gb,figsize=(15, 12))

for g, d in gb:
    # For subdomains, we need BCs and permeability
    bc_val: np.ndarray = np.zeros(g.num_faces)
    if g.dim == 1:
        source = np.ones(g.num_cells)
    else:
        source = np.zeros(g.num_cells)

    pp.initialize_data(
        g,
        d,
        param_key,
        {
            "bc_values": bc_val,
            "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
            "bc": pp.BoundaryCondition(g, faces=g.get_boundary_faces(), cond="dir"),
            'source': source
        },
    )
    
    # Initialise variable values
    pp.set_state(d, {node_var: np.zeros(g.num_cells)})
    pp.set_iterate(d, {node_var: np.zeros(g.num_cells)})
    # Define variable
    d[pp.PRIMARY_VARIABLES] = {node_var: {"cells": 1}}
for e, d in gb.edges():
    # Retrieve the mortar grid on the edge
    mg = d["mortar_grid"]
    # The single parameter is normal diffusivity (kappa / mu)
    pp.initialize_data(
        e,
        d,
        param_key,
        {
            "normal_diffusivity": np.ones(mg.num_cells),
        },
    )

    pp.set_state(d, {edge_var: np.zeros(mg.num_cells)})
    pp.set_iterate(d, {edge_var: np.zeros(mg.num_cells)})
    d[pp.PRIMARY_VARIABLES] = {edge_var: {"cells": 1}}


# Objects which handle degree-of-freedom bookkeeping and equation discretisation
# and assembly
dof_manager = pp.DofManager(gb)
eq_manager = pp.ad.EquationManager(gb, dof_manager)

# Define mixed-dimensional variables
grid_list: List = [g for g, _ in gb]
edge_list: List = [e for e, _ in gb.edges()]
pressure_var: pp.ad.MergedVariable = eq_manager.merge_variables(
    [(g, node_var) for g in grid_list]
)
flux_var: pp.ad.MergedVariable = eq_manager.merge_variables(
    [(e, edge_var) for e in edge_list]
)
# and projection operator between nodes and edges (subdomains and interfaces)
projection = pp.ad.MortarProjections(gb, grid_list, edge_list)

# Ad parameters,
bc_values = pp.ad.BoundaryCondition(param_key, grid_list)
source = pp.ad.ParameterArray('flow','source',grids=grid_list)
# operators
div = pp.ad.Divergence(grid_list)
# and discretizations
node_discretization = pp.ad.MpfaAd(param_key, grid_list)
edge_discretization = pp.ad.RobinCouplingAd(param_key, edge_list)

# Equation definition.
# For the subdomains, the flux has three contributions:
#     interior subdomain + exterior BC + internal boundaries
flux: pp.ad.Operator = (
    node_discretization.flux * pressure_var
    + node_discretization.bound_flux * bc_values
    + node_discretization.bound_flux * projection.mortar_to_primary_int * flux_var
)

node_equation: pp.ad.Operator = (
    div * flux - projection.mortar_to_secondary_int * flux_var - source
)


# The interface flux also consists of three terms according to Eq. 3.3.
# First we reconstruct the higher-dimensional pressure at the internal boundary,
# which has contributions from the internal (cell centre) pressure
# and the influence of the internal and external/global boundary conditions
p_h_bound: pp.ad.Operator = (
    node_discretization.bound_pressure_cell * pressure_var
    + node_discretization.bound_pressure_face
    * projection.mortar_to_primary_int
    * flux_var
    + node_discretization.bound_pressure_face * bc_values
)
# The other two terms are the lower-dimensional cell centre pressures and the
# interface flux variable, i.e. pressure_var and flux_var. All three terms are
# collected after the pressure terms have been projected to the interface.

# Note that mortar_discr is the inverse of kappa/mu in Eq. 3.3.
edge_equation: pp.ad.Operator = edge_discretization.mortar_scaling * (
    projection.primary_to_mortar_avg * p_h_bound
    - projection.secondary_to_mortar_avg * pressure_var
    + edge_discretization.mortar_discr * flux_var
)

# A general comment on equation specification: Sign conventions are not always
# obvious. One may have to consult the individual discretisation classes.

# Discretize, assemble and solve
# We are in the process of getting rid of Expressions:
interface_flow_eq = pp.ad.Expression(
    edge_equation, dof_manager, "interface flow equation", grid_order=edge_list
)
subdomain_flow_eq = pp.ad.Expression(
    node_equation, dof_manager, "flow equation on nodes", grid_order=grid_list
)
eq_manager.equations = [subdomain_flow_eq, interface_flow_eq]

eq_manager.discretize(gb)
A, b = eq_manager.assemble_matrix_rhs()
solution = sps.linalg.spsolve(A, b)
# Distribute the solution to the subdomains and interfaces for visualisation
assembler = pp.Assembler(gb, dof_manager)
assembler.distribute_variable(solution)

pp.plot_grid(gb, node_var)

